from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import build_mlp
from ..tools import scatter_sum

__all__ = ["MultipoleChargeHead", "LoremLongRangeReadout"]


class MultipoleChargeHead(nn.Module):
    """
    Build l<=2 latent multipole channels from CACE-LOREM short-range states.

    Output channel layout (per atom):
      [ monopole(1), dipole(3), quadrupole(5) ] -> total 9 channels.
    """

    def __init__(
        self,
        p_feature_key: str = "p_features",
        s_feature_key: str = "s_features",
        s_l1_feature_key: str = "s_l1_features",
        s_l2_feature_key: str = "s_l2_features",
        output_key: str = "q",
        scalar_hidden: int = 128,
        pair_hidden: int = 128,
    ):
        super().__init__()
        self.p_feature_key = p_feature_key
        self.s_feature_key = s_feature_key
        self.s_l1_feature_key = s_l1_feature_key
        self.s_l2_feature_key = s_l2_feature_key
        self.output_key = output_key
        self.model_outputs = [output_key]

        # Lazy input sizes: p/S feature dims depend on representation settings.
        self.scalar_head: Optional[nn.Module] = None
        self.l1_proj_head: Optional[nn.Linear] = None
        self.l2_proj_head: Optional[nn.Linear] = None
        # Per-branch positive scale (softplus, init ~1.0): LOREM-like linear readout, no tanh cap.
        _sp1 = torch.log(torch.expm1(torch.tensor(1.0)))
        self.l1_output_scale = nn.Parameter(_sp1.clone())
        self.l2_output_scale = nn.Parameter(_sp1.clone())
        self.scalar_hidden = scalar_hidden
        # Kept for backward-compatible signature in training scripts.
        self.pair_hidden = pair_hidden
        self._checked_l_order = False

    def _ensure_heads(
        self,
        p_dim: int,
        l1_nr: int,
        l1_a: int,
        l1_c: int,
        l2_nr: int,
        l2_a: int,
        l2_c: int,
        device: torch.device,
    ):
        if self.scalar_head is None:
            self.scalar_head = build_mlp(
                n_in=p_dim,
                n_out=1,
                n_hidden=[self.scalar_hidden, self.scalar_hidden // 2],
                n_layers=3,
                bias=True,
            ).to(device)

        if self.l1_proj_head is None and l1_nr > 0 and l1_a > 0 and l1_c > 0:
            # Linear projection readout: flatten l=1 tensor block to dipole channels.
            self.l1_proj_head = nn.Linear(l1_nr * l1_a * l1_c, l1_a, bias=True).to(device)
            nn.init.xavier_uniform_(self.l1_proj_head.weight, gain=1.0)
            nn.init.zeros_(self.l1_proj_head.bias)
        if self.l2_proj_head is None and l2_nr > 0 and l2_a > 0 and l2_c > 0:
            # Linear projection readout: flatten l=2 tensor block to quadrupole channels.
            self.l2_proj_head = nn.Linear(l2_nr * l2_a * l2_c, l2_a, bias=True).to(device)
            nn.init.xavier_uniform_(self.l2_proj_head.weight, gain=1.0)
            nn.init.zeros_(self.l2_proj_head.bias)

    @staticmethod
    def _quad6_to_quad5(q6: torch.Tensor) -> torch.Tensor:
        """
        Fixed projection from 6D Cartesian symmetric components to 5D traceless quadrupole.

        Assumed l=2 order in grouped Cartesian basis:
          [xx, xy, xz, yy, yz, zz]
        Output 5D basis:
          [xx-yy, 2zz-xx-yy, xy, xz, yz]
        """
        if q6.shape[-1] != 6:
            raise ValueError(f"Expected 6 l=2 Cartesian components before detrace, got {q6.shape[-1]}")
        q_xx = q6[:, 0]
        q_xy = q6[:, 1]
        q_xz = q6[:, 2]
        q_yy = q6[:, 3]
        q_yz = q6[:, 4]
        q_zz = q6[:, 5]
        return torch.stack(
            [
                q_xx - q_yy,
                2.0 * q_zz - q_xx - q_yy,
                q_xy,
                q_xz,
                q_yz,
            ],
            dim=-1,
        )

    def _assert_l_order(self, data: Dict[str, torch.Tensor]) -> None:
        if self._checked_l_order:
            return
        if "s_l1_lxlylz" not in data or "s_l2_lxlylz" not in data:
            # Keep backward compatibility for old checkpoints/pipelines.
            return

        expected_l1 = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=torch.long,
            device=data["s_l1_lxlylz"].device,
        )
        expected_l2 = torch.tensor(
            [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]],
            dtype=torch.long,
            device=data["s_l2_lxlylz"].device,
        )

        got_l1 = data["s_l1_lxlylz"].to(dtype=torch.long)
        got_l2 = data["s_l2_lxlylz"].to(dtype=torch.long)

        ok_l1 = got_l1.shape == expected_l1.shape and torch.equal(got_l1, expected_l1)
        ok_l2 = got_l2.shape == expected_l2.shape and torch.equal(got_l2, expected_l2)
        if not (ok_l1 and ok_l2):
            raise ValueError(
                "Detected unexpected l-block basis ordering for l=1/l=2. "
                "Current q2 detrace projection assumes l2 order "
                "[xx, xy, xz, yy, yz, zz] i.e. "
                "[(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]. "
                f"Got l1={got_l1.tolist()}, l2={got_l2.tolist()}."
            )
        self._checked_l_order = True

    def forward(self, data: Dict[str, torch.Tensor], training: bool = False, output_index: int = None):
        def _take_last_step(x: torch.Tensor) -> torch.Tensor:
            return x[..., -1] if x.dim() in (3, 5) else x

        self._assert_l_order(data)

        p_all = data[self.p_feature_key]
        p = _take_last_step(p_all)

        n_nodes = p.shape[0]
        p_dim = p.shape[1]

        if self.s_l1_feature_key in data and self.s_l2_feature_key in data:
            s_l1 = _take_last_step(data[self.s_l1_feature_key])  # [n, nr, a1, c]
            s_l2 = _take_last_step(data[self.s_l2_feature_key])  # [n, nr, a2, c]
        else:
            # Backward-compatible fallback when grouped S is unavailable.
            s_all = _take_last_step(data[self.s_feature_key])
            s_l1 = s_all[:, :, 1:4, :] if s_all.shape[2] >= 4 else s_all[:, :, :0, :]
            s_l2 = s_all[:, :, 4:10, :] if s_all.shape[2] >= 10 else s_all[:, :, :0, :]

        l1_nr = s_l1.shape[1] if s_l1.numel() > 0 else 0
        l1_a = s_l1.shape[2] if s_l1.numel() > 0 else 0
        l1_c = s_l1.shape[3] if s_l1.numel() > 0 else 0
        l2_nr = s_l2.shape[1] if s_l2.numel() > 0 else 0
        l2_a = s_l2.shape[2] if s_l2.numel() > 0 else 0
        l2_c = s_l2.shape[3] if s_l2.numel() > 0 else 0

        self._ensure_heads(
            p_dim=p_dim,
            l1_nr=l1_nr,
            l1_a=l1_a,
            l1_c=l1_c,
            l2_nr=l2_nr,
            l2_a=l2_a,
            l2_c=l2_c,
            device=p.device,
        )

        # Monopole from scalar node state P.
        q0 = self.scalar_head(p)
        if self.l1_proj_head is not None and l1_nr > 0 and l1_c > 0:
            # LOREM-like direct linear readout from l=1 block (no scalar gate).
            l1_flat = s_l1.reshape(n_nodes, -1)
            q1 = self.l1_proj_head(l1_flat)
            q1 = F.softplus(self.l1_output_scale) * q1
            if l1_a != 3:
                raise ValueError(f"Expected l=1 angular size 3, got {l1_a}")
        else:
            q1 = p.new_zeros((n_nodes, 3))

        if self.l2_proj_head is not None and l2_nr > 0 and l2_c > 0:
            # LOREM-like direct linear readout from l=2 block (before detrace).
            l2_flat = s_l2.reshape(n_nodes, -1)
            q2_raw6 = self.l2_proj_head(l2_flat)
            q2_raw6 = F.softplus(self.l2_output_scale) * q2_raw6
            if l2_a != 6:
                raise ValueError(f"Expected l=2 angular size 6 before detrace, got {l2_a}")
            q2 = self._quad6_to_quad5(q2_raw6)
        else:
            q2 = p.new_zeros((n_nodes, 5))

        data[self.output_key] = torch.cat([q0, q1, q2], dim=-1)
        data[self.output_key + "_monopole"] = q0
        data[self.output_key + "_dipole"] = q1
        data[self.output_key + "_quadrupole"] = q2
        return data


class LoremLongRangeReadout(nn.Module):
    """
    LOREM-style LR feedback:
      potentials from charge channels -> interact with S -> scalarize -> concat with monopole potential
      -> per-atom LR energy -> graph energy.
    """

    def __init__(
        self,
        s_feature_key: str = "s_features",
        s_l0_feature_key: str = "s_l0_features",
        s_l1_feature_key: str = "s_l1_features",
        s_l2_feature_key: str = "s_l2_features",
        field_key: str = "q_field",
        output_key: str = "lr_energy",
        per_atom_output_key: str = "lr_energy_atom",
        aggregation_mode: str = "sum",
        hidden: int = 128,
    ):
        super().__init__()
        self.s_feature_key = s_feature_key
        self.s_l0_feature_key = s_l0_feature_key
        self.s_l1_feature_key = s_l1_feature_key
        self.s_l2_feature_key = s_l2_feature_key
        self.field_key = field_key
        self.output_key = output_key
        self.per_atom_output_key = per_atom_output_key
        self.aggregation_mode = aggregation_mode
        self.hidden = hidden

        self.model_outputs = [output_key]
        if per_atom_output_key is not None:
            self.model_outputs.append(per_atom_output_key)

        # Lazy heads because S shape depends on representation settings.
        self.dipole_to_channel: Optional[nn.Linear] = None
        self.quadrupole_to_channel: Optional[nn.Linear] = None
        self.energy_head: Optional[nn.Module] = None
        # Directional branch scales (start conservative for stability).
        self.l1_dir_scale = nn.Parameter(torch.tensor(0.1))
        self.l2_dir_scale = nn.Parameter(torch.tensor(0.1))

    def _ensure_heads(self, s_channel_dim: int, lr_input_dim: int, device: torch.device):
        if self.dipole_to_channel is None:
            self.dipole_to_channel = nn.Linear(3, s_channel_dim, bias=False).to(device)
        if self.quadrupole_to_channel is None:
            self.quadrupole_to_channel = nn.Linear(5, s_channel_dim, bias=False).to(device)
        if self.energy_head is None:
            self.energy_head = build_mlp(
                n_in=lr_input_dim,
                n_out=1,
                n_hidden=[self.hidden, self.hidden // 2],
                n_layers=3,
                bias=True,
            ).to(device)

    def forward(self, data: Dict[str, torch.Tensor], training: bool = False, output_index: int = None):
        def _take_last_step(x: torch.Tensor) -> torch.Tensor:
            return x[..., -1] if x.dim() == 5 else x

        has_grouped_s = (
            self.s_l0_feature_key in data
            and self.s_l1_feature_key in data
            and self.s_l2_feature_key in data
        )
        if has_grouped_s:
            s_l0 = _take_last_step(data[self.s_l0_feature_key])  # [n, nr, a0, c]
            s_l1 = _take_last_step(data[self.s_l1_feature_key])  # [n, nr, a1, c]
            s_l2 = _take_last_step(data[self.s_l2_feature_key])  # [n, nr, a2, c]
        else:
            # Backward-compatible fallback: no grouped S available.
            s = _take_last_step(data[self.s_feature_key])
            s_l0, s_l1, s_l2 = s, s[:, :, :0, :], s[:, :, :0, :]

        field = data[self.field_key]  # [n_nodes, 1+3+5]
        mono_potential = field[:, :1]
        dipole_potential = field[:, 1:4]
        quadrupole_potential = field[:, 4:9]

        s_channel_dim = s_l0.shape[-1]
        n_radial = s_l0.shape[1]
        l1_a = s_l1.shape[2] if s_l1.dim() == 4 else 0
        l2_a = s_l2.shape[2] if s_l2.dim() == 4 else 0
        # Parallel features:
        # - norm branch: l0/l1/l2 -> scalar invariants [3 * nr * c]
        # - directional branch: preserve angular signed components [nr * (a1 + a2)]
        lr_input_dim = 1 + 3 * n_radial * s_channel_dim + n_radial * (l1_a + l2_a)
        self._ensure_heads(
            s_channel_dim=s_channel_dim,
            lr_input_dim=lr_input_dim,
            device=s_l0.device,
        )

        # l=0,1,2 grouped "tensor-like" coupling in Cartesian basis:
        # - l0 block gated by monopole potential
        # - l1 block gated by dipole potential
        # - l2 block gated by quadrupole potential
        mono_gate = mono_potential.unsqueeze(1).unsqueeze(1)  # [n,1,1,1]
        dip_gate = self.dipole_to_channel(dipole_potential).unsqueeze(1).unsqueeze(1)  # [n,1,1,c]
        quad_gate = self.quadrupole_to_channel(quadrupole_potential).unsqueeze(1).unsqueeze(1)  # [n,1,1,c]
        s0_updates = s_l0 * mono_gate
        s1_updates = s_l1 * dip_gate
        s2_updates = s_l2 * quad_gate

        # S^(l) -> scalar invariants via per-l norm over angular index.
        # Shapes: [n, nr, c]
        s0_norm = torch.sqrt(torch.clamp((s0_updates * s0_updates).sum(dim=2), min=1e-12))
        s1_norm = torch.sqrt(torch.clamp((s1_updates * s1_updates).sum(dim=2), min=1e-12))
        s2_norm = torch.sqrt(torch.clamp((s2_updates * s2_updates).sum(dim=2), min=1e-12))
        s0_scalar = s0_norm.reshape(s0_norm.shape[0], -1)
        s1_scalar = s1_norm.reshape(s1_norm.shape[0], -1)
        s2_scalar = s2_norm.reshape(s2_norm.shape[0], -1)

        # Keep directional angular components in parallel with norm branch.
        # Reduce channel only, preserve angular signs.
        if l1_a > 0:
            s1_dir = s1_updates.mean(dim=-1).reshape(s1_updates.shape[0], -1)
            s1_dir = torch.tanh(self.l1_dir_scale) * s1_dir
        else:
            s1_dir = s1_scalar[:, :0]
        if l2_a > 0:
            s2_dir = s2_updates.mean(dim=-1).reshape(s2_updates.shape[0], -1)
            s2_dir = torch.tanh(self.l2_dir_scale) * s2_dir
        else:
            s2_dir = s2_scalar[:, :0]

        lr_features = torch.cat(
            [mono_potential, s0_scalar, s1_scalar, s2_scalar, s1_dir, s2_dir], dim=-1
        )
        lr_atom = self.energy_head(lr_features)

        if self.per_atom_output_key is not None:
            data[self.per_atom_output_key] = lr_atom

        if data["batch"] is None:
            batch = torch.zeros(lr_atom.shape[0], dtype=torch.int64, device=lr_atom.device)
        else:
            batch = data["batch"]
        lr_energy = scatter_sum(src=lr_atom, index=batch, dim=0).squeeze(-1)
        if self.aggregation_mode == "avg":
            lr_energy = lr_energy / torch.bincount(batch)

        data[self.output_key] = lr_energy
        return data
