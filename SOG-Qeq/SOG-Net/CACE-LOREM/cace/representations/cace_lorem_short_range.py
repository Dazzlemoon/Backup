import torch
from torch import nn
from typing import Callable, Dict, Sequence, Optional, List, Any

from ..tools import elementwise_multiply_3tensors, scatter_sum
from ..modules import (
    NodeEncoder,
    NodeEmbedding,
    EdgeEncoder,
    AngularComponent,
    SharedRadialLinearTransform,
    Symmetrizer,
    MessageAr,
    MessageBchi,
    NodeMemory,
    get_edge_vectors_and_lengths,
)
from ..modules.blocks import build_mlp

__all__ = ["CaceLoremShortRange"]


class CaceLoremShortRange(nn.Module):
    """
    LOREM-style short-range flow implemented on top of CACE building blocks.

    Mapping of symbols:
      - Y: fixed edge angular basis from Cartesian monomials.
      - S: orientation-dependent node tensor (A-basis), updated by MP.
      - P: scalar node state, updated from angular norms of S.
    """

    def __init__(
        self,
        zs: Sequence[int],
        n_atom_basis: int,
        cutoff: float,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        max_l: int,
        max_nu: int,
        num_message_passing: int,
        n_scalar_features: int = 128,
        node_encoder: Optional[nn.Module] = None,
        edge_encoder: Optional[nn.Module] = None,
        type_message_passing: List[str] = ["M", "Ar", "Bchi"],
        args_message_passing: Dict[str, Any] = {"M": {}, "Ar": {}, "Bchi": {}},
        embed_receiver_nodes: bool = False,
        atom_embedding_random_seed: List[int] = [42, 42],
        n_radial_basis: Optional[int] = None,
        avg_num_neighbors: float = 10.0,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        timeit: bool = False,
        forward_features: List[str] = [],
    ):
        super().__init__()
        self.zs = zs
        self.nz = len(zs)
        self.n_atom_basis = n_atom_basis
        self.cutoff = cutoff
        self.max_l = max_l
        self.max_nu = max_nu
        self.num_message_passing = num_message_passing
        self.n_scalar_features = n_scalar_features
        self.mp_norm_factor = 1.0 / (avg_num_neighbors) ** 0.5
        self.timeit = timeit
        self.forward_features = forward_features
        self.device = device

        if node_encoder is None:
            self.node_onehot = NodeEncoder(self.zs)
            self.nz = len(zs)
        else:
            self.node_onehot = node_encoder
            self.nz = node_encoder.embedding_dim

        self.node_embedding_sender = NodeEmbedding(
            node_dim=self.nz,
            embedding_dim=self.n_atom_basis,
            random_seed=atom_embedding_random_seed[0],
        )
        if embed_receiver_nodes:
            self.node_embedding_receiver = NodeEmbedding(
                node_dim=self.nz,
                embedding_dim=self.n_atom_basis,
                random_seed=atom_embedding_random_seed[1],
            )
        else:
            self.node_embedding_receiver = self.node_embedding_sender

        if edge_encoder is not None:
            self.edge_coding = edge_encoder
        else:
            self.edge_coding = EdgeEncoder(directed=True)

        self.n_edge_channels = n_atom_basis**2
        self.radial_basis = radial_basis
        self.n_radial_func = self.radial_basis.n_rbf
        self.n_radial_basis = n_radial_basis or self.radial_basis.n_rbf
        self.cutoff_fn = cutoff_fn
        self.angular_basis = AngularComponent(self.max_l)
        self.radial_transform = SharedRadialLinearTransform(
            max_l=self.max_l,
            radial_dim=self.n_radial_func,
            radial_embedding_dim=self.n_radial_basis,
            channel_dim=self.n_edge_channels,
        )
        self.l_list = self.angular_basis.get_lxlylz_list()
        self.symmetrizer = Symmetrizer(self.max_nu, self.max_l, self.l_list)
        self._register_l_group_indices()

        self.message_passing_list = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        NodeMemory(
                            max_l=self.max_l,
                            radial_embedding_dim=self.n_radial_basis,
                            channel_dim=self.n_edge_channels,
                            **args_message_passing["M"] if "M" in args_message_passing else {},
                        )
                        if "M" in type_message_passing
                        else None,
                        MessageAr(
                            cutoff=cutoff,
                            max_l=self.max_l,
                            radial_embedding_dim=self.n_radial_basis,
                            channel_dim=self.n_edge_channels,
                            **args_message_passing["Ar"] if "Ar" in args_message_passing else {},
                        )
                        if "Ar" in type_message_passing
                        else None,
                        MessageBchi(
                            lxlylz_index=self.angular_basis.get_lxlylz_index(),
                            **args_message_passing["Bchi"] if "Bchi" in args_message_passing else {},
                        )
                        if "Bchi" in type_message_passing
                        else None,
                    ]
                )
                for _ in range(self.num_message_passing)
            ]
        )

        self.scalar_init = nn.Linear(self.n_atom_basis, self.n_scalar_features, bias=True)
        self.scalar_update = build_mlp(
            n_in=self.n_radial_basis * self.n_edge_channels,
            n_out=self.n_scalar_features,
            n_hidden=[2 * self.n_scalar_features, self.n_scalar_features],
            n_layers=3,
            bias=True,
        )
        self.scalar_norm = nn.LayerNorm(self.n_scalar_features)

    def _register_l_group_indices(self) -> None:
        idx_by_l = {0: [], 1: [], 2: []}
        for idx, lxlylz in enumerate(self.l_list):
            l_now = int(sum(lxlylz))
            if l_now in idx_by_l:
                idx_by_l[l_now].append(idx)

        for l_now in (0, 1, 2):
            if len(idx_by_l[l_now]) == 0:
                index_tensor = torch.empty(0, dtype=torch.long)
            else:
                index_tensor = torch.tensor(idx_by_l[l_now], dtype=torch.long)
            self.register_buffer(f"l{l_now}_indices", index_tensor, persistent=False)

        # Keep explicit angular basis order metadata for runtime self-checks in LR head.
        for l_now in (1, 2):
            indices = idx_by_l[l_now]
            if len(indices) == 0:
                lxlylz_tensor = torch.empty((0, 3), dtype=torch.long)
            else:
                lxlylz_tensor = torch.tensor([self.l_list[i] for i in indices], dtype=torch.long)
            self.register_buffer(f"l{l_now}_lxlylz", lxlylz_tensor, persistent=False)

    @staticmethod
    def _gather_angular_groups(
        s_tensor: torch.Tensor,
        l0_indices: torch.Tensor,
        l1_indices: torch.Tensor,
        l2_indices: torch.Tensor,
    ):
        # s_tensor shape: [n_nodes, n_radial, n_angular, n_channel, n_steps]
        def gather_by_indices(indices: torch.Tensor) -> torch.Tensor:
            if indices.numel() == 0:
                return s_tensor[:, :, :0, :, :]
            return s_tensor.index_select(dim=2, index=indices)

        return (
            gather_by_indices(l0_indices),
            gather_by_indices(l1_indices),
            gather_by_indices(l2_indices),
        )

    def _update_p_from_s(self, p_now: torch.Tensor, s_now: torch.Tensor) -> torch.Tensor:
        # S has shape [n_nodes, n_radial_basis, n_angular, n_channels]
        # We compress angular channels via norm to mimic spherical-norm-to-scalar update.
        s_norm = torch.sqrt(torch.clamp((s_now * s_now).sum(dim=2), min=1e-12))
        s_features = s_norm.reshape(s_norm.shape[0], -1)
        delta = self.scalar_update(s_features)
        return self.scalar_norm(p_now + delta)

    def forward(self, data: Dict[str, torch.Tensor]):
        n_nodes = data["positions"].shape[0]
        if data["batch"] is None:
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=self.device)
        else:
            batch_now = data["batch"]

        node_one_hot = self.node_onehot(data["atomic_numbers"])
        node_embedded_sender = self.node_embedding_sender(node_one_hot)
        node_embedded_receiver = self.node_embedding_receiver(node_one_hot)
        encoded_edges = self.edge_coding(
            edge_index=data["edge_index"],
            node_type=node_embedded_sender,
            node_type_2=node_embedded_receiver,
            data=data,
        )

        edge_vectors, edge_lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
            normalize=True,
        )
        radial_component = self.radial_basis(edge_lengths)
        radial_cutoff = self.cutoff_fn(edge_lengths)
        angular_component = self.angular_basis(edge_vectors)  # Y (fixed angular basis)

        edge_attri = elementwise_multiply_3tensors(
            radial_component * radial_cutoff,
            angular_component,
            encoded_edges,
        )

        s_now = scatter_sum(
            src=edge_attri,
            index=data["edge_index"][1],
            dim=0,
            dim_size=n_nodes,
        )
        s_now = self.radial_transform(s_now)
        p_now = self.scalar_init(node_embedded_sender)
        p_now = self._update_p_from_s(p_now, s_now)

        p_list = [p_now]
        b_list = [self.symmetrizer(node_attr=s_now)]
        s_list = [s_now]

        for nm, mp_Ar, mp_Bchi in self.message_passing_list:
            if nm is not None:
                memory_now = nm(node_feat=s_now)
            else:
                memory_now = 0.0

            if mp_Bchi is not None:
                message_Bchi = mp_Bchi(
                    node_feat=b_list[-1],
                    edge_attri=edge_attri,
                    edge_index=data["edge_index"],
                )
                s_from_bchi = scatter_sum(
                    src=message_Bchi,
                    index=data["edge_index"][1],
                    dim=0,
                    dim_size=n_nodes,
                )
                s_from_bchi = self.radial_transform(s_from_bchi)
            else:
                s_from_bchi = 0.0

            if mp_Ar is not None:
                message_Ar = mp_Ar(
                    node_feat=s_now,
                    edge_lengths=edge_lengths,
                    radial_cutoff_fn=radial_cutoff,
                    edge_index=data["edge_index"],
                )
                s_from_ar = scatter_sum(
                    src=message_Ar,
                    index=data["edge_index"][1],
                    dim=0,
                    dim_size=n_nodes,
                )
            else:
                s_from_ar = 0.0

            s_now = (s_from_ar + s_from_bchi) * self.mp_norm_factor + memory_now
            b_now = self.symmetrizer(node_attr=s_now)
            p_now = self._update_p_from_s(p_now, s_now)

            s_list.append(s_now)
            b_list.append(b_now)
            p_list.append(p_now)

        node_feats_out = torch.stack(p_list, dim=-1)
        s_out = torch.stack(s_list, dim=-1)
        b_out = torch.stack(b_list, dim=-1)
        s_l0_out, s_l1_out, s_l2_out = self._gather_angular_groups(
            s_out, self.l0_indices, self.l1_indices, self.l2_indices
        )

        try:
            displacement = data["displacement"]
        except Exception:
            displacement = None

        output = {
            "positions": data["positions"],
            "cell": data["cell"],
            "displacement": displacement,
            "batch": batch_now,
            "edge_index": data["edge_index"],
            "shifts": data["shifts"],
            # Keep compatibility: Atomwise defaults to feature_key="node_feats".
            "node_feats": node_feats_out,
            # Expose LOREM-style internal states (S/P-focused).
            "s_features": s_out,
            "s_l0_features": s_l0_out,
            "s_l1_features": s_l1_out,
            "s_l2_features": s_l2_out,
            # Runtime-only metadata to validate l-block component ordering.
            "s_l1_lxlylz": self.l1_lxlylz,
            "s_l2_lxlylz": self.l2_lxlylz,
            "p_features": node_feats_out,
            "b_features": b_out,
        }

        if hasattr(self, "forward_features") and len(self.forward_features) > 0:
            for key in self.forward_features:
                if key in data:
                    output[key] = data[key]

        return output
