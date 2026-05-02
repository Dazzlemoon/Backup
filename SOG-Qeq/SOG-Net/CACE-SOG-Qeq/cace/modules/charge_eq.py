from typing import Dict, List, Optional
import math

import torch
import torch.nn as nn

from .ewald import EwaldPotential

__all__ = ["ChargeEq"]


class ChargeEq(nn.Module):
    def __init__(
        self,
        dl: float = 1.5,
        sigma: float = 1.0,
        elements: List[int] = None,
        feature_key: str = "chi",
        output_key: str = "q_eq",
        ewald_key: str = "ewald_potential",
        system_charge: float = 0.0,
        remove_self_interaction: bool = True,
        aggregation_mode: str = "sum",
        compute_field: bool = True,
        norm_factor: float = (1.0 / 90.0474) ** 0.5,
        scaling_factor: float = 1.0,
        system_charge_key: str = "system_charge",
        # 可选：用 SOG 近似 1/r 构造 A，而不是直接从 EwaldPotential 取 A
        use_sog_kernel: bool = False,
        sog_num_components: int = 4,
        # 可选：与 SOGPotential 共用同一套 (wl, sl)，训练中不调用 SOGPotential.forward
        shared_sog_potential: Optional[nn.Module] = None,
    ):
        """
        Qeq (charge equilibration) 模块：
        - 从上游网络给出的 per-atom 电负性特征 chi 出发
        - 在总电荷约束下解线性方程得到平衡电荷 q_eq
        - 同时基于 EwaldPotential 构造的核矩阵 A 计算长程能量
        """
        super().__init__()

        self.feature_key = feature_key
        self.output_key = output_key
        self.ewald_key = ewald_key
        # 额外输出：
        # - J_elem: 每种元素的硬度参数（按 self.elements 顺序）
        # - J_i: 每个原子的硬度参数（按原子类型从 J_elem 映射）
        self.model_outputs = [output_key, ewald_key, "J_elem", "J_i"]
        self.normalization_factor = norm_factor
        self.scaling_factor = scaling_factor
        self.compute_field = compute_field
        self.system_charge = system_charge
        self.aggregation_mode = aggregation_mode
        self.system_charge_key = system_charge_key
        self.use_sog_kernel = use_sog_kernel
        self.shared_sog_potential = shared_sog_potential

        self.ep = EwaldPotential(
            dl=dl,
            sigma=sigma,
            remove_self_interaction=remove_self_interaction,
            # 重要：使子模块的输出 key 与 ChargeEq 的 ewald_key 一致。
            # 否则 NeuralNetworkPotential.collect_outputs() 会递归收集到 ep.model_outputs=['ewald_potential']，
            # 进而在 extract_outputs 时强制要求 data['ewald_potential'] 存在（即使训练流程从不调用 ep.forward）。
            output_key=ewald_key,
            aggregation_mode=aggregation_mode,
        )
        self.elements = elements
        Z_max = max(elements)
        Z_index_map = torch.full((Z_max + 1,), -1)
        for i, z in enumerate(elements):
            Z_index_map[z] = i
        self.register_buffer("Z_index_map", Z_index_map)

        # 元素硬度 J_i（按元素存储，训练参数）
        init_J = torch.ones(len(elements))
        self.J_raw = nn.Parameter(data=init_J, requires_grad=True)

        # 可选：SOG 核参数。若 shared_sog_potential 已提供，则用其 wl/sl，不在此处新建参数
        if self.use_sog_kernel and self.shared_sog_potential is None:
            # 初始化一组覆盖不同 length scale 的宽度，单位与坐标一致
            # 用 log-parameterization 保证 alpha > 0
            init_sigmas = torch.linspace(0.5, 5.0, sog_num_components)
            init_alphas = 1.0 / (init_sigmas ** 2 + 1e-6)
            self.sog_log_alpha = nn.Parameter(torch.log(init_alphas))
            # 初始化权重为接近库仑核的衰减（粗略均匀）
            self.sog_weights = nn.Parameter(torch.ones(sog_num_components) / sog_num_components)

    def init_sog_from_bsa(
        self,
        r_cut: float,
        b: float = 2.0,
    ) -> None:
        """
        用 BSA（双边级数近似）近似 1/r 核得到的实空间高斯和参数，覆盖当前 SOG 初值，
        与 Ji 等 2026 文中 Section II.C 一致（实空间形式）。
        仅当 use_sog_kernel=True 且 shared_sog_potential is None 时有效。
        """
        if not self.use_sog_kernel or self.shared_sog_potential is not None:
            return
        # 文中/补充材料：s = r_cut / r0，r0 = 1.9892536839080267 (b=2 时)
        r0 = 1.9892536839080267
        s = r_cut / r0
        M = self.sog_weights.numel()
        device = self.sog_weights.device
        dtype = self.sog_weights.dtype
        ell = torch.arange(M, device=device, dtype=dtype)
        # alpha_l = 1 / (2 b^{2l} s^2)
        alphas = 1.0 / (2.0 * (b ** (2 * ell)) * (s ** 2) + 1e-12)
        # weights_l = (2 log b) / sqrt(2 pi s^2) * b^{-l}
        coef = (2.0 * math.log(b)) / math.sqrt(2.0 * math.pi * s * s)
        weights = coef * (b ** (-ell))
        self.sog_log_alpha.data.copy_(torch.log(alphas).to(device=device, dtype=dtype))
        self.sog_weights.data.copy_(weights.to(device=device, dtype=dtype))

    def forward(self, data: Dict[str, torch.Tensor], **kwargs):
        if data["batch"] is None:
            n_nodes = data["positions"].shape[0]
            batch_now = torch.zeros(
                n_nodes, dtype=torch.int64, device=data["positions"].device
            )
        else:
            batch_now = data["batch"]

        box = data["cell"]
        r = data["positions"]
        chi = data[self.feature_key]
        # 兼容不同数据管线：有的 batch 用 atomic_numbers，有的用 z
        if "atomic_numbers" in data:
            Z = data["atomic_numbers"]
        elif "z" in data:
            Z = data["z"]
        else:
            raise KeyError("Missing atomic numbers: expected 'atomic_numbers' or 'z' in data.")
        element_types = torch.unique(Z)
        # 允许当前 batch 只包含部分元素（例如某些帧里没有 Au），
        # 只要它们都是 ChargeEq.elements 的子集即可。
        assert all(
            int(z.item()) in self.elements for z in element_types
        ), (
            f"Found unknown element types {element_types.tolist()} "
            f"not included in ChargeEq.elements={self.elements}."
        )
        if chi.dim() == 1:
            chi = chi.unsqueeze(1)

        J_raw = self.J_raw
        J_elem = torch.square(J_raw)  # 保证正数
        idx = self.Z_index_map[Z]
        J_i = J_elem[idx]
        # 便于在推理/分析阶段直接读取
        data["J_elem"] = J_elem
        data["J_i"] = J_i.unsqueeze(1) if J_i.dim() == 1 else J_i

        n, d = r.shape
        assert d == 3, "r dimension error"
        assert n == chi.size(0), "chi dimension error"

        unique_batches = torch.unique(batch_now)

        if (
            self.system_charge_key not in data
            or data[self.system_charge_key] is None
        ) and self.system_charge is not None:
            system_charge = torch.full(
                (len(unique_batches),),
                self.system_charge,
                device=data["positions"].device,
            )
        else:
            system_charge = data[self.system_charge_key]

        results = []
        ewald_results = []

        for i in unique_batches:
            mask = batch_now == i
            r_now, chi_now, box_now = r[mask], chi[mask], box[i]
            system_charge_now = system_charge[i]
            J_i_now = J_i[mask]

            # 构造核矩阵 A：
            # - 默认：从 EwaldPotential 取出等效的 A；
            # - 可选：用 SOG 高斯和近似 1/r 构造 A_sog（可学习）。
            if self.use_sog_kernel:
                A_now = self._build_A_sog(r_now, box_now)
            else:
                # 第二个返回量在本实现中就是 A 矩阵
                _, A_now = self.ep.compute_potential_triclinic(
                    r_now,
                    torch.eye(r_now.size(0), device=r_now.device),
                    box_now,
                    compute_field=self.compute_field,
                )
            q_eq, lambda_eq = self._compute_q_eq(
                A_now, chi_now, J_i_now, system_charge_now
            )
            results.append(q_eq)
            ewald_energy = 0.5 * q_eq.unsqueeze(1).T @ A_now @ q_eq.unsqueeze(1)
            ewald_results.append(ewald_energy)

        all_q_eq = torch.cat(results, dim=0)
        if all_q_eq.dim() == 1:
            all_q_eq = all_q_eq.unsqueeze(1)
        data[self.output_key] = all_q_eq

        all_ewald = (
            torch.stack(ewald_results, dim=0).sum(axis=1)
            if self.aggregation_mode == "sum"
            else torch.stack(ewald_results, dim=0)
        )
        if all_ewald.dim() != 1:
            all_ewald = all_ewald.squeeze(-1)
        data[self.ewald_key] = all_ewald

        return data

    def _build_A_sog(self, r: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """
        用 SOG（高斯和）构造核矩阵 A_sog。
        - 若 shared_sog_potential 已设置，则使用其 wl/sl，与 SOGPotential 的实空间核一致：K(r)=sum_l wl*exp(-r^2/sl^2)。
        - 否则使用本模块的 sog_log_alpha / sog_weights：K(r)=sum_l weights*exp(-alpha*r^2)。
        当前为显式构造 A（O(N^2)），后续可改为算子+PCG。
        """
        device, dtype = r.device, r.dtype
        N = r.size(0)
        diff = r.unsqueeze(0) - r.unsqueeze(1)  # [N, N, 3]
        dist = torch.linalg.norm(diff, dim=-1) + 1e-8  # [N, N]
        r2 = dist ** 2  # [N, N]

        if self.shared_sog_potential is not None:
            # 与 SOGPotential 共用 (wl, sl)，核形式与 compute_potential_SOG_realspace 一致
            wl = self.shared_sog_potential.wl.to(device=device, dtype=dtype)   # [L]
            sl = self.shared_sog_potential.sl.to(device=device, dtype=dtype)   # [L]
            min_term = -1.0 / (sl ** 2 + 1e-12)  # [L]
            sog_terms = torch.exp(r2.unsqueeze(0) * min_term.view(-1, 1, 1))  # [L, N, N]
            A_sog = (wl.view(-1, 1, 1) * sog_terms).sum(dim=0)
        else:
            alphas = torch.exp(self.sog_log_alpha).to(device=device, dtype=dtype)  # [L]
            weights = self.sog_weights.to(device=device, dtype=dtype)  # [L]
            sog_terms = torch.exp(-alphas.view(-1, 1, 1) * r2.unsqueeze(0))  # [L, N, N]
            A_sog = (weights.view(-1, 1, 1) * sog_terms).sum(dim=0)

        return A_sog.to(dtype=dtype)

    def _compute_q_eq(
        self,
        A_mat: torch.Tensor,
        chi: torch.Tensor,
        J: torch.Tensor,
        system_Q: torch.Tensor,
    ):
        device, dtype = A_mat.device, A_mat.dtype
        N_atoms = A_mat.size(0)
        A_plus_J = A_mat + torch.diag(J.to(dtype))
        coeffs = torch.ones((N_atoms + 1, N_atoms + 1), device=device, dtype=dtype)
        coeffs[:N_atoms, :N_atoms] = A_plus_J
        coeffs[N_atoms, N_atoms] = 0.0
        Q_tot = system_Q / self.normalization_factor
        chi_vector = torch.cat(
            [-chi.view(-1), torch.tensor([Q_tot], device=device, dtype=dtype)]
        )
        chi_vector = chi.unsqueeze(1) if chi.dim() == 1 else chi_vector
        sol = torch.linalg.solve(coeffs, chi_vector)
        q_eq = sol[:N_atoms]
        lambda_eq = sol[N_atoms]
        return q_eq, lambda_eq

