from typing import Dict, List, Optional

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
        # 迭代求解相关开关与参数（4.1）
        use_iterative_solver: bool = False,
        max_cg_iters: int = 200,
        cg_tol: float = 1e-6,
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
        self.model_outputs = [output_key, ewald_key]
        self.normalization_factor = norm_factor
        self.scaling_factor = scaling_factor
        self.compute_field = compute_field
        self.system_charge = system_charge
        self.aggregation_mode = aggregation_mode
        self.system_charge_key = system_charge_key
        self.use_sog_kernel = use_sog_kernel
        self.use_iterative_solver = use_iterative_solver
        self.max_cg_iters = max_cg_iters
        self.cg_tol = cg_tol
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
        assert (
            len(element_types) == len(self.elements)
        ), f"Number of unique elements {len(element_types)} != expected number {len(self.elements)}."
        if chi.dim() == 1:
            chi = chi.unsqueeze(1)

        J_raw = self.J_raw
        J_elem = torch.square(J_raw)  # 保证正数
        idx = self.Z_index_map[Z]
        J_i = J_elem[idx]

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

            # 构造/应用核矩阵 A：
            # - 默认：从 EwaldPotential 取出等效的 A；
            # - 可选：用 SOG 高斯和近似 1/r 构造 A_sog（可学习）。
            if self.use_sog_kernel and self.use_iterative_solver:
                # 4.1 + 4.2：不显式存 A，直接通过算子 + PCG 解 M q = b
                q_eq, lambda_eq = self._compute_q_eq_iterative(
                    A_mat=None,
                    r=r_now,
                    cell=box_now,
                    chi=chi_now,
                    J=J_i_now,
                    system_Q=system_charge_now,
                )
                # 能量同样通过算子形式计算：0.5 * q^T A q
                A_q = self._apply_A_sog(r_now, q_eq.squeeze(-1), box_now)
                ewald_energy = 0.5 * (q_eq.squeeze(-1) * A_q).sum()
            else:
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
                ewald_energy = 0.5 * q_eq.unsqueeze(1).T @ A_now @ q_eq.unsqueeze(1)

            results.append(q_eq)
            ewald_results.append(ewald_energy)

        all_q_eq = torch.cat(results, dim=0)
        if all_q_eq.dim() == 1:
            all_q_eq = all_q_eq.unsqueeze(1)
        data[self.output_key] = all_q_eq

        stacked = torch.stack(ewald_results, dim=0)
        if self.aggregation_mode == "sum":
            all_ewald = stacked.sum(dim=1) if stacked.dim() > 1 else stacked.flatten()
        else:
            all_ewald = stacked
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

    def _apply_A_sog(
        self,
        r: torch.Tensor,
        v: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        最小版本的 SOG 算子形式 Av（4.2 的第一步）：
        - 仍然是 O(N^2) 计算，但不在模块状态中长期保存 A，仅在函数内部构造并立即用于 matvec。
        - 后续可以用 FFT/NUFFT/FMM 等替换内部实现，而不影响外部接口。
        """
        device, dtype = r.device, r.dtype
        N = r.size(0)
        if v.dim() == 1:
            v = v.unsqueeze(-1)

        diff = r.unsqueeze(0) - r.unsqueeze(1)  # [N, N, 3]
        dist = torch.linalg.norm(diff, dim=-1) + 1e-8  # [N, N]
        r2 = dist ** 2  # [N, N]

        if self.shared_sog_potential is not None:
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

        Av = A_sog.to(dtype=dtype) @ v  # [N, 1]
        return Av.squeeze(-1)

    def _pcg(
        self,
        matvec,
        b: torch.Tensor,
        M=None,
        tol: float = 1e-6,
        max_iter: int = 200,
    ) -> torch.Tensor:
        """
        标准 PCG 实现（作用在对称正定矩阵 M 上），完全在 torch 中，以保证可微。
        matvec: 函数 v -> M v
        M: 预条件器函数 v -> P^{-1} v（可为 None，表示不预条件）
        """
        x = torch.zeros_like(b)
        r = b - matvec(x)
        z = M(r) if M is not None else r
        p = z.clone()
        rz_old = (r * z).sum()

        for _ in range(max_iter):
            Ap = matvec(p)
            alpha = rz_old / (p * Ap).sum()
            x = x + alpha * p
            r = r - alpha * Ap
            if torch.norm(r) < tol:
                break
            z = M(r) if M is not None else r
            rz_new = (r * z).sum()
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new

        return x

    def _compute_q_eq_iterative(
        self,
        A_mat: Optional[torch.Tensor],
        r: torch.Tensor,
        cell: torch.Tensor,
        chi: torch.Tensor,
        J: torch.Tensor,
        system_Q: torch.Tensor,
    ):
        """
        使用 PCG 在子问题空间上解 M q = b，并通过 Schur 消元处理电中性约束：
        M = A + J，对称正定（J_i > 0）。
        约束：1^T q = Q。
        步骤：
        1) 求解 M u = 1
        2) 求解 M v = -chi
        3) λ = (Q - 1^T v) / (1^T u)
        4) q = v + λ u
        """
        device, dtype = r.device, r.dtype
        N_atoms = r.size(0)

        if chi.dim() == 2 and chi.size(1) == 1:
            chi_vec = chi.view(-1)
        else:
            chi_vec = chi.view(-1)

        J_vec = J.to(dtype=dtype).view(-1)

        # 定义 matvec: M v = (A + J) v
        if A_mat is not None:
            def matvec(v):
                return A_mat @ v + J_vec * v
        else:
            def matvec(v):
                return self._apply_A_sog(r, v, cell) + J_vec * v

        # Jacobi 预条件器：P = diag(M) ≈ J（这里直接用 J）
        def precond(v):
            return v / (J_vec + 1e-8)

        tol = self.cg_tol
        max_iter = self.max_cg_iters

        ones = torch.ones(N_atoms, device=device, dtype=dtype)
        b_u = ones
        b_v = -chi_vec.to(device=device, dtype=dtype)

        u = self._pcg(matvec, b_u, M=precond if self.use_iterative_solver else None, tol=tol, max_iter=max_iter)
        v = self._pcg(matvec, b_v, M=precond if self.use_iterative_solver else None, tol=tol, max_iter=max_iter)

        Q_tot = system_Q / self.normalization_factor
        oneTu = ones @ u
        oneTv = ones @ v
        lambda_eq = (Q_tot - oneTv) / (oneTu + 1e-8)

        q_eq = v + lambda_eq * u
        q_eq = q_eq.view(-1, 1)

        return q_eq, lambda_eq

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

