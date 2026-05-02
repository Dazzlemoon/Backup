import torch
import torch.nn as nn
from itertools import product
from typing import Dict
import pytorch_finufft
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SOGPotential(nn.Module):
    def __init__(self,
                 N_dl=1,  # Fourier modes
                 bandwidth_num = 12,
                 external_field = None, # external field
                 external_field_direction: int = 0, # external field direction, 0 for x, 1 for y, 2 for z
                 charge_neutral_lambda: float = None,
                 remove_self_interaction=False,
                 feature_key: str = 'q',
                 output_key: str = 'SOG_potential',
                 aggregation_mode: str = "sum",
                 compute_field: bool = False,
                 Periodic: bool = False,
                 ):
        super().__init__()
        self.N_dl = N_dl
        self.bandwidth_num = bandwidth_num
        # Create bandwidth 
        # self.bandwidth = torch.linspace(-5, 1.2, self.bandwidth_num)  # Exponential decay
        # Parameters to learn during training
        self.shift_1 = torch.nn.Parameter(torch.linspace(-0.5, 1.0, self.bandwidth_num, dtype=torch.float32))
        self.amplitude_1 = torch.nn.Parameter(torch.ones(self.bandwidth_num, dtype=torch.float32))

        # self.shift_1 = torch.nn.Parameter(torch.linspace(-3.0, 2.0, self.bandwidth_num, dtype=torch.float32))
        # self.amplitude_1 = torch.nn.Parameter(torch.tensor([-7.0450, 11.4645, -4.9724, 0.4311, 0.1973, -0.1282, 0.4223, 1.3309, 3.2130, 8.1743, 19.3299, 55.2736],dtype=torch.float32))# dimer-CC
        # self.amplitude_1 = torch.nn.Parameter(torch.tensor([0.2750, 0.1375, 0.0688, 0.0344, 0.0172, 0.0086, 0.0043, 0.0021, 0.0011, 0.0005, 0.0003, 0.0001], dtype=torch.float32))
        # self.shift_1 = torch.nn.Parameter(torch.tensor([2.8, 5.7, 11.4, 22.7, 45.5, 91.0, 182.0, 364.0, 728.0, 1456.0, 2912.0, 5823.9],dtype=torch.float32))
        
        # self.amplitude_1 = torch.tensor([0.2750, 0.1375, 0.0688, 0.0344, 0.0172, 0.0086, 0.0043, 0.0021, 0.0011, 0.0005, 0.0003, 0.0001], dtype=torch.float32).to(device)
        # self.shift_1 = torch.tensor([2.8, 5.7, 11.4, 22.7, 45.5, 91.0, 182.0, 364.0, 728.0, 1456.0, 2912.0, 5823.9],dtype=torch.float32).to(device)
 
        self.Periodic = Periodic
        #print("shift_begin:",self.shift_1)

        self.norm_factor = torch.tensor(1.0)# self.norm_factor = torch.nn.Parameter(torch.tensor(1.0))
        self.ene_factor = torch.nn.Parameter(torch.tensor(0.0))#self.ene_factor = torch.tensor(0.0) # self.ene_factor = torch.nn.Parameter(torch.tensor(0.0))
        # self.ene_factor = torch.tensor(0.0)

        self.remove_self_interaction = remove_self_interaction
        self.feature_key = feature_key
        self.output_key = output_key
        self.aggregation_mode = aggregation_mode
        self.model_outputs = [output_key]
        self.external_field = external_field
        self.external_field_direction = external_field_direction
        self.compute_field = compute_field
        if self.compute_field:
            self.model_outputs.append(feature_key+'_field')
        self.charge_neutral_lambda = charge_neutral_lambda

        self.dl = self.N_dl
        self.sigma = 1.0
        self.exponent = 1 ##6
        self.sigma_sq_half = self.sigma ** 2 / 2.0
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        #self.norm_factor = 1.0 
        self.k_sq_max = (self.twopi / self.dl) ** 2

    def forward(self, data: Dict[str, torch.Tensor], **kwargs):
        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]

        # this is just for compatibility with the previous version
        if hasattr(self, 'exponent') == False:
            self.exponent = 1
        if hasattr(self, 'compute_field') == False:
            self.compute_field = False
        
        # box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1)
        box = data['cell'].view(-1, 3, 3)
        #print("cell:",box)

        r = data['positions'] # (total_atom_number_of_all_configurations_in_batch, 3)
        #print(r.shape)

        q = data[self.feature_key]
        if q.dim() == 1:
            q = q.unsqueeze(1)
        #print(q.shape) # (total_atom_number_of_all_configurations_in_batch, number_of_q_layers)

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'

        unique_batches = torch.unique(batch_now)  # Get unique batch indices. Batch_now saves the corresponding configuration index [0 0 ... 0 1 ... 1 2 ... 2]. Unique is used to get the total number of configurations in the batch
        #print(batch_now)
        #print(batch_now.shape, unique_batches.shape)
        results = []
        field_results = []
        for i in unique_batches:
            mask = (batch_now == i)  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_raw_now, q_now, box_now = r[mask], q[mask], box[i] # Extract the atomic information for each configuration.
            #print("Mask:",mask,i,r_raw_now.shape, q_now.shape, box_now.shape,r.shape,q.shape,box.shape)
            #print("box_now:", box_now)
            box_diag = box[i].diagonal(dim1=-2, dim2=-1)

            if self.Periodic:
                # the box is periodic, we use the reciprocal sum
                # print("use SOG")
                # pot = self.compute_potential_SOG(r_raw_now, q_now, box_now, self.compute_field)
                # print("use SOG-Ewald")
                pot, field = self.compute_potential_SOG_Ewald(r_raw_now, q_now, box_now, self.compute_field)
            else:
                pot, field = self.compute_potential_Gaussian_realspace(r_raw_now, q_now, self.compute_field)
                #pot = self.compute_potential_SOG(r_raw_now, q_now, box_now, self.compute_field)
            # pot, field = self.compute_potential_Gaussian_realspace(r_raw_now, q_now, self.compute_field)
            if self.exponent == 1 and hasattr(self, 'external_field') and self.external_field is not None:
                # if self.external_field_direction is an integer, then external_field_direction is the direction index
                if isinstance(self.external_field_direction, int):
                    direction_index_now = self.external_field_direction
                    # if self.external_field_direction is a string, then it is the key to the external field
                else:
                    try:
                        direction_index_now = int(data[self.external_field_direction][i])
                    except:
                        raise ValueError("external_field_direction must be an integer or a key to the external field")
                if isinstance(self.external_field, float):
                    external_field_now = self.external_field
                else:
                    try:
                        external_field_now = data[self.external_field][i]
                    except:
                        raise ValueError("external_field must be a float or a key to the external field")
                box_now = box_now.diagonal(dim1=-2, dim2=-1)
                pot_ext = self.add_external_field(r_raw_now, q_now, box_now, direction_index_now, external_field_now)
            else:
                pot_ext = 0.0

            if hasattr(self, 'charge_neutral_lambda') and self.charge_neutral_lambda is not None:
                q_mean = torch.mean(q[mask])
                pot_neutral = self.charge_neutral_lambda * (q_mean)**2.
                #print(pot_neutral, pot)
            else:
                pot_neutral = 0.0
            
            #print("pot", pot)
            #print("pot_ext",pot_ext)
            #print("pot_neutral",pot_neutral)
            # print(pot + pot_ext + pot_neutral)
            results.append(pot + pot_ext + pot_neutral)
            # print("SOG:",(pot + pot_ext + pot_neutral).shape)
            # results.append(pot + self.ene_factor)
        #print(results[0].shape,results[1].shape,results[2].shape, pot.shape)
        data[self.output_key] = torch.stack(results, dim=0).sum(axis=1) if self.aggregation_mode == "sum" else torch.stack(results, dim=0)
        if self.compute_field:
            field_results.append(field)
            data[self.feature_key+'_field'] = torch.cat(field_results, dim=0)
        return data
 
    def compute_potential_SOG(self, r_raw, q, box, compute_field=False):
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device
        #print(device)
        
        #print(r_raw.shape,q.shape, box[0,0], box[1,1], box[2,2]) 

        # **动态计算盒子体积**
        self.V = box[0,0] * box[1,1] * box[2,2]  # 体积
        Lx = box[0,0]
        Ly = box[1,1]
        Lz = box[2,2]  # 盒子尺寸
        #print("Box:",box,Lx,Ly,Lz)

        N_dl_x = torch.ceil(Lx/self.N_dl)
        N_dl_y = torch.ceil(Ly/self.N_dl)
        N_dl_z = torch.ceil(Lz/self.N_dl)
        
        #print(Lx, r_raw, q)

        # **动态计算傅里叶空间 k 网格**
        #kx = torch.linspace(-(self.NpointsMesh // 2), self.NpointsMesh // 2, self.NpointsMesh, dtype=torch.float64, device=device) * (2 * np.pi / Lx)
        kx = torch.fft.fftfreq(N_dl_x, d=self.N_dl / (2 * np.pi), device=device, dtype=torch.float32)
        #print("kx:",Lx,kx,torch.linspace(-(self.NpointsMesh // 2), self.NpointsMesh // 2, self.NpointsMesh, dtype=torch.float64, device=device) * (2 * np.pi / Lx))

        #ky = torch.linspace(-(self.NpointsMesh // 2), self.NpointsMesh // 2, self.NpointsMesh, dtype=torch.float64, device=device) * (2 * np.pi / Ly)
        ky = torch.fft.fftfreq(N_dl_y, d=self.N_dl / (2 * np.pi), device=device, dtype=torch.float32)
        
        #kz = torch.linspace(-(self.NpointsMesh // 2), self.NpointsMesh // 2, self.NpointsMesh, dtype=torch.float64, device=device) * (2 * np.pi / Lz)
        kz = torch.fft.fftfreq(N_dl_z, d=self.N_dl / (2 * np.pi), device=device, dtype=torch.float32)

        #print(torch.linspace(-(self.NpointsMesh // 2), self.NpointsMesh // 2, self.NpointsMesh, dtype=torch.float64, device=device),(2 * np.pi / Lx))
        #print(torch.linspace(-(self.NpointsMesh // 2), self.NpointsMesh // 2, self.NpointsMesh, dtype=torch.float64, device=device),(2 * np.pi / Lx))
        #print(torch.linspace(-(self.NpointsMesh // 2), self.NpointsMesh // 2, self.NpointsMesh, dtype=torch.float64, device=device),(2 * np.pi / Lz))
        
        #print(kx.shape, ky.shape, kz.shape, kx, ky, kz)
        kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='xy')
        # print("kx_grid:",kx_grid)
        # print("ky_grid:",ky_grid)
        # print("kz_grid:",kz_grid)
        #print(kx_grid)
        # **计算 k 平方和**
        squared_sum = kx_grid ** 2 + ky_grid ** 2 + kz_grid ** 2
        condition = (squared_sum == 0)  # 过滤零模
        a = squared_sum.unsqueeze(-1)  # 维度扩展以匹配 SOG 频率通道
        
        #print(self.shift_1)
        # **计算 SOG 滤波器**
        # print("shift_1:", self.shift_1)
        min_term = -1 / torch.exp(-2 * self.shift_1)  # 计算指数衰减
        # print("min_term:",min_term.shape,min_term)
        min_term = min_term.view(1, 1, 1, -1)  # 扩展维度
        # print("min_term_expand:",min_term.shape,min_term)
        # print("squ_exp:",a.shape,a)
        multiplier = self.amplitude_1.view(1, 1, 1, -1) * torch.exp(a * min_term)  # 计算 SOG 频谱响应
        # print("multiplier:",multiplier.shape,multiplier)
        multiplier = multiplier.sum(dim=-1)  # 在 SOG 频谱维度求和
        # print("multiplier1:",multiplier.shape,multiplier)

        #multiplier = multiplier/(squared_sum+1e-12) ## 加的
        #multiplier = torch.exp( - squared_sum / 2) / (squared_sum+1e-12) ## Ewald

        multiplier[condition] = 0.0  # 移除零模项
        #print(condition)
        #print(multiplier[0,0])

        # **计算归一化项**
        diag_sum = multiplier.sum(dim=-1).sum(dim=-1).sum(dim=-1) / (2 * self.V)
        # print("diag_sum:",diag_sum)
        # **归一化输入坐标**
        #print(r_raw)
        #r = r_raw / box  # 归一化粒子坐标
        L = torch.tensor([Lx, Ly, Lz]).to(device)
        #print(L.shape, type(L))
       
        # **转换电荷数据**
        charge_complex = q.view(q.shape[0], -1).to(dtype)
        #charge_complex = q.transpose(0, 1).unsqueeze(-1).to(dtype)
        # print("charge:",charge_complex.shape, q)
    
        #print(charge_complex.shape)
        #print("raw:",r_raw.shape,r_raw[0])
        #wrapped_r_raw = torch.remainder(r_raw, L)
        #print("mod_raw:",wrapped_r_raw.shape,wrapped_r_raw[0])
        #input_te1 = (2 * np.pi / L) * (wrapped_r_raw - L / 2)  # 归一化输入坐标
        #print("raw1:",r_raw.shape,input_te1[0])
        #print(torch.remainder(r_raw[0], L),r_raw[0])
        r_raw = torch.remainder(r_raw, L)
        # print("raw:",r_raw.shape,r_raw)
        input_te = (2 * np.pi / L) * (r_raw - L / 2) 
        # print("input_te:",input_te.shape,input_te)
        #print("raw:",r_raw.shape,input_te[0], r_raw[0])
        #pot = torch.zeros((input_te.shape[0], input_te.shape[1]), dtype=torch.float64, device=device)
        pot = 0
        
        # torch.Size([703, 3]) torch.Size([703, 1]) 0
        #print(input_te.shape, charge_complex.shape, pot)
        
        transposed_position = (input_te.permute(1, 0)).to(torch.float32)
        # print("transposed_position:",transposed_position.shape,transposed_position)
        #print(transposed_position.shape)
        #print(torch.remainder(r_raw, L), r_raw)
        # **执行 NUFFT 计算**
        for i in range(charge_complex.shape[-1]):
            charge_select = charge_complex[:, i]  # 选取某一通道的电荷
            # print("charge_select:",charge_select.shape,charge_select)
            #print({stop})
            #transposed_charge = charge_select.permute(1, 0)
            #print(charge_select.real,charge_select)
            # print("NUFFT:",kx_grid.shape)
            recon = pytorch_finufft.functional.finufft_type1(transposed_position, charge_select, output_shape=kx_grid.shape, eps=1e-4, isign = -1)
            # 0 1 2 3 -3 -2 -1
            # print("recon:",recon[1,1,1],recon[2,2,2],recon[3,3,3],recon)
            #print(kx_grid.shape,recon.shape)
            # **应用 SOG 滤波**
            mult_fft = torch.complex(torch.mul(multiplier,recon.real), torch.mul(multiplier,recon.imag)).to(torch.complex64)
            # print("mult_fft:",mult_fft[1,1,1],mult_fft[2,2,2],mult_fft[3,3,3],mult_fft)
            # **逆 FFT**
            Ifftcon = pytorch_finufft.functional.finufft_type2(transposed_position, mult_fft, eps = 1e-4, isign = 1) / (2 * self.V)
            #print((charge_select.real * (Ifftcon - diag_sum)).real)
            #print(Ifftcon.shape, diag_sum, charge_select.shape)
            #torch.Size([17]) tensor(0.0006, dtype=torch.float64, grad_fn=<DivBackward0>) torch.Size([17])
            # **计算最终势能**
            #pot[:,i] += (charge_select.real * (Ifftcon - diag_sum)).real
            pot += (charge_select.real * (Ifftcon - diag_sum)).real.sum()
            #pot += (charge_select.real * (Ifftcon)).real.sum() 
            #print((charge_select.real * (Ifftcon - diag_sum)).real,pot)
        #print({stop})
        #print("pot",pot.shape)
        #print(pot.unsqueeze(0).shape)
        #print(pot,(charge_select.real * (Ifftcon - diag_sum)).real.sum())
        return pot.unsqueeze(0)* self.norm_factor

    def compute_potential_SOG_Ewald0(self, r_raw, q, box, compute_field=False):
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device
        
        #print(self.shift_1)
        
        cell_inv = torch.linalg.inv(box)
        G = 2 * torch.pi * cell_inv.T

        norms = torch.norm(box, dim=1)
        Lx = box[0,0]
        Ly = box[1,1]
        Lz = box[2,2]
        N_dl_x = (torch.ceil(Lx / self.N_dl)).int()
        N_dl_y = (torch.ceil(Ly / self.N_dl)).int()
        N_dl_z = (torch.ceil(Lz / self.N_dl)).int()
        #print(N_dl_x, N_dl_y,N_dl_z)
        n1 = torch.arange(-N_dl_x, N_dl_x + 1, device=device)
        n2 = torch.arange(-N_dl_y, N_dl_y + 1, device=device)
        n3 = torch.arange(-N_dl_z, N_dl_z + 1, device=device)
        
        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3)
        nvec = nvec.to(G.dtype)
        kvec = (nvec.float() @ G).to(device)
        
        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec ** 2, dim=1)

        mask = (k_sq > 0)
        
        kvec = kvec[mask] # [M, 3]
        k_sq = k_sq[mask] # [M]
        nvec = nvec[mask] # [M, 3]
        non_zero = (nvec != 0).to(torch.int)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = torch.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0)
        
        # Compute structure factor S(k), Σq*e^(ikr)
        k_dot_r = torch.matmul(r_raw, kvec.T)  # [n, M]
        exp_ikr = torch.exp(1j * k_dot_r)
        q_expanded = q.unsqueeze(-1)  # 将 q 的形状从 (16, 3) 扩展为 (16, 3, 1)
        # 扩展 exp_ikr 的维度
        exp_ikr_expanded = exp_ikr.unsqueeze(1)  # 将 exp_ikr 的形状从 (16, 4484342) 扩展为 (16, 1, 4484342)
        # 逐元素乘法
        product = q_expanded * exp_ikr_expanded  # 形状为 (16, 3, 4484342)
        # 计算结构因子 S(k)
        S_k = torch.sum(product, dim=0)  # 形状为 (3, 4484342)
        
        min_term = -1 / torch.exp(-2 * self.shift_1)  # 计算指数衰减
        min_term = min_term.view(1, 1, 1, -1)  # 扩展维度
        kfac = self.amplitude_1.view(1, 1, 1, -1) * torch.exp(k_sq.unsqueeze(-1) * min_term)  # 计算 SOG 频谱响应
        # print("multiplier:",multiplier.shape,multiplier)
        kfac = kfac.sum(dim=-1)  # 在 SOG 频谱维度求和
        #kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq

        volume = torch.det(box)
        pot = (factors * kfac * torch.abs(S_k)**2).sum() / (2*volume)

        kfac1 = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        pot1 = (factors * kfac1 * torch.abs(S_k)**2).sum() / volume
        # print("pot error:",pot.item(),pot1.item(),(pot-pot1).item())

        q_field = torch.zeros_like(q, dtype=r_raw.dtype, device=device)
        if compute_field:
            sk_field = 2 * kfac * torch.conj(S_k)
            q_field = (factors * torch.real(exp_ikr * sk_field)).sum(dim=1) / volume
            print("compute field")
        if self.remove_self_interaction and self.exponent == 1:
            #print("Here is self-interaction")
            #pot -= torch.sum(q**2) / (self.sigma * (2 * torch.pi)**1.5)  
            diag_sum = kfac.sum(dim=-1).sum(dim=-1).sum(dim=-1) / (2 * volume)
            pot -= torch.sum(q**2)*diag_sum
            q_field -= q * (2 *diag_sum)
            print("remove_self_interaction")

        return pot.unsqueeze(0) * self.norm_factor, q_field.unsqueeze(1) * self.norm_factor

    def compute_potential_SOG_Ewald(self, r_raw, q, box, compute_field=False):
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device
        
        # print(self.shift_1)
        
        cell_inv = torch.linalg.inv(box)
        G = 2 * torch.pi * cell_inv.T

        norms = torch.norm(box, dim=1)
        Lx = box[0,0]
        Ly = box[1,1]
        Lz = box[2,2]
        N_dl_x = (torch.ceil(Lx / self.N_dl)).int()
        N_dl_y = (torch.ceil(Ly / self.N_dl)).int()
        N_dl_z = (torch.ceil(Lz / self.N_dl)).int()
        #print(N_dl_x, N_dl_y,N_dl_z)
        n1 = torch.arange(-N_dl_x, N_dl_x + 1, device=device)
        n2 = torch.arange(-N_dl_y, N_dl_y + 1, device=device)
        n3 = torch.arange(-N_dl_z, N_dl_z + 1, device=device)
        
        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3)
        nvec = nvec.to(G.dtype)
        kvec = (nvec.float() @ G).to(device)
        
        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec ** 2, dim=1)
        mask = (k_sq > 0)
        kvec = kvec[mask] # [M, 3]
        k_sq = k_sq[mask] # [M]
        nvec = nvec[mask] # [M, 3]
        non_zero = (nvec != 0).to(torch.int)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = torch.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0)
        
        # Compute structure factor S(k), Σq*e^(ikr)
        k_dot_r = torch.matmul(r_raw, kvec.T)  # [n, M]
        exp_ikr = torch.exp(1j * k_dot_r)
        q_expanded = q.unsqueeze(-1)  # 将 q 的形状从 (16, 3) 扩展为 (16, 3, 1)
        # 扩展 exp_ikr 的维度
        exp_ikr_expanded = exp_ikr.unsqueeze(1)  # 将 exp_ikr 的形状从 (16, 4484342) 扩展为 (16, 1, 4484342)
        # 逐元素乘法
        product = q_expanded * exp_ikr_expanded  # 形状为 (16, 3, 4484342)
        # 计算结构因子 S(k)
        S_k = torch.sum(product, dim=0)  # 形状为 (3, 4484342)
        
        min_term = -1 / torch.exp(-2 * self.shift_1)  # 计算指数衰减
        min_term = min_term.view(1, 1, 1, -1)  # 扩展维度
        kfac = self.amplitude_1.view(1, 1, 1, -1) * torch.exp(k_sq.unsqueeze(-1) * min_term)  # 计算 SOG 频谱响应
        # print("multiplier:",multiplier.shape,multiplier)
        kfac = kfac.sum(dim=-1)  # 在 SOG 频谱维度求和
        #kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        
        volume = torch.det(box)
        pot = (factors * kfac * torch.abs(S_k)**2).sum() / volume
        

        kfac1 = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        pot1 = (factors * kfac1 * torch.abs(S_k)**2).sum() / volume
        # print("pot error:",pot.item(),pot1.item(),(pot-pot1).item())
        
        q_field = torch.zeros_like(q, dtype=r_raw.dtype, device=device)
        
        # if compute_field:
        #     sk_field = 2 * kfac * torch.conj(S_k)
        #     q_field = (factors * torch.real(exp_ikr * sk_field)).sum(dim=1) / volume
        # if self.remove_self_interaction and self.exponent == 1:
        #     pot -= torch.sum(q**2) / (self.sigma * (2 * torch.pi)**1.5)
        #     q_field -= q * (2 / (self.sigma * (2 * torch.pi)**1.5))
        if compute_field:
            sk_field = 2 * kfac * torch.conj(S_k)
            q_field = (factors * torch.real(exp_ikr * sk_field)).sum(dim=1) / volume
        if self.remove_self_interaction and self.exponent == 1:
            #print("Here is self-interaction")
            #pot -= torch.sum(q**2) / (self.sigma * (2 * torch.pi)**1.5)  
            diag_sum = kfac.sum(dim=-1).sum(dim=-1).sum(dim=-1) / (2 * volume)
            pot -= torch.sum(q**2)*diag_sum
            q_field -= q * (2 *diag_sum)
        return pot.unsqueeze(0) * self.norm_factor, q_field.unsqueeze(1) * self.norm_factor

        # **计算 SOG 滤波器**
        #min_term = -1 / torch.exp(-2 * self.shift_1)  # 计算指数衰减
        #min_term = min_term.view(1, 1, 1, -1)  # 扩展维度
        #multiplier = self.amplitude_1.view(1, 1, 1, -1) * torch.exp(a * min_term)  # 计算 SOG 频谱响应
        #multiplier = multiplier.sum(dim=-1)  # 在 SOG 频谱维度求和
        #multiplier[condition] = 0.0  # 移除零模项

        # **计算归一化项**
        #diag_sum = multiplier.sum(dim=-1).sum(dim=-1).sum(dim=-1) / (2 * self.V)

        #r = r_raw / box  # 归一化粒子坐标


    def compute_potential_Gaussian_realspace(self, r_raw, q, compute_field=False):
        #print(r_raw.shape)
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        #print(r_ij.shape)
        r_ij_norm = torch.norm(r_ij, dim=-1)

        # min_term = -4*torch.exp(-2 * self.shift_1)  # 计算指数衰减
        min_term = -1/self.shift_1**2
        #print(self.shift_1)
        # print(min_term,"herr",min_term.shape)
        min_term = min_term.view(1, 1, -1)  # 扩展维度
        # print(min_term.shape)
        convergence_func_ij = self.amplitude_1.view(1, 1, -1) * torch.exp( torch.square(r_ij_norm).unsqueeze(2) * min_term)
        #print(self.amplitude_1)
        # print((torch.square(r_ij_norm).unsqueeze(2)).shape)
        # print(convergence_func_ij.shape,convergence_func_ij)
        # print(self.amplitude_1)
        convergence_func_ij = torch.sum(convergence_func_ij, dim=2)
        # print(convergence_func_ij.shape)
        #print(convergence_func_ij.shape)
        # epsilon = 1e-6
        # r_p_ij = 1.0 / (r_ij_norm + epsilon)
        if q.dim() == 1:
            # [n_node, n_q]
            q = q.unsqueeze(1)
        
        # print(q.shape,compute_field,self.remove_self_interaction)

        Ewald_convergence_func_ij = torch.special.erf(r_ij_norm / self.sigma / (2.0 ** 0.5))
        Ewald_r_p_ij = 1.0 / (r_ij_norm + 1e-6)
        Ewald_pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1) * Ewald_r_p_ij.unsqueeze(2) * Ewald_convergence_func_ij.unsqueeze(2)).view(-1) / self.twopi / 2.0

        # Compute potential energy
        n_node, n_q = q.shape
        #print((q.unsqueeze(0) * q.unsqueeze(1)).shape, (r_p_ij.unsqueeze(2)).shape, (convergence_func_ij.unsqueeze(2)).shape )
        #pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2)).view(-1) / self.twopi / 2.

        # tensor = convergence_func_ij.unsqueeze(2).squeeze(-1)
        # print("aa",convergence_func_ij.shape,tensor.shape)
        idx = torch.arange(convergence_func_ij.size(0))
        convergence_func_ij[idx, idx] = 0
        # convergence_func_ij=tensor.unsqueeze(-1)
        # print("bb",convergence_func_ij.shape)

        pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1) * convergence_func_ij.unsqueeze(2)).view(-1) / self.twopi / 2.0        
        # print("Here:",convergence_func_ij.unsqueeze(2).shape,convergence_func_ij.unsqueeze(2))
        pot = pot.to(torch.float32)

        # print("Ewald vs SOG1:",(Ewald_r_p_ij.unsqueeze(2) * Ewald_convergence_func_ij.unsqueeze(2)), (convergence_func_ij.unsqueeze(2)),(Ewald_r_p_ij.unsqueeze(2) * Ewald_convergence_func_ij.unsqueeze(2)).shape,(convergence_func_ij.unsqueeze(2)).shape)
        #print(pot.shape)
        q_field = torch.zeros_like(q, dtype=q.dtype, device=q.device) # Field due to q
        # Compute field if requested
        if compute_field:
            # [n_node, 1 , n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
            #q_field = torch.sum(q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2), dim=0) / self.twopi
            q_field = torch.sum(q.unsqueeze(1) * convergence_func_ij.unsqueeze(2), dim=0) / self.twopi

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if self.remove_self_interaction == False and self.exponent == 1:
            pot += torch.sum(q ** 2) * self.amplitude_1.sum() / self.twopi / 2.0 #torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
            q_field = q_field + q * self.amplitude_1.sum() / self.twopi 

            Ewald_pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
            q_field = q_field + q / (self.sigma * self.twopi**(3./2.)) * 2.
        #print(pot.shape, q_field.shape)
        #print(pot* self.norm_factor,self.norm_factor.shape)
        #print(pot.unsqueeze(0).shape,(pot.unsqueeze(0) * self.norm_factor).shape)
        # print("Ewald:",Ewald_pot.item(),"vs SOG:", pot.item())
        return pot* self.norm_factor, q_field.unsqueeze(1) * self.norm_factor 
        # # Compute pairwise distances (norm of vector differences)
        # r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        # #print(r_ij.shape)
        # r_ij_norm = torch.norm(r_ij, dim=-1)

        # min_term = -4*torch.exp(-2 * self.shift_1)  # 计算指数衰减
        # min_term = min_term.view(1, 1, -1)  # 扩展维度
        # convergence_func_ij = self.amplitude_1.view(1, 1, -1) * torch.exp( torch.square(r_ij_norm).unsqueeze(2) * min_term)
        # #print(convergence_func_ij.shape)
        # convergence_func_ij = torch.sum(convergence_func_ij, dim=2)
        # #print(convergence_func_ij.shape)
        # # epsilon = 1e-6
        # # r_p_ij = 1.0 / (r_ij_norm + epsilon)
        # if q.dim() == 1:
        #     # [n_node, n_q]
        #     q = q.unsqueeze(1)
        # # Compute potential energy
        # n_node, n_q = q.shape
        # #print((q.unsqueeze(0) * q.unsqueeze(1)).shape, (r_p_ij.unsqueeze(2)).shape, (convergence_func_ij.unsqueeze(2)).shape )
        # #pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2)).view(-1) / self.twopi / 2.0
        # pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1) * convergence_func_ij.unsqueeze(2)).view(-1) / self.twopi / 2.0        
        # pot = pot.squeeze().to(torch.float32)
        # #print(pot.shape)
        # q_field = torch.zeros_like(q, dtype=q.dtype, device=q.device) # Field due to q
        # # Compute field if requested
        # if compute_field:
        #     # [n_node, 1 , n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
        #     #q_field = torch.sum(q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2), dim=0) / self.twopi
        #     q_field = torch.sum(q.unsqueeze(1) * convergence_func_ij.unsqueeze(2), dim=0) / self.twopi

        # # because this realspace sum already removed self-interaction, we need to add it back if needed
        # #if self.remove_self_interaction == False and self.exponent == 1:
        # #    pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
        # #    q_field = q_field + q / (self.sigma * self.twopi**(3./2.)) * 2.
        # #print(pot.shape, q_field.shape)
        # #print(pot* self.norm_factor,self.norm_factor.shape)
        # #print(pot.unsqueeze(0).shape,(pot.unsqueeze(0) * self.norm_factor).shape)
        # return pot.unsqueeze(0) * self.norm_factor, q_field.unsqueeze(1) * self.norm_factor 

    def compute_potential_realspace(self, r_raw, q, compute_field=False):
        # Compute pairwise distances (norm of vector differences)
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        r_ij_norm = torch.norm(r_ij, dim=-1)
        #print(r_ij_norm)
 
        # Error function scaling for long-range interactions
        convergence_func_ij = torch.special.erf(r_ij_norm / self.sigma / (2.0 ** 0.5))
        #print(convergence_func_ij)
   
        # Compute inverse distance safely
        # [n_node, n_node]
        #r_p_ij = torch.where(r_ij_norm > 1e-3, 1.0 / r_ij_norm, 0.0) # this causes gradient issues
        epsilon = 1e-6
        r_p_ij = 1.0 / (r_ij_norm + epsilon)

        if q.dim() == 1:
            # [n_node, n_q]
            q = q.unsqueeze(1)
    
        # Compute potential energy
        n_node, n_q = q.shape
        # Use broadcasting to set diagonal elements to 0
        #mask = torch.ones(n_node, n_node, n_q, dtype=torch.int64, device=q.device)
        #diag_indices = torch.arange(n_node)
        #mask[diag_indices, diag_indices, :] = 0
        # [1, n_node, n_q] * [n_node, 1, n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
        pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2)).view(-1) / self.twopi / 2.0
    
        q_field = torch.zeros_like(q, dtype=q.dtype, device=q.device) # Field due to q
        # Compute field if requested
        if compute_field:
            # [n_node, 1 , n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
            q_field = torch.sum(q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2), dim=0) / self.twopi

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if self.remove_self_interaction == False and self.exponent == 1:
            pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
            q_field = q_field + q / (self.sigma * self.twopi**(3./2.)) * 2.
    
        return pot * self.norm_factor, q_field * self.norm_factor

    def add_external_field(self, r_raw, q, box, direction_index, external_field):
        external_field_norm_factor = (self.norm_factor/90.0474)**0.5
        # wrap in box
        r = r_raw[:, direction_index] / box[direction_index]
        r =  r - torch.round(r)
        r = r * box[direction_index]
        return external_field * torch.sum(q * r.unsqueeze(1)) * external_field_norm_factor

    def change_external_field(self, external_field):
        self.external_field = external_field

    def is_orthorhombic(self, cell_matrix):
        diag_matrix = torch.diag(torch.diagonal(cell_matrix))
        is_orthorhombic = torch.allclose(cell_matrix, diag_matrix, atol=1e-6)
        return is_orthorhombic

class PSWFPotential(nn.Module):
    def __init__(self,
                 dl = 2.0,  # grid resolution
                 rcut = 4.0,  # width of the Gaussian on each atom
                 exponent=1, # default is for electrostattics with p=1, we can do London dispersion with p=6
                 external_field = None, # external field
                 external_field_direction: int = 0, # external field direction, 0 for x, 1 for y, 2 for z
                 charge_neutral_lambda: float = None,
                 remove_self_interaction=False,
                 feature_key: str = 'q',
                 output_key: str = 'ewald_potential',
                 aggregation_mode: str = "sum",
                 compute_field: bool = False,
                 ):
        super().__init__()
        self.dl = dl
        self.rcut = rcut
        self.c = 12.024
        self.poly_order = 10 # <= poly_order
        self.mono_coef = [1.99999288998575, 0.00168787785084523, -11.3222097075807,
                          0.987921696028683, 21.5457341750713, 32.0951969525458,
                          -127.138762636063, 118.131826030792, -29.1811517575215,
                          -13.5720497316504, 6.45196584189916]
        self.C0 = 0.708928398746538
        self.phi0 = 1.96140338029212

        self.exponent = exponent
        
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        self.remove_self_interaction = remove_self_interaction
        self.feature_key = feature_key
        self.output_key = output_key
        self.aggregation_mode = aggregation_mode
        self.model_outputs = [output_key]
        # 1/2\epsilon_0, where \epsilon_0 is the vacuum permittivity
        # \epsilon_0 = 5.55263*10^{-3} e^2 eV^{-1} A^{-1}
        #self.norm_factor = 90.0474
        self.norm_factor = 1.0 
        # when using a norm_factor = 1, all "charges" are scaled by sqrt(90.0474)
        # the external field is then scaled by sqrt(90.0474) = 9.48933
        self.k_sq_max = (self.twopi / self.dl) ** 2
        self.external_field = external_field
        self.external_field_direction = external_field_direction
        self.compute_field = compute_field
        if self.compute_field:
            self.model_outputs.append(feature_key+'_field')

        self.charge_neutral_lambda = charge_neutral_lambda

    def forward(self, data: Dict[str, torch.Tensor], **kwargs):
        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]

        # this is just for compatibility with the previous version
        if hasattr(self, 'exponent') == False:
            self.exponent = 1
        if hasattr(self, 'compute_field') == False:
            self.compute_field = False
        
        # box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1)
        box = data['cell'].view(-1, 3, 3)
        # print("Ewald box:",box)
        r = data['positions']
        q = data[self.feature_key]
        if q.dim() == 1:
            q = q.unsqueeze(1)
        #print("q_shape",q.dim(),q.size(0),q.shape) ##(2,82)

        # Check the input dimension 2
        n, d = r.shape
        #print("r_shape",n,d) ##(82,3)
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'

        unique_batches = torch.unique(batch_now)  # Get unique batch indices

        results = []
        field_results = []
        for i in unique_batches:
            mask = batch_now == i  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_raw_now, q_now, box_now = r[mask], q[mask], box[i]
            #print(r_raw_now.shape, q_now.shape, box_now.shape)
            box_diag = box[i].diagonal(dim1=-2, dim2=-1)
            if box_diag[0] < 1e-6 and box_diag[1] < 1e-6 and box_diag[2] < 1e-6 and self.exponent == 1:
                # the box is not periodic, we use the direct sum
                pot, field = self.compute_potential_realspace(r_raw_now, q_now, self.compute_field)
            elif box_diag[0] > 0 and box_diag[1] > 0 and box_diag[2] > 0:
                # the box is periodic, we use the reciprocal sum
                pot, field = self.compute_potential_triclinic(r_raw_now, q_now, box_now, self.compute_field)
            else:
                raise ValueError("Either all box dimensions must be positive or aperiodic box must be provided.")

            if self.exponent == 1 and hasattr(self, 'external_field') and self.external_field is not None:
                # if self.external_field_direction is an integer, then external_field_direction is the direction index
                if isinstance(self.external_field_direction, int):
                    direction_index_now = self.external_field_direction
                    # if self.external_field_direction is a string, then it is the key to the external field
                else:
                    try:
                        direction_index_now = int(data[self.external_field_direction][i])
                    except:
                        raise ValueError("external_field_direction must be an integer or a key to the external field")
                if isinstance(self.external_field, float):
                    external_field_now = self.external_field
                else:
                    try:
                        external_field_now = data[self.external_field][i]
                    except:
                        raise ValueError("external_field must be a float or a key to the external field")
                box_now = box_now.diagonal(dim1=-2, dim2=-1)
                pot_ext = self.add_external_field(r_raw_now, q_now, box_now, direction_index_now, external_field_now)
            else:
                pot_ext = 0.0

            if hasattr(self, 'charge_neutral_lambda') and self.charge_neutral_lambda is not None:
                q_mean = torch.mean(q[mask])
                pot_neutral = self.charge_neutral_lambda * (q_mean)**2.
                #print(pot_neutral, pot)
            else:
                pot_neutral = 0.0

            results.append(pot + pot_ext + pot_neutral)
            field_results.append(field)

        #print(results[0].shape,results[1].shape,results[2].shape, pot.shape)
        data[self.output_key] = torch.stack(results, dim=0).sum(axis=1) if self.aggregation_mode == "sum" else torch.stack(results, dim=0)
        if self.compute_field:
            data[self.feature_key+'_field'] = torch.cat(field_results, dim=0)
        return data

    def compute_potential_realspace(self, r_raw, q, compute_field=False):
        # Compute pairwise distances (norm of vector differences)
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        r_ij_norm = torch.norm(r_ij, dim=-1)
        #print(r_ij_norm)
 
        # Error function scaling for long-range interactions
        convergence_func_ij = torch.special.erf(r_ij_norm / self.sigma / (2.0 ** 0.5))
        #print(convergence_func_ij)
   
        # Compute inverse distance safely
        # [n_node, n_node]
        #r_p_ij = torch.where(r_ij_norm > 1e-3, 1.0 / r_ij_norm, 0.0) # this causes gradient issues
        epsilon = 1e-6
        r_p_ij = 1.0 / (r_ij_norm + epsilon)

        if q.dim() == 1:
            # [n_node, n_q]
            q = q.unsqueeze(1)
    
        # Compute potential energy
        n_node, n_q = q.shape
        # Use broadcasting to set diagonal elements to 0
        #mask = torch.ones(n_node, n_node, n_q, dtype=torch.int64, device=q.device)
        #diag_indices = torch.arange(n_node)
        #mask[diag_indices, diag_indices, :] = 0
        # [1, n_node, n_q] * [n_node, 1, n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
        pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2)).view(-1) / self.twopi / 2.0
    
        q_field = torch.zeros_like(q, dtype=q.dtype, device=q.device) # Field due to q
        # Compute field if requested
        if compute_field:
            # [n_node, 1 , n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
            q_field = torch.sum(q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2), dim=0) / self.twopi

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if self.remove_self_interaction == False and self.exponent == 1:
            pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
            q_field = q_field + q / (self.sigma * self.twopi**(3./2.)) * 2.
    
        return pot * self.norm_factor, q_field * self.norm_factor
 
    def add_external_field(self, r_raw, q, box, direction_index, external_field):
        external_field_norm_factor = (self.norm_factor/90.0474)**0.5
        # wrap in box
        r = r_raw[:, direction_index] / box[direction_index]
        r =  r - torch.round(r)
        r = r * box[direction_index]
        return external_field * torch.sum(q * r.unsqueeze(1)) * external_field_norm_factor

    def change_external_field(self, external_field):
        self.external_field = external_field

    def is_orthorhombic(self, cell_matrix):
        diag_matrix = torch.diag(torch.diagonal(cell_matrix))
        is_orthorhombic = torch.allclose(cell_matrix, diag_matrix, atol=1e-6)
        return is_orthorhombic
    
    # Triclinic box(could be orthorhombic)
    def compute_potential_triclinic(self, r_raw, q, cell_now, compute_field=False):
        device = r_raw.device

        cell_inv = torch.linalg.inv(cell_now)
        G = 2 * torch.pi * cell_inv.T  # Reciprocal lattice vectors [3,3], G = 2π(M^{-1}).T

        # max Nk for each axis
        norms = torch.norm(cell_now, dim=1)
        Nk = [max(1, int(n.item() / self.dl)) for n in norms]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)
        
        #print(G)
        #print(cell_inv.T)
        #print("The number of Fourier grids is ", Nk) # 6

        # Create nvec grid and compute k vectors
        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3)
        nvec = nvec.to(G.dtype)
        # kvec = G @ nvec
        kvec = (nvec.float() @ G).to(device)  # [N_total, 3]

        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec ** 2, dim=1)
        mask = (k_sq > 0) & (k_sq <= self.k_sq_max)
        kvec = kvec[mask] # [M, 3]
        k_sq = k_sq[mask] # [M]
        nvec = nvec[mask] # [M, 3]
        
        # Determine symmetry factors (handle hemisphere to avoid double-counting)
        # Include nvec if first non-zero component is positive
        non_zero = (nvec != 0).to(torch.int)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = torch.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0)

        # Compute structure factor S(k), Σq*e^(ikr)
        k_dot_r = torch.matmul(r_raw, kvec.T)  # [n, M]
        exp_ikr = torch.exp(1j * k_dot_r)
        #print(k_dot_r.shape,exp_ikr.shape, q.shape)
        #S_k0 = torch.sum(q * exp_ikr, dim=0)  # [M]
        #print("S_k0",S_k0.shape)
        # 扩展 q 的维度
        q_expanded = q.unsqueeze(-1)  # 将 q 的形状从 (16, 3) 扩展为 (16, 3, 1)
        # 扩展 exp_ikr 的维度
        exp_ikr_expanded = exp_ikr.unsqueeze(1)  # 将 exp_ikr 的形状从 (16, 4484342) 扩展为 (16, 1, 4484342)
        # 逐元素乘法
        product = q_expanded * exp_ikr_expanded  # 形状为 (16, 3, 4484342)
        # 计算结构因子 S(k)
        S_k = torch.sum(product, dim=0)  # 形状为 (3, 4484342)
        #print("S_k",S_k.shape)

        # Compute kfac,  exp(-σ^2/2 k^2) / k^2 for exponent = 1
        if self.exponent == 1:
            k_abs = torch.sqrt(k_sq)
            input_kfactor = k_abs * self.rcut / self.c
            kfac = torch.zeros_like(input_kfactor)
            for coef in reversed(self.mono_coef):
                kfac = kfac * input_kfactor + coef
            kfac = torch.where(input_kfactor > 1, torch.zeros_like(kfac), kfac)
            kfac = kfac / k_sq
            # kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        elif self.exponent == 6:
            raise Exception("Error: PSWF currently does not support LJ potential")

            #b_sq = k_sq * self.sigma_sq_half
            #b = torch.sqrt(b_sq)
            #chukfac = -1.0 * k_sq**(3/2) * (
            #    torch.sqrt(torch.tensor(torch.pi)) * torch.special.erfc(b) + 
            #    (1/(2*b**3) - 1/b) * torch.exp(-b_sq)
            #)
        #print("kfac",kfac.shape)
        
        # Compute potential, (2π/volume)* sum(factors * kfac * |S(k)|^2)
        volume = torch.det(cell_now)
        #pot0 = (factors * kfac * torch.abs(S_k0)**2).sum() / volume
        #print("pot0",pot0.shape,pot0)
        
        #pot = (factors * kfac * torch.abs(S_k)**2).sum() / volume  # Ewald
        pot = (factors * kfac * torch.abs(S_k)**2).sum() / (2 * volume)
        
        #print("pot",pot.shape,pot)
        
        #print(compute_field) # fauls
        # Compute electric field if needed
        q_field = torch.zeros_like(q, dtype=r_raw.dtype, device=device)

        if compute_field:
            sk_field = 2 * kfac * torch.conj(S_k)
            #q_field = (factors * torch.real(exp_ikr * sk_field)).sum(dim=1) / volume # Ewald
            q_field = (factors * torch.real(exp_ikr * sk_field)).sum(dim=1) / (2 * volume)
        
        #print(self.remove_self_interaction) # fauls
        # Remove self-interaction if applicable
        if self.remove_self_interaction and self.exponent == 1:
            #pot -= torch.sum(q**2) / (self.sigma * (2 * torch.pi)**1.5) # Ewald
            pot -= torch.sum(q**2) * self.phi0 / (4 * torch.pi * self.C0 *self.rcut)
            #q_field -= q * (2 / (self.sigma * (2 * torch.pi)**1.5)) # Ewald
            q_field -= q * self.phi0 / (2 * torch.pi * self.C0 *self.rcut)

        return pot.unsqueeze(0) * self.norm_factor, q_field.unsqueeze(1) * self.norm_factor



class EwaldPotential(nn.Module):
    def __init__(self,
                 dl=2.0,  # grid resolution
                 sigma=1.0,  # width of the Gaussian on each atom
                 exponent=1, # default is for electrostattics with p=1, we can do London dispersion with p=6
                 external_field = None, # external field
                 external_field_direction: int = 0, # external field direction, 0 for x, 1 for y, 2 for z
                 charge_neutral_lambda: float = None,
                 remove_self_interaction=False,
                 feature_key: str = 'q',
                 output_key: str = 'ewald_potential',
                 aggregation_mode: str = "sum",
                 compute_field: bool = False,
                 ):
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.exponent = exponent
        self.sigma_sq_half = sigma ** 2 / 2.0
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        self.remove_self_interaction = remove_self_interaction
        self.feature_key = feature_key
        self.output_key = output_key
        self.aggregation_mode = aggregation_mode
        self.model_outputs = [output_key]
        # 1/2\epsilon_0, where \epsilon_0 is the vacuum permittivity
        # \epsilon_0 = 5.55263*10^{-3} e^2 eV^{-1} A^{-1}
        #self.norm_factor = 90.0474
        self.norm_factor = 1.0 
        # when using a norm_factor = 1, all "charges" are scaled by sqrt(90.0474)
        # the external field is then scaled by sqrt(90.0474) = 9.48933
        self.k_sq_max = (self.twopi / self.dl) ** 2
        self.external_field = external_field
        self.external_field_direction = external_field_direction
        self.compute_field = compute_field
        if self.compute_field:
            self.model_outputs.append(feature_key+'_field')

        self.charge_neutral_lambda = charge_neutral_lambda

    def forward(self, data: Dict[str, torch.Tensor], **kwargs):
        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]

        # this is just for compatibility with the previous version
        if hasattr(self, 'exponent') == False:
            self.exponent = 1
        if hasattr(self, 'compute_field') == False:
            self.compute_field = False
        
        # box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1)
        box = data['cell'].view(-1, 3, 3)
        #print("Ewald box:",box)
        r = data['positions']
        q = data[self.feature_key]
        if q.dim() == 1:
            q = q.unsqueeze(1)
        #print("q_shape",q.dim(),q.size(0),q.shape) ##(2,82)

        # Check the input dimension 2
        n, d = r.shape
        #print("r_shape",n,d) ##(82,3)
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'

        unique_batches = torch.unique(batch_now)  # Get unique batch indices

        results = []
        field_results = []
        for i in unique_batches:
            mask = batch_now == i  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_raw_now, q_now, box_now = r[mask], q[mask], box[i]
            #print(r_raw_now.shape, q_now.shape, box_now.shape)
            box_diag = box[i].diagonal(dim1=-2, dim2=-1)
            if box_diag[0] < 1e-6 and box_diag[1] < 1e-6 and box_diag[2] < 1e-6 and self.exponent == 1:
                # the box is not periodic, we use the direct sum
                #print("use realspace")
                pot, field = self.compute_potential_realspace(r_raw_now, q_now, self.compute_field)
            elif box_diag[0] > 0 and box_diag[1] > 0 and box_diag[2] > 0:
                # the box is periodic, we use the reciprocal sum
                #print("use triclinic")
                pot, field = self.compute_potential_triclinic(r_raw_now, q_now, box_now, self.compute_field)
            else:
                raise ValueError("Either all box dimensions must be positive or aperiodic box must be provided.")

            if self.exponent == 1 and hasattr(self, 'external_field') and self.external_field is not None:
                # if self.external_field_direction is an integer, then external_field_direction is the direction index
                if isinstance(self.external_field_direction, int):
                    direction_index_now = self.external_field_direction
                    # if self.external_field_direction is a string, then it is the key to the external field
                else:
                    try:
                        direction_index_now = int(data[self.external_field_direction][i])
                    except:
                        raise ValueError("external_field_direction must be an integer or a key to the external field")
                if isinstance(self.external_field, float):
                    external_field_now = self.external_field
                else:
                    try:
                        external_field_now = data[self.external_field][i]
                    except:
                        raise ValueError("external_field must be a float or a key to the external field")
                box_now = box_now.diagonal(dim1=-2, dim2=-1)
                pot_ext = self.add_external_field(r_raw_now, q_now, box_now, direction_index_now, external_field_now)
            else:
                pot_ext = 0.0

            if hasattr(self, 'charge_neutral_lambda') and self.charge_neutral_lambda is not None:
                q_mean = torch.mean(q[mask])
                pot_neutral = self.charge_neutral_lambda * (q_mean)**2.
                #print(pot_neutral, pot)
            else:
                pot_neutral = 0.0

            results.append(pot + pot_ext + pot_neutral)
            # print("Ewald:",(pot + pot_ext + pot_neutral).shape)
            field_results.append(field)

        #print(results[0].shape,results[1].shape,results[2].shape, pot.shape)
        data[self.output_key] = torch.stack(results, dim=0).sum(axis=1) if self.aggregation_mode == "sum" else torch.stack(results, dim=0)
        if self.compute_field:
            data[self.feature_key+'_field'] = torch.cat(field_results, dim=0)
        return data

    def compute_potential_realspace(self, r_raw, q, compute_field=False):
        # Compute pairwise distances (norm of vector differences)
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        r_ij_norm = torch.norm(r_ij, dim=-1)
        #print(r_ij_norm)
 
        # Error function scaling for long-range interactions
        convergence_func_ij = torch.special.erf(r_ij_norm / self.sigma / (2.0 ** 0.5))
        #print(convergence_func_ij)
   
        # Compute inverse distance safely
        # [n_node, n_node]
        #r_p_ij = torch.where(r_ij_norm > 1e-3, 1.0 / r_ij_norm, 0.0) # this causes gradient issues
        epsilon = 1e-6
        r_p_ij = 1.0 / (r_ij_norm + epsilon)

        if q.dim() == 1:
            # [n_node, n_q]
            q = q.unsqueeze(1)
    
        # Compute potential energy
        n_node, n_q = q.shape
        # Use broadcasting to set diagonal elements to 0
        #mask = torch.ones(n_node, n_node, n_q, dtype=torch.int64, device=q.device)
        #diag_indices = torch.arange(n_node)
        #mask[diag_indices, diag_indices, :] = 0
        # [1, n_node, n_q] * [n_node, 1, n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
        pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2)).view(-1) / self.twopi / 2.0
    
        q_field = torch.zeros_like(q, dtype=q.dtype, device=q.device) # Field due to q
        # Compute field if requested
        if compute_field:
            # [n_node, 1 , n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
            q_field = torch.sum(q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2), dim=0) / self.twopi

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if self.remove_self_interaction == False and self.exponent == 1:
            pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
            q_field = q_field + q / (self.sigma * self.twopi**(3./2.)) * 2.
    
        return pot * self.norm_factor, q_field * self.norm_factor
 

    def compute_potential(self, r_raw, q, box, compute_field=False):
        """ Compute the Ewald long-range potential for one configuration """
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device


        volume = box[0] * box[1] * box[2]
        r = r_raw / box  # Work with scaled positions
        # r =  r - torch.round(r) # periodic boundary condition

        # Calculate nk based on the provided box dimensions and resolution
        nk = (box / self.dl).int().tolist()
        for i in range(3):
            if nk[i] < 1: nk[i] = 1
        n = r.shape[0]
        eikx = torch.zeros((n, nk[0] + 1), dtype=dtype, device=device)
        eiky = torch.zeros((n, 2 * nk[1] + 1), dtype=dtype, device=device)
        eikz = torch.zeros((n, 2 * nk[2] + 1), dtype=dtype, device=device)

        eikx[:, 0] = torch.ones(n, dtype=dtype, device=device)
        eiky[:, nk[1]] = torch.ones(n, dtype=dtype, device=device)
        eikz[:, nk[2]] = torch.ones(n, dtype=dtype, device=device)

        # Calculate remaining positive kx, ky, and kz terms by recursion
        for k in range(1, nk[0] + 1):
            eikx[:, k] = torch.exp(1j * self.twopi * k * r[:, 0]) 
        for k in range(1, nk[1] + 1):
            eiky[:, nk[1] + k] = torch.exp(1j * self.twopi * k * r[:, 1])
        for k in range(1, nk[2] + 1):
            eikz[:, nk[2] + k] = torch.exp(1j * self.twopi * k * r[:, 2])

        # Negative k values are complex conjugates of positive ones
        for k in range(nk[1]):
            eiky[:, k] = torch.conj(eiky[:, 2 * nk[1] - k])
        for k in range(nk[2]):
            eikz[:, k] = torch.conj(eikz[:, 2 * nk[2] - k])

        pot_list = []
        q_field = torch.zeros_like(q, dtype=r_raw.dtype, device=device) # Field due to q
        
        for kx in range(nk[0] + 1):
            # for negative kx, the Fourier transform is just the complex conjugate of the positive kx
            factor = 1.0 if kx == 0 else 2.0

            for ky, kz in product(range(-nk[1], nk[1] + 1), range(-nk[2], nk[2] + 1)):
                k_sq = self.twopi_sq * ((kx / box[0]) ** 2 + (ky / box[1]) ** 2 + (kz / box[2]) ** 2)
                if k_sq <= self.k_sq_max and k_sq > 0:  # remove the k=0 term
                    if self.exponent == 1:
                        kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
                    elif self.exponent == 6:
                        b_sq = k_sq * self.sigma_sq_half
                        b = torch.sqrt(b_sq)
                        kfac = -1.0 * k_sq**(3/2) * (torch.pi**0.5 * torch.special.erfc(b) + (1 / (2 * b**3) - 1 / b) * torch.exp(-b_sq))
                    eik = (eikx[:, kx] * eiky[:, nk[1] + ky] * eikz[:, nk[2] + kz]).unsqueeze(1) # [n, 1]
                    sk = torch.sum(q * eik, dim=0) # [n_q]
                    sk_conj = torch.conj(sk)
                    sk_field = 2. * kfac * sk_conj # the factor of 2 comes from normalization factor 2\epsilon
                    pot_list.append(factor * kfac * torch.real(sk * sk_conj))
                    if compute_field:
                        # The reverse transform to get the real-space potential field
                        q_field += factor * torch.real(sk_field.unsqueeze(0) * eik)

        pot = torch.stack(pot_list).sum(axis=0) / volume
        if compute_field:
            q_field /= volume
        #print(pot, torch.sum(q * q_field, dim=0) /2) #should be the same

        if self.remove_self_interaction and self.exponent == 1:
            pot -= torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
            q_field = q_field - q / (self.sigma * self.twopi**(3./2.)) * 2.

        return pot.real * self.norm_factor, q_field * self.norm_factor

    # Optimized function
    def compute_potential_optimized(self, r_raw, q, box, compute_field=False):
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device

        volume = box[0] * box[1] * box[2]
        r = r_raw / box

        q_field = torch.zeros_like(q, dtype=r_raw.dtype, device=device) # Field due to q

        nk = (box / self.dl).int().tolist()
        nk = [max(1, k) for k in nk]

        n = r.shape[0]
        eikx = torch.ones((n, nk[0] + 1), dtype=dtype, device=device)
        eiky = torch.ones((n, 2 * nk[1] + 1), dtype=dtype, device=device)
        eikz = torch.ones((n, 2 * nk[2] + 1), dtype=dtype, device=device)

        eikx[:, 1] = torch.exp(1j * self.twopi * r[:, 0])
        eiky[:, nk[1] + 1] = torch.exp(1j * self.twopi * r[:, 1])
        eikz[:, nk[2] + 1] = torch.exp(1j * self.twopi * r[:, 2])
        # Calculate remaining positive kx, ky, and kz terms by recursion
        for k in range(2, nk[0] + 1):
            eikx[:, k] = eikx[:, k - 1].clone() * eikx[:, 1].clone()
        for k in range(2, nk[1] + 1):
            eiky[:, nk[1] + k] = eiky[:, nk[1] + k - 1].clone() * eiky[:, nk[1] + 1].clone()
        for k in range(2, nk[2] + 1):
            eikz[:, nk[2] + k] = eikz[:, nk[2] + k - 1].clone() * eikz[:, nk[2] + 1].clone()

        # Negative k values are complex conjugates of positive ones
        for k in range(nk[1]):
            eiky[:, k] = torch.conj(eiky[:, 2 * nk[1] - k])
        for k in range(nk[2]):
            eikz[:, k] = torch.conj(eikz[:, 2 * nk[2] - k])

        kx = torch.arange(nk[0] + 1, device=device)
        ky = torch.arange(-nk[1], nk[1] + 1, device=device)
        kz = torch.arange(-nk[2], nk[2] + 1, device=device)

        kx_term = (kx / box[0]) ** 2
        ky_term = (ky / box[1]) ** 2
        kz_term = (kz / box[2]) ** 2

        kx_sq = kx_term.view(-1, 1, 1)
        ky_sq = ky_term.view(1, -1, 1)
        kz_sq = kz_term.view(1, 1, -1)

        k_sq = self.twopi_sq * (kx_sq + ky_sq + kz_sq) # [nx, ny, nz]

        if self.exponent == 1:
            kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq # [nx, ny, nz]
        elif self.exponent == 6:
            # Calculate b_sq and b
            b_sq = k_sq * self.sigma_sq_half
            b = torch.sqrt(b_sq)

            # Compute kfac based on the provided expression
            kfac = -1.0 * k_sq ** (3 / 2) * ( torch.pi ** 0.5 * torch.special.erfc(b) + (1 / (2 * b ** 3) - 1 / b) * torch.exp(-b_sq))
            #kfac = -1.0 * k_sq ** (3 / 2) * torch.exp(-b_sq) # this assumed a Gaussian smearing

        mask = (k_sq <= self.k_sq_max) & (k_sq > 0)
        kfac[~mask] = 0

        eikx_expanded = eikx.unsqueeze(2).unsqueeze(3) #[n_node, n_x, 1, 1]
        eiky_expanded = eiky.unsqueeze(1).unsqueeze(3) #[n_node, 1, n_y, 1]
        eikz_expanded = eikz.unsqueeze(1).unsqueeze(2) #[n_node, 1, 1, n_z]

        factor = torch.ones_like(kx, dtype=r_raw.dtype, device=device)
        factor[1:] = 2.0

        if q.dim() == 1:
            # [n_node, n_q, 1, 1, 1]
            q_expanded = q.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        elif q.dim() == 2:
            q_expanded = q.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        else:
            raise ValueError("q must be 1D or 2D tensor")
        # q_expanded: [n_node, n_q, 1, 1, 1]
        # eik: [n_node, n_x, n_y, n_z]
        # sk: [n_q, n_x, n_y, n_z]
        # kfac: [n_x, n_y, n_z]
        eik = eikx_expanded * eiky_expanded * eikz_expanded
        sk = torch.sum(q_expanded * eik.unsqueeze(1), dim=[0])
        sk_conj = torch.conj(sk)
        pot = (kfac.unsqueeze(0) * factor.view(1, -1, 1, 1) * torch.real(sk_conj * sk)).sum(dim=[1, 2, 3])
        # The reverse transform to get the real-space potential field
        if compute_field:
            sk_field = 2. * kfac.unsqueeze(0) * sk_conj  # the factor of 2 comes from normalization factor 2\epsilon
            q_field = (factor.view(1, 1, -1, 1, 1) * torch.real(eik.unsqueeze(1) * sk_field.unsqueeze(0))).sum(dim=[2, 3, 4])
            q_field /= volume

        pot /= volume

        if self.remove_self_interaction and self.exponent == 1:
            pot -= torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
            q_field = q_field - q / (self.sigma * self.twopi**(3./2.)) * 2.

        return pot.real * self.norm_factor, q_field * self.norm_factor

    def add_external_field(self, r_raw, q, box, direction_index, external_field):
        external_field_norm_factor = (self.norm_factor/90.0474)**0.5
        # wrap in box
        r = r_raw[:, direction_index] / box[direction_index]
        r =  r - torch.round(r)
        r = r * box[direction_index]
        return external_field * torch.sum(q * r.unsqueeze(1)) * external_field_norm_factor

    def change_external_field(self, external_field):
        self.external_field = external_field

    def is_orthorhombic(self, cell_matrix):
        diag_matrix = torch.diag(torch.diagonal(cell_matrix))
        is_orthorhombic = torch.allclose(cell_matrix, diag_matrix, atol=1e-6)
        return is_orthorhombic
    
    # Triclinic box(could be orthorhombic)
    def compute_potential_triclinic(self, r_raw, q, cell_now, compute_field=False):
        device = r_raw.device

        cell_inv = torch.linalg.inv(cell_now)
        G = 2 * torch.pi * cell_inv.T  # Reciprocal lattice vectors [3,3], G = 2π(M^{-1}).T

        # max Nk for each axis
        norms = torch.norm(cell_now, dim=1)
        Nk = [max(1, int(n.item() / self.dl)) for n in norms]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)
        
        #print(G)
        #print(cell_inv.T)
        #print("The number of Fourier grids is ", Nk) # 6

        # Create nvec grid and compute k vectors
        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3)
        nvec = nvec.to(G.dtype)
        # kvec = G @ nvec
        kvec = (nvec.float() @ G).to(device)  # [N_total, 3]

        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec ** 2, dim=1)
        mask = (k_sq > 0) & (k_sq <= self.k_sq_max)
        kvec = kvec[mask] # [M, 3]
        k_sq = k_sq[mask] # [M]
        nvec = nvec[mask] # [M, 3]
        
        # Determine symmetry factors (handle hemisphere to avoid double-counting)
        # Include nvec if first non-zero component is positive
        non_zero = (nvec != 0).to(torch.int)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = torch.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0)

        # Compute structure factor S(k), Σq*e^(ikr)
        k_dot_r = torch.matmul(r_raw, kvec.T)  # [n, M]
        exp_ikr = torch.exp(1j * k_dot_r)
        #print(k_dot_r.shape,exp_ikr.shape, q.shape)
        #S_k0 = torch.sum(q * exp_ikr, dim=0)  # [M]
        #print("S_k0",S_k0.shape)
        # 扩展 q 的维度
        q_expanded = q.unsqueeze(-1)  # 将 q 的形状从 (16, 3) 扩展为 (16, 3, 1)
        # 扩展 exp_ikr 的维度
        exp_ikr_expanded = exp_ikr.unsqueeze(1)  # 将 exp_ikr 的形状从 (16, 4484342) 扩展为 (16, 1, 4484342)
        # 逐元素乘法
        product = q_expanded * exp_ikr_expanded  # 形状为 (16, 3, 4484342)
        # 计算结构因子 S(k)
        S_k = torch.sum(product, dim=0)  # 形状为 (3, 4484342)
        #print("S_k",S_k.shape)

        # Compute kfac,  exp(-σ^2/2 k^2) / k^2 for exponent = 1
        if self.exponent == 1:
            kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        elif self.exponent == 6:
            b_sq = k_sq * self.sigma_sq_half
            b = torch.sqrt(b_sq)
            kfac = -1.0 * k_sq**(3/2) * (
                torch.sqrt(torch.tensor(torch.pi)) * torch.special.erfc(b) + 
                (1/(2*b**3) - 1/b) * torch.exp(-b_sq)
            )
        #print("kfac",kfac.shape)
        
        # Compute potential, (2π/volume)* sum(factors * kfac * |S(k)|^2)
        volume = torch.det(cell_now)
        #pot0 = (factors * kfac * torch.abs(S_k0)**2).sum() / volume
        #print("pot0",pot0.shape,pot0)
        pot = (factors * kfac * torch.abs(S_k)**2).sum() / volume
        #print("pot",pot.shape,pot)
        
        #print(compute_field) # fauls
        # Compute electric field if needed
        q_field = torch.zeros_like(q, dtype=r_raw.dtype, device=device)
        if compute_field:
            sk_field = 2 * kfac * torch.conj(S_k)
            q_field = (factors * torch.real(exp_ikr * sk_field)).sum(dim=1) / volume
        
        #print(self.remove_self_interaction) # fauls
        # Remove self-interaction if applicable
        if self.remove_self_interaction and self.exponent == 1:
            pot -= torch.sum(q**2) / (self.sigma * (2 * torch.pi)**1.5)
            q_field -= q * (2 / (self.sigma * (2 * torch.pi)**1.5))

        # return pot.unsqueeze(0) * self.norm_factor, q_field.unsqueeze(1) * self.norm_factor
        return pot.unsqueeze(0) * self.norm_factor, q_field.unsqueeze(1) * self.norm_factor


class SOGPotentialReal(nn.Module):
    def __init__(self,
                 N_dl=1,  # Fourier modes
                 bandwidth_num = 4,
                 external_field = None, # external field
                 external_field_direction: int = 0, # external field direction, 0 for x, 1 for y, 2 for z
                 charge_neutral_lambda: float = None,
                 remove_self_interaction=False,
                 feature_key: str = 'q',
                 output_key: str = 'SOG_potential',
                 aggregation_mode: str = "sum",
                 compute_field: bool = False,
                 Periodic: bool = False,
                 ):
        super().__init__()
        self.N_dl = N_dl
        self.bandwidth_num = bandwidth_num
        # Create bandwidth 
        # self.bandwidth = torch.linspace(-5, 1.2, self.bandwidth_num)  # Exponential decay
        # Parameters to learn during training
        self.shift_1 = torch.nn.Parameter(torch.linspace(-0.5, 1.0, self.bandwidth_num, dtype=torch.float32))
        self.amplitude_1 = torch.nn.Parameter(torch.ones(self.bandwidth_num, dtype=torch.float32))

        # self.shift_1 = torch.nn.Parameter(torch.linspace(-3.0, 2.0, self.bandwidth_num, dtype=torch.float32))
        # self.amplitude_1 = torch.nn.Parameter(torch.tensor([-7.0450, 11.4645, -4.9724, 0.4311, 0.1973, -0.1282, 0.4223, 1.3309, 3.2130, 8.1743, 19.3299, 55.2736],dtype=torch.float32))# dimer-CC
        # self.amplitude_1 = torch.nn.Parameter(torch.tensor([0.2750, 0.1375, 0.0688, 0.0344, 0.0172, 0.0086, 0.0043, 0.0021, 0.0011, 0.0005, 0.0003, 0.0001], dtype=torch.float32))
        # self.shift_1 = torch.nn.Parameter(torch.tensor([2.8, 5.7, 11.4, 22.7, 45.5, 91.0, 182.0, 364.0, 728.0, 1456.0, 2912.0, 5823.9],dtype=torch.float32))
        
        # self.amplitude_1 = torch.tensor([0.2750, 0.1375, 0.0688, 0.0344, 0.0172, 0.0086, 0.0043, 0.0021, 0.0011, 0.0005, 0.0003, 0.0001], dtype=torch.float32).to(device)
        # self.shift_1 = torch.tensor([2.8, 5.7, 11.4, 22.7, 45.5, 91.0, 182.0, 364.0, 728.0, 1456.0, 2912.0, 5823.9],dtype=torch.float32).to(device)
 
        self.Periodic = Periodic
        #print("shift_begin:",self.shift_1)

        self.norm_factor = torch.tensor(1.0)# self.norm_factor = torch.nn.Parameter(torch.tensor(1.0))
        self.ene_factor = torch.nn.Parameter(torch.tensor(0.0))#self.ene_factor = torch.tensor(0.0) # self.ene_factor = torch.nn.Parameter(torch.tensor(0.0))

        self.remove_self_interaction = remove_self_interaction
        self.feature_key = feature_key
        self.output_key = output_key
        self.aggregation_mode = aggregation_mode
        self.model_outputs = [output_key]
        self.external_field = external_field
        self.external_field_direction = external_field_direction
        self.compute_field = compute_field
        if self.compute_field:
            self.model_outputs.append(feature_key+'_field')
        self.charge_neutral_lambda = charge_neutral_lambda

        self.dl = self.N_dl
        self.sigma = 1.0
        self.exponent = 1 ##6
        self.sigma_sq_half = self.sigma ** 2 / 2.0
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        #self.norm_factor = 1.0 
        self.k_sq_max = (self.twopi / self.dl) ** 2

    def forward(self, data: Dict[str, torch.Tensor], **kwargs):
        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]

        # this is just for compatibility with the previous version
        if hasattr(self, 'exponent') == False:
            self.exponent = 1
        if hasattr(self, 'compute_field') == False:
            self.compute_field = False
        
        # box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1)
        box = data['cell'].view(-1, 3, 3)
        #print("cell:",box)

        r = data['positions'] # (total_atom_number_of_all_configurations_in_batch, 3)
        #print(r.shape)

        q = data[self.feature_key]
        if q.dim() == 1:
            q = q.unsqueeze(1)
        #print(q.shape) # (total_atom_number_of_all_configurations_in_batch, number_of_q_layers)

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'

        unique_batches = torch.unique(batch_now)  # Get unique batch indices. Batch_now saves the corresponding configuration index [0 0 ... 0 1 ... 1 2 ... 2]. Unique is used to get the total number of configurations in the batch
        #print(batch_now)
        #print(batch_now.shape, unique_batches.shape)
        results = []
        field_results = []
        for i in unique_batches:
            mask = (batch_now == i)  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_raw_now, q_now, box_now = r[mask], q[mask], box[i] # Extract the atomic information for each configuration.
            #print("Mask:",mask,i,r_raw_now.shape, q_now.shape, box_now.shape,r.shape,q.shape,box.shape)
            #print("box_now:", box_now)
            box_diag = box[i].diagonal(dim1=-2, dim2=-1)

            if self.Periodic:
                # the box is periodic, we use the reciprocal sum
                # print("use SOG")
                # pot = self.compute_potential_SOG(r_raw_now, q_now, box_now, self.compute_field)
                # print("use SOG-Ewald")
                pot, field = self.compute_potential_Gaussian_realspace(r_raw_now, q_now, self.compute_field)
            else:
                pot, field = self.compute_potential_Gaussian_realspace(r_raw_now, q_now, self.compute_field)
                #pot = self.compute_potential_SOG(r_raw_now, q_now, box_now, self.compute_field)
            # pot, field = self.compute_potential_Gaussian_realspace(r_raw_now, q_now, self.compute_field)
            if self.exponent == 1 and hasattr(self, 'external_field') and self.external_field is not None:
                # if self.external_field_direction is an integer, then external_field_direction is the direction index
                if isinstance(self.external_field_direction, int):
                    direction_index_now = self.external_field_direction
                    # if self.external_field_direction is a string, then it is the key to the external field
                else:
                    try:
                        direction_index_now = int(data[self.external_field_direction][i])
                    except:
                        raise ValueError("external_field_direction must be an integer or a key to the external field")
                if isinstance(self.external_field, float):
                    external_field_now = self.external_field
                else:
                    try:
                        external_field_now = data[self.external_field][i]
                    except:
                        raise ValueError("external_field must be a float or a key to the external field")
                box_now = box_now.diagonal(dim1=-2, dim2=-1)
                pot_ext = self.add_external_field(r_raw_now, q_now, box_now, direction_index_now, external_field_now)
            else:
                pot_ext = 0.0

            if hasattr(self, 'charge_neutral_lambda') and self.charge_neutral_lambda is not None:
                q_mean = torch.mean(q[mask])
                pot_neutral = self.charge_neutral_lambda * (q_mean)**2.
                #print(pot_neutral, pot)
            else:
                pot_neutral = 0.0

            results.append(pot + pot_ext + pot_neutral)

        data[self.output_key] = torch.stack(results, dim=0).sum(axis=1) if self.aggregation_mode == "sum" else torch.stack(results, dim=0)
        if self.compute_field:
            field_results.append(field)
            data[self.feature_key+'_field'] = torch.cat(field_results, dim=0)
        return data

    def compute_potential_Gaussian_realspace(self, r_raw, q, compute_field=False):
        #print(r_raw.shape)
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        #print(r_ij.shape)
        r_ij_norm = torch.norm(r_ij, dim=-1)
        
        r2 = torch.square(r_ij_norm).unsqueeze(2)  # [n_node, n_node, 1]
        #print(r2.shape)
        min_term = -1.0 * torch.exp(-2 * self.shift_1) / 4.0          # [4]
        min_term = min_term.view(1, 1, self.bandwidth_num)          # [1,1,4]
        amplitude = self.amplitude_1.view(1, 1, self.bandwidth_num) # [1,1,4]
        # print(self.shift_1,amplitude)
        f_ij_all = amplitude * torch.exp(r2 * min_term)  # [n_node, n_node, 4]
        idx = torch.arange(f_ij_all.size(0))
        f_ij_all[idx, idx, :] = 0.0
                
# Step 3: 逐通道计算 q^T f_ij q
# q shape: [n_node, 4]
# q.unsqueeze(0): [1, n_node, 4]
# q.unsqueeze(1): [n_node, 1, 4]
        energy_per_channel = torch.sum(
            q.unsqueeze(0) * q.unsqueeze(1) * f_ij_all, dim=(0, 1)
        )
        # print(energy_per_channel)
# Step 4: 求和并归一化
        pot = energy_per_channel / self.twopi / 2.0  # [4]
        pot = torch.sum(pot).view(-1).to(torch.float32)
        
        q_field = torch.zeros_like(q, dtype=r_raw.dtype, device=device)
        if compute_field:
    # 每个通道都计算一遍场
    # q.unsqueeze(1): [n_node, 1, 4]
            q_field = torch.sum(q.unsqueeze(1) * f_ij_all, dim=0) / self.twopi  # [n_node, 4]

        #if self.remove_self_interaction == False and self.exponent == 1:
            # pot: shape [n_channel]
        #    pot += torch.sum(q ** 2, dim=0) * self.amplitude_1 / self.twopi / 2.0  # [4]

            # q_field: shape [n_node, n_channel]
        #    q_field = q_field + q * (self.amplitude_1 / self.twopi)  # broadcasting

        #print(pot.shape, q_field.shape)
        #print(pot* self.norm_factor,self.norm_factor.shape)
        #print(pot.unsqueeze(0).shape,(pot.unsqueeze(0) * self.norm_factor).shape)
        # print("Ewald:",Ewald_pot.item(),"vs SOG:", pot.item())
        return pot* self.norm_factor, q_field.unsqueeze(1) * self.norm_factor 

    def add_external_field(self, r_raw, q, box, direction_index, external_field):
        external_field_norm_factor = (self.norm_factor/90.0474)**0.5
        # wrap in box
        r = r_raw[:, direction_index] / box[direction_index]
        r =  r - torch.round(r)
        r = r * box[direction_index]
        return external_field * torch.sum(q * r.unsqueeze(1)) * external_field_norm_factor

    def change_external_field(self, external_field):
        self.external_field = external_field

    def is_orthorhombic(self, cell_matrix):
        diag_matrix = torch.diag(torch.diagonal(cell_matrix))
        is_orthorhombic = torch.allclose(cell_matrix, diag_matrix, atol=1e-6)
        return is_orthorhombic
