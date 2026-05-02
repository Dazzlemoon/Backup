from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .atomic_data import AtomicData
from .atomic_data import default_data_key  # 仅为类型提示/潜在复用
from .neighborhood import get_neighborhood
from ..tasks.load_data import SubsetAtoms, random_train_valid_split


def read_extxyz_with_charge(
    path: str,
    cutoff: float,
    atomic_energies: Optional[Dict[int, float]] = None,
    z_map: Optional[Dict[str, int]] = None,
) -> List[AtomicData]:
    """
    手工解析 extxyz 文件（带 Properties=species,pos,forces,charge），构造 AtomicData 列表。

    - 不依赖 ASE，直接用 numpy/纯 Python。
    - 默认用于 NaCl：z_map = {'Na': 11, 'Cl': 17}。
    """
    if z_map is None:
        z_map = {"Na": 11, "Cl": 17}

    data_list: List[AtomicData] = []

    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                nat = int(line)
            except ValueError:
                break

            header = f.readline().strip()

            # 解析 Lattice
            cell = np.eye(3, dtype=float)
            if "Lattice=" in header:
                try:
                    lat_str = header.split("Lattice=")[1].split('"')[1]
                    vals = [float(x) for x in lat_str.split()]
                    cell = np.array(vals, dtype=float).reshape(3, 3)
                except Exception:
                    cell = np.eye(3, dtype=float)

            # 解析 energy（若存在）
            energy = None
            if "energy=" in header:
                try:
                    e_str = header.split("energy=")[1].split()[0]
                    energy = float(e_str)
                except Exception:
                    energy = None

            species: List[str] = []
            pos_list: List[Sequence[float]] = []
            forces_list: List[Sequence[float]] = []
            charge_list: List[float] = []
            for _ in range(nat):
                parts = f.readline().split()
                if not parts:
                    continue
                s = parts[0]
                species.append(s)
                vals = [float(x) for x in parts[1:]]
                if len(vals) < 7:
                    raise ValueError(
                        "Expect at least 7 numeric columns (pos[3], forces[3], charge[1])"
                    )
                pos_list.append(vals[0:3])
                forces_list.append(vals[3:6])
                charge_list.append(vals[6])

            positions = np.asarray(pos_list, dtype=float)
            forces = np.asarray(forces_list, dtype=float)
            charges = np.asarray(charge_list, dtype=float)
            Z = np.array([z_map[s] for s in species], dtype=int)

            # 邻域
            edge_index, shifts, unit_shifts = get_neighborhood(
                positions=positions,
                cutoff=cutoff,
                pbc=tuple([False, False, False]),
                cell=cell,
            )

            # 减去原子能（若提供）
            if atomic_energies is not None and energy is not None:
                energy -= sum(atomic_energies.get(int(z), 0.0) for z in Z)

            tdtype = torch.get_default_dtype()
            positions_t = torch.tensor(positions, dtype=tdtype)
            forces_t = torch.tensor(forces, dtype=tdtype)
            cell_t = torch.tensor(cell, dtype=tdtype)
            edge_index_t = torch.tensor(edge_index, dtype=torch.long)
            shifts_t = torch.tensor(shifts, dtype=tdtype)
            unit_shifts_t = torch.tensor(unit_shifts, dtype=tdtype)
            atomic_numbers_t = torch.tensor(Z, dtype=torch.long)
            energy_t = (
                torch.tensor(energy, dtype=tdtype) if energy is not None else None
            )
            charges_t = torch.tensor(charges, dtype=tdtype)

            additional_info = {"charge": charges_t}
            data = AtomicData(
                edge_index=edge_index_t,
                positions=positions_t,
                shifts=shifts_t,
                unit_shifts=unit_shifts_t,
                cell=cell_t,
                atomic_numbers=atomic_numbers_t,
                num_nodes=atomic_numbers_t.shape[0],
                forces=forces_t,
                molecular_index=None,
                energy=energy_t,
                stress=None,
                virials=None,
                additional_info=additional_info,
            )
            data_list.append(data)

    return data_list


def get_dataset_from_extxyz_with_charge(
    train_path: str,
    cutoff: float,
    valid_fraction: float = 0.1,
    seed: int = 1,
    atomic_energies: Optional[Dict[int, float]] = None,
) -> SubsetAtoms:
    """
    手工解析 extxyz，构造带 charge 的 AtomicData，并拆分 train/valid。

    返回值接口与原来的 get_dataset_from_xyz 一致，方便直接替换。
    """
    all_data = read_extxyz_with_charge(train_path, cutoff, atomic_energies)
    train_data, valid_data = random_train_valid_split(all_data, valid_fraction, seed)
    data_key = {"energy": "energy", "forces": "forces", "charge": "charge"}
    return SubsetAtoms(
        train=train_data,
        valid=valid_data,
        test=[],
        cutoff=cutoff,
        data_key=data_key,
        atomic_energies=atomic_energies or {},
    )

