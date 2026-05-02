import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from pathlib import Path


def load_systems(filenames):
    systems = []
    for filename in filenames:
        systems += read(filename, index=":")
    return systems


def main(calculator, dataset_folder, workdir=Path().cwd(), label=None):
    if label is None:
        label = workdir.stem

    filenames = [
        dataset_folder / "AuMgO_non-wetting-doped.xyz",
        dataset_folder / "AuMgO_non-wetting-undoped.xyz",
        dataset_folder / "AuMgO_wetting-doped.xyz",
        dataset_folder / "AuMgO_wetting-undoped.xyz",
    ]

    systems = [read(f, format="extxyz") for f in filenames]
    for atoms in systems:
        atoms.calc = calculator

    # we relax *only* the gold atoms, which are always the last two atoms
    constraint_doped = FixAtoms(indices=np.arange(len(systems[0]))[:-2])
    constraint_undoped = FixAtoms(indices=np.arange(len(systems[1]))[:-2])

    systems[0].set_constraint(constraint_doped)
    systems[2].set_constraint(constraint_doped)

    systems[1].set_constraint(constraint_undoped)
    systems[3].set_constraint(constraint_undoped)

    # execute relaxations
    for atoms in systems:
        opt = BFGS(atoms)
        opt.run(fmax=0.005)

    nw_d, nw_u, w_d, w_u = [atoms.get_potential_energy() for atoms in systems]

    doped = (w_d - nw_d) * 1e3
    undoped = (w_u - nw_u) * 1e3

    print(f"doped: {doped}meV")
    print(f"undoped: {undoped}meV")

    np.savez(
        workdir / "Au-O_wetting-vs-non-wetting.npz",
        doped=doped,
        undoped=undoped,
    )
