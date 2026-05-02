import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from pathlib import Path

def assign_calculators(systems, calculator):
    for system in systems:
        system.calc = calculator


def order_by_dihedrals(systems, atom_indices=(0, 1, 11, 12)):
    dihedrals = []
    for sys in systems:
        dihedrals.append(sys.get_dihedral(*atom_indices))
    dihedrals = np.array(dihedrals)
    order = np.argsort(dihedrals)

    systems[:] = [systems[i] for i in order]

    return dihedrals[order]


def get_energies(systems):
    energies = []
    for sys in systems:
        energies.append(sys.get_potential_energy())

    min_energy = min(energies)
    normalized_energies = [e - min_energy for e in energies]

    return normalized_energies


def plot_results(dihedrals, predicted_energies, actual_energies, output_path, label):
    plt.figure(figsize=(10, 6))

    # Energy vs. Dihedral Angle
    plt.plot(dihedrals, predicted_energies, label="Predicted energies", alpha=0.7)
    plt.plot(dihedrals, actual_energies, label="Energies from DFT", alpha=0.7)
    plt.xlabel("Dihedral Angle (degrees)")
    plt.ylabel("Potential Energy (eV)")
    plt.legend()
    plt.title(f"Energy vs. Dihedral Angle ({label})")

    plt.savefig(output_path)


def main(calculator, dataset_folder, workdir=Path().cwd(), label=None):
    if label is None:
        label = workdir.stem

    systems = read(dataset_folder / "cumulene_profile.xyz", index=":")

    dihedrals = order_by_dihedrals(systems)

    actual_energies = get_energies(systems)

    assign_calculators(systems, calculator)
    predicted_energies = get_energies(systems)

    output_path = workdir / "cumulene_dihedral_energy_curve.png"
    plot_results(dihedrals, predicted_energies, actual_energies, output_path, label)

    np.savez(
        workdir / "cumulene_dihedral_energy_curve.npz",
        dihedrals=dihedrals,
        predicted_energies=predicted_energies,
        actual_energies=actual_energies,
    )
