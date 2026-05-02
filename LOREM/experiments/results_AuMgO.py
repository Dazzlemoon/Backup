import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from pathlib import Path

def load_systems(filenames):
    systems = []
    for filename in filenames:
        systems += read(filename, index=":")
    return systems


def assign_calculators(systems, calculator):
    for system in systems:
        system.calc = calculator


def split_by_al_presence(systems):
    with_al = [sys for sys in systems if "Al" in sys.get_chemical_symbols()]
    without_al = [sys for sys in systems if "Al" not in sys.get_chemical_symbols()]
    return with_al, without_al


def compute_au_o_distance(system, atom_indices):
    return system.get_distance(*atom_indices)


def extract_properties(systems, reference_indices):
    distances = [compute_au_o_distance(system, reference_indices) for system in systems]
    energies = [system.get_potential_energy() for system in systems]
    direction = (
        systems[0].get_positions()[1] - systems[0].get_positions()[108]
    ) / np.linalg.norm(systems[0].get_positions()[1] - systems[0].get_positions()[108])
    forces = [
        direction @ (system.get_forces()[108] + system.get_forces()[109])
        for system in systems
    ]

    min_energy = min(energies)
    normalized_energies = [e - min_energy for e in energies]

    return distances, normalized_energies, forces


def plot_results(
    distances_doped,
    energies_doped,
    forces_doped,
    distances_undoped,
    energies_undoped,
    forces_undoped,
    output_path,
    label,
):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(distances_doped, energies_doped, label="Doped (with Al)", alpha=0.7)
    plt.scatter(distances_undoped, energies_undoped, label="Undoped", alpha=0.7)
    plt.xlabel("Distance between Au–O (Å)")
    plt.ylabel("Normalized Potential Energy (eV)")
    plt.legend()
    plt.title(f"Energy vs. Au–O Distance {label}")

    plt.subplot(1, 2, 2)
    plt.scatter(distances_doped, forces_doped, label="Doped (with Al)", alpha=0.7)
    plt.scatter(distances_undoped, forces_undoped, label="Undoped", alpha=0.7)
    plt.xlabel("Distance between Au–O (Å)")
    plt.ylabel("Total Force Norm (eV/Å)")
    plt.legend()
    plt.title(f"Force vs. Au–O Distance {label}")

    plt.tight_layout()
    plt.savefig(output_path)


def main(calculator, dataset_folder, workdir=Path().cwd(), label=None):
    if label is None:
        label = workdir.stem

    filenames = [
        dataset_folder / "AuMgO_non-wetting-doped_curve.xyz",
        dataset_folder / "AuMgO_non-wetting-undoped_curve.xyz",
    ]

    systems = load_systems(filenames)
    assign_calculators(systems, calculator)

    systems_doped, systems_undoped = split_by_al_presence(systems)

    # Distance between Au and O atoms (indices may need to be adjusted for your data)
    reference_indices = [1, 108]

    dist_doped, energy_doped, force_doped = extract_properties(
        systems_doped, reference_indices
    )
    dist_undoped, energy_undoped, force_undoped = extract_properties(
        systems_undoped, reference_indices
    )
    output_path = workdir / "Au-O_energy_force_curve.png"
    plot_results(
        dist_doped,
        energy_doped,
        force_doped,
        dist_undoped,
        energy_undoped,
        force_undoped,
        output_path,
        label,
    )

    np.savez(
        workdir / "Au-O_energy_force_curve_data.npz",
        dist_doped=dist_doped,
        energy_doped=energy_doped,
        force_doped=force_doped,
        dist_undoped=dist_undoped,
        energy_undoped=energy_undoped,
        force_undoped=force_undoped,
    )
