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


def split_by_atom_count(systems, count1, count2):
    group1 = [sys for sys in systems if len(sys) == count1]
    group2 = [sys for sys in systems if len(sys) == count2]
    return group1, group2


def compute_na_na_distance(system, na_indices):
    return system.get_distance(*na_indices)


def extract_properties(systems, reference_indices):
    distances = [compute_na_na_distance(system, reference_indices) for system in systems]
    energies = [system.get_potential_energy() for system in systems]
    direction = (
        systems[0].get_positions()[0] - systems[0].get_positions()[3]
    ) / np.linalg.norm(systems[0].get_positions()[0] - systems[0].get_positions()[3])
    forces = [direction @ (system.get_forces()[0]) for system in systems]

    min_energy = min(energies)
    normalized_energies = [e - min_energy for e in energies]

    return distances, normalized_energies, forces


def plot_results(
    distances_16,
    energies_16,
    forces_16,
    distances_17,
    energies_17,
    forces_17,
    output_path,
    label,
):
    plt.figure(figsize=(12, 6))

    # Energy vs. Distance
    plt.subplot(1, 2, 1)
    plt.scatter(distances_16, energies_16, label="16 atoms", alpha=0.7)
    plt.scatter(distances_17, energies_17, label="17 atoms", alpha=0.7)
    plt.xlabel("Na–Na Distance (Å)")
    plt.ylabel("Normalized Potential Energy (eV)")
    plt.legend()
    plt.title(f"Energy vs. Na–Na Distance {label}")

    # Force vs. Distance
    plt.subplot(1, 2, 2)
    plt.scatter(distances_16, forces_16, label="16 atoms", alpha=0.7)
    plt.scatter(distances_17, forces_17, label="17 atoms", alpha=0.7)
    plt.xlabel("Na–Na Distance (Å)")
    plt.ylabel("Force Norm (eV/Å)")
    plt.legend()
    plt.title(f"Force vs. Na–Na Distance {label}")

    plt.tight_layout()
    plt.savefig(output_path)


def main(calculator, dataset_folder, workdir=Path().cwd(), label=None):
    if label is None:
        label = workdir.stem

    systems = load_systems(
        [dataset_folder / "Na8Cl8_curve.xyz", dataset_folder / "Na9Cl8_curve.xyz"]
    )
    assign_calculators(systems, calculator)

    systems_16, systems_17 = split_by_atom_count(systems, 16, 17)

    dist_16, energy_16, force_16 = extract_properties(systems_16, [0, 3])
    dist_17, energy_17, force_17 = extract_properties(systems_17, [0, 3])
    output_path = workdir / "Na-Na_energy_force_curve.png"
    plot_results(
        dist_16, energy_16, force_16, dist_17, energy_17, force_17, output_path, label
    )

    np.savez(
        workdir / "Na-Na_energy_force_curve_data.npz",
        dist_16=dist_16,
        energy_16=energy_16,
        force_16=force_16,
        dist_17=dist_17,
        energy_17=energy_17,
        force_17=force_17,
    )
