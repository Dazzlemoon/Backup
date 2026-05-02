import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read


def load_systems(filenames):
    systems = []
    for filename in filenames:
        systems += read(filename, index=":")
    return systems


def split_by_atom_count(systems, count1, count2):
    group1 = [sys for sys in systems if len(sys) == count1]
    group2 = [sys for sys in systems if len(sys) == count2]
    return group1, group2


def compute_na_na_distance(system, na_indices):
    return system.get_distance(*na_indices)


def extract_properties(systems, npz_energy, npz_force, reference_indices):
    distances = [compute_na_na_distance(system, reference_indices) for system in systems]
    energy_interpolator = lambda x: np.interp(x, npz_energy["distance"], npz_energy["energy"], left=0, right = 0)
    forces_interpolator = lambda x: np.interp(x, npz_force["distance"], npz_force["force"], left=0, right = 0)
    energies = [energy_interpolator(d) for d in distances]
    forces = [forces_interpolator(d) for d in distances]

    min_energy = min(energies)
    normalized_energies = [e - min_energy for e in energies]

    return distances, normalized_energies, forces


def plot_results(distances_16, energies_16, forces_16,
                 distances_17, energies_17, forces_17, output_path):

    plt.figure(figsize=(12, 6))

    # Energy vs. Distance
    plt.subplot(1, 2, 1)
    plt.scatter(distances_16, energies_16, label='16 atoms', alpha=0.7)
    plt.scatter(distances_17, energies_17, label='17 atoms', alpha=0.7)
    plt.xlabel('Na–Na Distance (Å)')
    plt.ylabel('Normalized Potential Energy (eV)')
    plt.legend()
    plt.title('Energy vs. Na–Na Distance')

    # Force vs. Distance
    plt.subplot(1, 2, 2)
    plt.scatter(distances_16, forces_16, label='16 atoms', alpha=0.7)
    plt.scatter(distances_17, forces_17, label='17 atoms', alpha=0.7)
    plt.xlabel('Na–Na Distance (Å)')
    plt.ylabel('Force Norm (eV/Å)')
    plt.legend()
    plt.title('Force vs. Na–Na Distance')

    plt.tight_layout()
    plt.savefig(output_path)


def main():
    systems = load_systems(
        [os.path.dirname(__file__) + '/../../../datasets/4G/NaCl/Na8Cl8_curve.xyz', 
         os.path.dirname(__file__) + '/../../../datasets/4G/NaCl/Na9Cl8_curve.xyz']
        )

    systems_16, systems_17 = split_by_atom_count(systems, 16, 17)

    Na8Cl8_energy = np.load("4G_Na8Cl8_energy.npz")
    Na8Cl8_force = np.load("4G_Na8Cl8_force.npz")
    Na9Cl8_energy = np.load("4G_Na9Cl8_energy.npz")
    Na9Cl8_force = np.load("4G_Na9Cl8_force.npz")
    dist_16, energy_16, force_16 = extract_properties(systems_16, Na8Cl8_energy, Na8Cl8_force, [0, 3])
    dist_17, energy_17, force_17 = extract_properties(systems_17, Na9Cl8_energy, Na9Cl8_force, [0, 3])
    output_path = os.path.dirname(__file__) + "/Na-Na_energy_force_curve.png"
    plot_results(dist_16, energy_16, force_16, dist_17, energy_17, force_17, output_path)

    np.savez(
        os.path.dirname(__file__) + "/Na-Na_energy_force_curve_data.npz",
        dist_16=dist_16,
        energy_16=energy_16,
        force_16=force_16,
        dist_17=dist_17,
        energy_17=energy_17,
        force_17=force_17,
    )

if __name__ == "__main__":
    main()
