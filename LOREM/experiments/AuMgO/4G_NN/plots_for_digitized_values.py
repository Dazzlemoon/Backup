import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read


def load_systems(filenames):
    systems = []
    for filename in filenames:
        systems += read(filename, index=":")
    return systems

def split_by_al_presence(systems):
    with_al = [sys for sys in systems if "Al" in sys.get_chemical_symbols()]
    without_al = [sys for sys in systems if "Al" not in sys.get_chemical_symbols()]
    return with_al, without_al


def compute_au_o_distance(system, atom_indices):
    return system.get_distance(*atom_indices)


def extract_properties(systems, npz_energy, npz_force, reference_indices):
    distances = [compute_au_o_distance(system, reference_indices) for system in systems]
    energy_interpolator = lambda x: np.interp(x, npz_energy["distance"], npz_energy["energy"], left=0, right = 0)
    forces_interpolator = lambda x: np.interp(x, npz_force["distance"], npz_force["force"], left=0, right = 0)
    energies = [energy_interpolator(d) for d in distances]
    forces = [forces_interpolator(d) for d in distances]

    min_energy = min(energies)
    normalized_energies = [e - min_energy for e in energies]

    return distances, normalized_energies, forces


def plot_results(distances_doped, energies_doped, forces_doped,
                 distances_undoped, energies_undoped, forces_undoped, output_path):

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(distances_doped, energies_doped, label='Doped (with Al)', alpha=0.7)
    plt.scatter(distances_undoped, energies_undoped, label='Undoped', alpha=0.7)
    plt.xlabel('Distance between Au–O (Å)')
    plt.ylabel('Normalized Potential Energy (eV)')
    plt.legend()
    plt.title('Energy vs. Au–O Distance')

    plt.subplot(1, 2, 2)
    plt.scatter(distances_doped, forces_doped, label='Doped (with Al)', alpha=0.7)
    plt.scatter(distances_undoped, forces_undoped, label='Undoped', alpha=0.7)
    plt.xlabel('Distance between Au–O (Å)')
    plt.ylabel('Total Force Norm (eV/Å)')
    plt.legend()
    plt.title('Force vs. Au–O Distance')

    plt.tight_layout()
    plt.savefig(output_path)


def main():
    filenames = [
        os.path.dirname(__file__) + '/../../../datasets/4G/Au-MgO-Al/Au-MgO-Al-non-wetting-doped_curve.xyz',
        os.path.dirname(__file__) + '/../../../datasets/4G/Au-MgO-Al/Au-MgO-Al-non-wetting-undoped_curve.xyz'
    ]

    systems = load_systems(filenames)

    systems_doped, systems_undoped = split_by_al_presence(systems)

    # Distance between Au and O atoms (indices may need to be adjusted for your data)
    reference_indices = [1, 108]
    doped_energy = np.load("4G_doped_energy.npz")
    doped_force = np.load("4G_doped_force.npz")
    undoped_energy = np.load("4G_undoped_energy.npz")
    undoped_force = np.load("4G_undoped_force.npz")
    dist_doped, energy_doped, force_doped = extract_properties(systems_doped, doped_energy, doped_force, reference_indices)
    dist_undoped, energy_undoped, force_undoped = extract_properties(systems_undoped, undoped_energy, undoped_force, reference_indices)
    output_path = os.path.dirname(__file__) + "/Au-O_energy_force_curve.png"
    plot_results(dist_doped, energy_doped, force_doped,
                 dist_undoped, energy_undoped, force_undoped, output_path)

    np.savez(
        os.path.dirname(__file__) + "/Au-O_energy_force_curve_data.npz",
        dist_doped=dist_doped,
        energy_doped=energy_doped,
        force_doped=force_doped,
        dist_undoped=dist_undoped,
        energy_undoped=energy_undoped,
        force_undoped=force_undoped
    )

if __name__ == "__main__":
    main()
