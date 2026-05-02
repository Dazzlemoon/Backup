import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from pathlib import Path

def assign_calculators(systems, calculator):
    for system in systems:
        system.calc = calculator


def plot_results(predicted_energies, actual_energies, output_path, label):
    plt.figure(figsize=(10, 6))

    plt.plot(actual_energies, label="Energies from DFT", alpha=0.7, color="black", linewidth=5)
    plt.plot(predicted_energies, label="Predicted energies", color="red")
    plt.xlabel("Reaction coordinate (arb. units)")
    plt.ylabel("Potential Energy (eV)")
    plt.legend()
    plt.title(f"Energy along reaction coordinate ({label})")

    plt.savefig(output_path)


def main(calculator, dataset_folder, workdir=Path().cwd(), label=None):
    if label is None:
        label = workdir.stem

    output_dir = workdir / "minimum_energy_path"
    output_dir.mkdir(exist_ok=True)

    results = {}
    for path in dataset_folder.glob("sn2_minimum_energy_path/*.xyz"):
        system = path.stem
        print(system)
        systems = read(path, index=":")

        actual_energies = [atoms.get_potential_energy() for atoms in systems]

        assign_calculators(systems, calculator)
        predicted_energies = [atoms.get_potential_energy() for atoms in systems]

        output_path = output_dir / f"{system}.png"
        plot_results(predicted_energies, actual_energies, output_path, label)

        results[f"{system}_true"] = actual_energies
        results[f"{system}_pred"] = predicted_energies

    np.savez(
        output_dir / "minimum_energy_path.npz",
        **results,
    )
