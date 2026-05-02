import os
import numpy as np
from ase.io import read
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

def assign_calculators(systems, calculator):
    for system in systems:
        system.calc = calculator


def main(calculator, dataset_folder, workdir=Path().cwd()):
    systems = read(dataset_folder / "cumulene_test.xyz", index=":")

    energy_reference = [
        system.get_potential_energy() / len(system) for system in systems
    ]
    forces_reference = [system.get_forces() for system in systems]
    assign_calculators(systems, calculator)
    energy_mlip = [system.get_potential_energy() / len(system) for system in systems]
    forces_mlip = [system.get_forces() for system in systems]
    energy_rmse = np.sqrt(mean_squared_error(energy_reference, energy_mlip))
    energy_mae = mean_absolute_error(energy_reference, energy_mlip)
    # energy_r2 = r2_score(energy_reference, energy_mlip)

    forces_reference_flat = np.concatenate(forces_reference, axis=0)
    forces_mlip_flat = np.concatenate(forces_mlip, axis=0)

    forces_rmse = np.sqrt(mean_squared_error(forces_reference_flat, forces_mlip_flat))
    forces_mae = mean_absolute_error(forces_reference_flat, forces_mlip_flat)
    # forces_r2 = r2_score(forces_reference_flat, forces_mlip_flat)

    print(
        f"Energy RMSE : {energy_rmse * 1000:.6f} meV/atom, MAE: {energy_mae * 1000:.6f} meV/atom"
    )  # , R2: {energy_r2:.6f}")
    print(
        f"Forces RMSE: {forces_rmse * 1000:.6f} meV/Å, MAE: {forces_mae * 1000:.6f} meV/Å"
    )  # R2: {forces_r2:.6f}")

    np.savez(
        workdir / "metrics.npz",
        energy_rmse=energy_rmse * 1000,
        energy_mae=energy_mae * 1000,
        # energy_r2=energy_r2,
        forces_rmse=forces_rmse * 1000,
        forces_mae=forces_mae * 1000,
        # forces_r2=forces_r2
    )
