import os
import numpy as np
from ase.io import read
import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error


def assign_calculators(systems, calculator):
    for system in systems:
        system.calc = calculator


def main(calculator, dataset_folder, workdir=Path().cwd()):
    filename = dataset_folder / "bio_dimers_test.xyz"

    systems = read(filename, index=":")
    energy_reference = np.array(
        [system.get_potential_energy() / len(system) for system in systems]
    )
    forces_reference = [system.get_forces() for system in systems]
    assign_calculators(systems, calculator)
    energy_mlip = []
    forces_mlip = []
    for system in tqdm.tqdm(systems):
        energy_mlip.append(system.get_potential_energy() / len(system))
        forces_mlip.append(system.get_forces())

    energy_mlip = np.array(energy_mlip)

    energy_rmse = 1e3 * np.sqrt(mean_squared_error(energy_reference, energy_mlip))
    energy_mae = 1e3 * mean_absolute_error(energy_reference, energy_mlip)
    # energy_r2 = r2_score(energy_reference, energy_mlip)

    forces_reference_flat = np.concatenate(forces_reference, axis=0)
    forces_mlip_flat = np.concatenate(forces_mlip, axis=0)

    forces_rmse = 1e3 * np.sqrt(mean_squared_error(forces_reference_flat, forces_mlip_flat))
    forces_mae = 1e3 * mean_absolute_error(forces_reference_flat, forces_mlip_flat)
    # forces_r2 = r2_score(forces_reference_flat, forces_mlip_flat)

    print(f"Energy RMSE : {energy_rmse:.6f} meV/atom, MAE: {energy_mae:.6f} meV/atom")
    print(f"Forces RMSE: {forces_rmse:.6f} meV/Å, MAE: {forces_mae:.6f} meV/Å")

    np.savez(
        workdir / "metrics.npz",
        energy_rmse=energy_rmse,
        energy_mae=energy_mae,
        forces_rmse=forces_rmse,
        forces_mae=forces_mae,
    )

    classes = {}
    for i, atoms in enumerate(systems):
        label = atoms.info["label"]
        if label in classes:
            classes[label].append(i)
        else:
            classes[label] = [i]

    for c, idx in classes.items():
        energy_rmse = 1e3 * np.sqrt(
            mean_squared_error(energy_reference[idx], energy_mlip[idx])
        )
        energy_mae = 1e3 * mean_absolute_error(energy_reference[idx], energy_mlip[idx])
        forces_rmse = 1e3 * np.sqrt(
            mean_squared_error(forces_reference_flat[idx], forces_mlip_flat[idx])
        )
        forces_mae = 1e3 * mean_absolute_error(
            forces_reference_flat[idx], forces_mlip_flat[idx]
        )

        print(f"Class {c}:")
        print(f"Energy RMSE : {energy_rmse:.6f} meV/atom, MAE: {energy_mae:.6f} meV/atom")
        print(f"Forces RMSE: {forces_rmse:.6f} meV/Å, MAE: {forces_mae:.6f} meV/Å")
        print()

        np.savez(
            workdir / f"metrics_{c}.npz",
            energy_rmse=energy_rmse,
            energy_mae=energy_mae,
            forces_rmse=forces_rmse,
            forces_mae=forces_mae,
        )
