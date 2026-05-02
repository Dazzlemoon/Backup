import os
import torch
import numpy as np
from ase.io import read
from collections import defaultdict
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def assign_calculators(systems):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calculator = MetatensorCalculator("best_model.pt", 
                                device=device,)
    for system in systems:
        system.calc = calculator


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def main():
    # Load dataset
    filename = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets', 'biodimers', 'test.xyz')
    systems = read(filename, index=':')

    # Assign ML calculators

    # Prepare storage for global and per-label metrics
    energies_ref = []
    energies_pred = []
    forces_ref_flat = []
    forces_pred_flat = []
    energy_ref_by_label = defaultdict(list)
    energy_pred_by_label = defaultdict(list)
    forces_ref_by_label = defaultdict(list)
    forces_pred_by_label = defaultdict(list)

    # Evaluate each system
    for system in systems:
        label = system.info["label"]
        n_atoms = len(system)

        # Reference values
        e_ref = system.get_potential_energy() / n_atoms
        f_ref = system.get_forces()
        energies_ref.append(e_ref)
        forces_ref_flat.append(f_ref)
        energy_ref_by_label[label].append(e_ref)
        forces_ref_by_label[label].append(f_ref)
    assign_calculators(systems)
    for system in systems:
        label = system.info["label"]
        n_atoms = len(system)
        e_pred = system.get_potential_energy() / n_atoms
        f_pred = system.get_forces()

        # Global lists
        
        energies_pred.append(e_pred)
        
        forces_pred_flat.append(f_pred)

        # Per-label lists
        
        energy_pred_by_label[label].append(e_pred)
        
        forces_pred_by_label[label].append(f_pred)

    # Flatten global forces
    forces_ref_flat = np.vstack(forces_ref_flat)
    forces_pred_flat = np.vstack(forces_pred_flat)

    # Compute global metrics
    e_rmse, e_mae, e_r2 = compute_metrics(energies_ref, energies_pred)
    f_rmse, f_mae, f_r2 = compute_metrics(forces_ref_flat.flatten(), forces_pred_flat.flatten())

    print("Global metrics:")
    print(f"  Energy   RMSE: {e_rmse*1000:.3f} meV/atom, MAE: {e_mae*1000:.3f} meV/atom, R2: {e_r2:.3f}")
    print(f"  Forces   RMSE: {f_rmse*1000:.3f} meV/Å, MAE: {f_mae*1000:.3f} meV/Å, R2: {f_r2:.3f}\n")

    # Save global metrics
    np.savez(
        "metrics.npz",
        energy_rmse=e_rmse * 1000,
        energy_mae=e_mae * 1000,
        energy_r2=e_r2,
        forces_rmse=f_rmse * 1000,
        forces_mae=f_mae * 1000,
        forces_r2=f_r2
    )
    # Compute and print per-label metrics, then save each to its own .npz
    print("Metrics per subclass label:")
    for label in energy_ref_by_label:
        e_ref = energy_ref_by_label[label]
        e_pred = energy_pred_by_label[label]
        f_ref = np.vstack(forces_ref_by_label[label])
        f_pred = np.vstack(forces_pred_by_label[label])

        er, ea, er2 = compute_metrics(e_ref, e_pred)
        fr, fa, fr2 = compute_metrics(f_ref.flatten(), f_pred.flatten())

        print(f"Label {label}:")
        print(f"  Energy   RMSE: {er*1000:.3f} meV/atom, MAE: {ea*1000:.3f} meV/atom, R2: {er2:.3f}")
        print(f"  Forces   RMSE: {fr*1000:.3f} meV/Å, MAE: {fa*1000:.3f} meV/Å, R2: {fr2:.3f}\n")

        # Save metrics for this label
        np.savez(
            f"metrics_{label}.npz",
            energy_rmse=er * 1000,
            energy_mae=ea * 1000,
            energy_r2=er2,
            forces_rmse=fr * 1000,
            forces_mae=fa * 1000,
            forces_r2=fr2
        )

if __name__ == '__main__':
    main()
