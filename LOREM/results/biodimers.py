from tbx import *


classes = [
    "AA",
    "CA",
    "CC",
    "CP",
    "PA",
    "PP",
]


metrics = {}

for file in (datadir / "bio_dimers").glob("*/metrics_AA.npz"):
    folder = file.parent
    model = file.parent.stem

    metrics[model] = {}
    for cl in classes:
        data = np.load(folder / f"metrics_{cl}.npz")
        metrics[model][cl] = {
            "energy_rmse": float(data["energy_rmse"]),
            "energy_mae": float(data["energy_mae"]),
            "forces_rmse": float(data["forces_rmse"]),
            "forces_mae": float(data["forces_mae"]),
        }

print(metrics)