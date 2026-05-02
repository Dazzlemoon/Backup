from tbx import *

experiments = ["AuMgO", "bio_dimers", "carbon_chain", "cumulene", "NaCl"]

metrics = {}

for experiment in experiments:
    metrics[experiment] = {}
    for file in (datadir / experiment).glob("*/metrics.npz"):
        model = file.parent.stem
        data = np.load(file)

        metrics[experiment][model] = {}

        try:
            metrics[experiment][model]["energy_mae"] = float(data["energy_mae"])
            metrics[experiment][model]["forces_mae"] = float(data["forces_mae"])
        except KeyError:
            print(f"{experiment}/{model} does not have mae metrics!")


        try:
            metrics[experiment][model]["energy_rmse"] = float(data["energy_rmse"])
            metrics[experiment][model]["forces_rmse"] = float(data["forces_rmse"])
        except KeyError:
            print(f"{experiment}/{model} does not have rmse metrics!")


    metrics[experiment] = {
        k: v for k, v in sorted(metrics[experiment].items(), key=lambda x: x[0].lower())
    }
