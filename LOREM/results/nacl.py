from tbx import *
from ase.io import read


# -- reference data --
traj_na8 = read(datasetdir / "4G/NaCl/Na8Cl8_curve.xyz", index=":", format="extxyz")
traj_na9 = read(datasetdir / "4G/NaCl/Na9Cl8_curve.xyz", index=":", format="extxyz")

mask = slice(12, -6)

reference_energy_na8 = np.array([a.get_potential_energy() for a in traj_na8])[mask]
reference_energy_na9 = np.array([a.get_potential_energy() for a in traj_na9])[mask]

reference_distance_na8 = np.array([a.info["distance"] for a in traj_na8])[mask]
reference_distance_na9 = np.array([a.info["distance"] for a in traj_na9])[mask]


# -- results --

results_na8 = {}
results_na9 = {}

distance_na8 = None
distance_na9 = None


for file in (datadir / "NaCl").glob("*/Na-Na_energy_force_curve_data.npz"):
    model = file.parent.stem
    data = np.load(file)

    if distance_na8 is None:
        distance_na8 = data["dist_16"]
    else:
        np.testing.assert_allclose(distance_na8, data["dist_16"])

    if distance_na9 is None:
        distance_na9 = data["dist_17"]
    else:
        np.testing.assert_allclose(distance_na9, data["dist_17"])

    results_na8[model] = data["energy_16"]
    results_na9[model] = data["energy_17"]

    # if "4G" in model:
    #     # for consistency, we also remove the minimum value
    #     results_na8[model] -= results_na8[model][20:90].min()
    #     results_na9[model] -= results_na9[model][20:90].min()
