from tbx import *

# -- our data --
doped_vs_undoped = {}

for file in (datadir / "AuMgO").glob("*/Au-O_wetting-vs-non-wetting.npz"):
    model = file.parent.stem
    data = np.load(file)

    doped_vs_undoped[model] = (data["doped"], data["undoped"])


curve = {}

mask = slice(15, 90)  # manually obtained data is padded w/ zeros
distance_doped = None
distance_undoped = None

for file in (datadir / "AuMgO").glob("*/Au-O_energy_force_curve_data.npz"):
    model = file.parent.stem
    data = np.load(file)

    this_dist_doped = data["dist_doped"]
    if distance_doped is None:
        distance_doped = this_dist_doped
    else:
        np.testing.assert_allclose(distance_doped, this_dist_doped)

    this_dist_undoped = data["dist_undoped"]
    if distance_undoped is None:
        distance_undoped = this_dist_undoped
    else:
        np.testing.assert_allclose(distance_undoped, this_dist_undoped)

    curve[model] = {
        "energy_doped": data["energy_doped"],
        "force_doped": data["force_doped"],
        "energy_undoped": data["energy_undoped"],
        "force_undoped": data["force_undoped"],
    }


# -- reference data --
from ase.io import read


traj_doped = read(
    datasetdir / "4G/Au-MgO-Al/Au-MgO-Al-non-wetting-doped_curve.xyz",
    index=":",
    format="extxyz",
)
reference_energy_doped = np.array([a.get_potential_energy() for a in traj_doped])[mask]
reference_distance_doped = np.array([a.info["distance"] for a in traj_doped])[mask]


traj_undoped = read(
    datasetdir / "4G/Au-MgO-Al/Au-MgO-Al-non-wetting-undoped_curve.xyz",
    index=":",
    format="extxyz",
)
reference_energy_undoped = np.array([a.get_potential_energy() for a in traj_undoped])[
    mask
]
reference_distance_undoped = np.array([a.info["distance"] for a in traj_undoped])[mask]
