from tbx import *

# -- our data --
experiments = {}

for file in (datadir / "sn2").glob("*/minimum_energy_path/minimum_energy_path.npz"):
    model = file.parent.parent.stem
    experiments[model] = np.load(file)

by_system = {}

for system in [
    "F_Br",
    "I_I",
    "Cl_Cl",
    "Br_I",
    "F_F",
    "F_I",
    "Br_Br",
    "Cl_I",
    "Cl_Br",
    "F_Cl",
]:
    this_data = {}
    reference = None

    for model, data in experiments.items():
        if reference is None:
            reference = data[f"{system}_true"]
        else:
            np.testing.assert_allclose(reference, data[f"{system}_true"])

        this_data[model] = data[f"{system}_pred"] - reference[0]

    this_data["reference"] = reference - reference[0]

    by_system[system] = this_data
