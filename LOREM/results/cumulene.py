from tbx import *

results = {}

dihedrals = None
reference = None

is_zero = []

for file in (datadir / "cumulene").glob("*/cumulene_dihedral_energy_curve.npz"):
    model = file.parent.stem
    data = np.load(file)
    if dihedrals is None:
        dihedrals = data["dihedrals"]
    else:
        np.testing.assert_allclose(dihedrals, data["dihedrals"])

    if reference is None:
        reference = data["actual_energies"]
    else:
        np.testing.assert_allclose(reference, data["actual_energies"])

    results[model] = data["predicted_energies"]

    if ((results[model] - results[model][0]) > 1e-5).sum() < 5:
        is_zero.append(model)

print("models that yield zero:")
print(is_zero)


def get_cutoff_variants_name(cutoff, mp, lr):
    if lr:
        middle = "-lr-"
    else:
        middle = "-sr-"

    cu = int(cutoff * 10)

    folder = "lorem" + f"-cu{cu}" + middle + f"mp{mp}"
    # name = r"$X\times\qty{Y}{\angstrom}$ SR-MP".replace("X", str(mp)).replace(
    #     "Y", f"{cutoff:.1f}"
    # )
    name = r"$X\times$SR".replace("X", str(mp))
    if lr:
        name += r" + $1\times$LR"

    return folder, name
