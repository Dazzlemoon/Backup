from tbx import *
from tbx.tables import *
from metrics import metrics

# experiment = "AuMgO"
for experiment in ["AuMgO", "bio_dimers", "carbon_chain", "cumulene", "NaCl"]:
    data = metrics[experiment]

    def get_row(data, key):
        dct = {k: v[key] for k, v in data.items()}
        best = min(dct, key=dct.get)

        row = list(dct.values())
        row = map_across([row], rounder(3))[0]

        for i, k in enumerate(dct.keys()):
            if k == best:
                row[i] = "*" + row[i] + "*"

        return row

    E = get_row(data, "energy_mae")
    F = get_row(data, "forces_mae")

    table = table_to_string(
        [E, F],
        colnames=[""] + list(data.keys()),
        width=16,
        rownames=["E (meV/at)", "F (meV/Å)"],
    )

    print(f"## {experiment} ##")
    print(table)
