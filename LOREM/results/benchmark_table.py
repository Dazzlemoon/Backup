from tbx import *
from tbx.tables import *
from benchmark import lr, nolr

per_n = {}

for (n, name), value in lr.items():
    if n < 10:
        continue

    wo = 1e3 * nolr[(n, name)]
    w = 1e3 * value
    val = (name, wo, w, w - wo, 100 * (w - wo) / w)

    if n not in per_n:
        per_n[n] = [val]
    else:
        per_n[n].append(val)

per_n = sorted(per_n.items(), key=lambda kv: kv[0])


rows = []
for n, values in per_n:
    for i, (name, wo, w, diff, perc) in enumerate(values):

        nums = [f"{wo:.1f}", f"{w:.1f}", f"{diff:.1f}", f"{perc:.1f}"]

        nums = map_across([nums], lambda x: to_num(x) if x else "")[0]

        row = [f"{n}", f"{name.replace("_", " ")}", *nums]

        rows.append(row)


layout = " ".join(["r", "|"] + ["r"] * (len(rows[0]) - 1))

table = make_tabular(
    [r"$N$", "Crystal", "SR (ms)", "SR+LR (ms)", r"$\Delta$ (ms)", r"$\Delta$ (\%)"],
    rows,
    width=18,
    layout=layout,
)


print(table)
savefile(table, "tables/benchmark.tex")
