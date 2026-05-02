from cumulene import is_zero, get_cutoff_variants_name, results
from tbx import savefile
from tbx.tables import *


table = []


def get_cell(cutoff, mp, lr):
    name = get_cutoff_variants_name(cutoff, mp, lr)[0]
    if name in is_zero:
        return r"\no"
    elif name not in results:
        return "TBD"
    else:
        return r"\yes"


# table.append(["Steps of SR-MP", "1", "2", "1", "2", "3"])
# table.append(["LR-MP", "Yes", "Yes", "No", "No", "No"])

for cutoff in [2.5, 3.0, 3.5]:
    row = [r"\qty{cutoff}{\angstrom}".replace("cutoff", f"{cutoff:.1f}")]
    lr = True
    for mp in [1, 2]:
        row.append(get_cell(cutoff, mp, lr))

    lr = False
    for mp in [1, 2, 3, 4]:
        row.append(get_cell(cutoff, mp, lr))

    table.append(row)

titles = [
    "Cutoff",
    r"\makecell{$1\times$SR \\+ LR}",
    r"\makecell{$2\times$SR \\+ LR}",
    r"$1\times$SR",
    r"$2\times$SR",
    r"$3\times$SR",
    r"$4\times$SR",
]

out = make_tabular(titles, table, layout="r|cc cccc")

print(out)
savefile(out, "tables/cumulene_cutoffs.tex")

# for mp in [1, 2, 3]:
#     for lr in [True, False]:
#         if lr and mp > 1:
#             continue

#         name = get_cutoff_variants_name(cutoff, mp, lr)[0]
#         if name in is_zero:
#             print(f"{name} is zero")
#         elif name in results:
#             print(f"{name} is nonzeor")
#         else:
#             print(f"{name} is missing")

# print(table_to_string(table, colnames=colnames, width=14))
