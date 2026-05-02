textwidth = 5.5  # 5.5 inches

nolabel = "__nolabel__"

tol_vibrant = [
    "#EE7733",
    "#0077BB",
    "#33BBEE",
    "#EE3377",
    "#CC3311",
    "#009988",
    "#BBBBBB",
    "#000000",
]

tol_muted = [
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#332288",
    "#DDCC77",
    "#999933",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#DDDDDD",
]


black = "#000000"
# red = "#CC3311"
red = "#EE2200"
teal = "#009988"
orange = "#EE7733"
blue = "#0077BB"
cyan = "#33BBEE"
magenta = "#EE3377"
grey = "#BBBBBB"
darkgrey = "#888888"


solid = "solid"
dashed = "dashed"
dotted = "dotted"
dashdot = "dashdot"
loosedot_thick = (0, (1, 2))
loosedot = (0, (1, 3))
loosedash = ((1, 1), 0)
finedot = (0, (0.5, 2))
finedash = (0, (4, 3))

cross = "x"
diamond = "D"
star = "*"
dot = "."
bigdot = "o"
square = "s"
thiamond = "d"
pentagon = "p"
plus = "P"


model_names = {
    "cace": r"\textsc{Cace-Les} {\small $1\times$SR + LR}",
    "MACE": r"\textsc{Mace} {\small $2\times$SR}",
    "NativePET-SR": r"\textsc{Pet} {\small $2\times$ SR}",
    "lorem": r"\textsc{Lorem} {\small $1\times$SR + LR}",
    "4G_NN": r"4G-NN {\small $1\times$SR + LR}",
}


model_colors = {
    "cace": orange,
    "MACE": blue,
    "NativePET-SR": teal,
    "lorem": red,
    "4G_NN": magenta,
}


model_linestyles = {
    "cace": dashed,
    "MACE": dotted,
    "NativePET-SR": finedot,
    "lorem": solid,
    "4G_NN": solid,
}


dataset_names = {
    "AuMgO": r"\ch{MgO} surface",
    "bio_dimers": "Biodimers",
    "carbon_chain": "Carbon chain",
    "cumulene": "Cumulene",
    "NaCl": r"\ch{NaCl} cluster",
}
