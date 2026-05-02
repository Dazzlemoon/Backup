from tbx import *
from itertools import cycle
from sn2 import by_system

system = "Cl_Br"

data = by_system[system]

fig, ax = fig_and_ax(figsize=(textwidth, 3.5))

ax.plot(data["reference"], color=darkgrey, linewidth=4, label="Reference")

keys = [
    "lorem",
    "lorem-sr-mp2",
]


linestyles = {
    "lorem": solid,
    "lorem-sr-mp2": dotted,
}

linewidths = {
    "lorem": 2,
    "lorem-sr-mp2": 2,
}

model_names = {
    "lorem": r"$1\times$SR + LR",
    "lorem-sr-mp2": r"$2\times$SR",
}

color = model_colors["lorem"]

zero_labels = []

for i, key in enumerate(keys):
    ax.plot(
        data[key],
        label=model_names[key],
        color=color,
        linestyle=linestyles[key],
        alpha=0.8,
        linewidth=linewidths[key],
    )

leg = ax.legend(loc="upper right")

no_ticks(ax, direction="x")
scale_labels(ax, 1e-3, direction="y")

ax.set_ylabel("Energy (meV)")
ax.set_xlabel("Reaction coordinate")

savefig(fig, "figures/sn2")
