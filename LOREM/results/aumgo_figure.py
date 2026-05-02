from tbx import *
from itertools import cycle
from aumgo import *

fig, axs = plt.subplots(
    1, 2, figsize=(textwidth, 3.5), sharey=True, constrained_layout=True
)

a = axs[0]
b = axs[1]


def plot(
    ax,
    distance,
    energy,
    offset,
    label,
    color=black,
    linestyle=solid,
    alpha=0.8,
    linewidth=2,
    marker=True,
):
    minimum = np.argmin(energy)
    value = energy[minimum]

    energy -= value
    energy += offset

    value = energy[minimum]

    ax.plot(
        distance,
        energy,
        label=label,
        color=color,
        linestyle=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )

    if marker:
        ax.plot(
            [distance[minimum]],
            [value],
            color=color,
            alpha=alpha,
            marker=diamond,
        )


keys = [
    "lorem",
    "cace",
    "4G_NN",
    "MACE",
    "NativePET-SR",
]
keys = reversed(keys)


for i, key in enumerate(keys):
    linestyle = model_linestyles[key]

    if key == "4G_NN":
        this_mask = mask
    else:
        this_mask = slice(None)

    offset = 0.02 * i

    a.axhline(offset, color=black, alpha=0.1, linewidth=0.5)
    b.axhline(offset, color=black, alpha=0.1, linewidth=0.5)

    plot(
        a,
        reference_distance_doped,
        reference_energy_doped,
        offset,
        color=darkgrey,
        label="Reference" if i == 0 else nolabel,
        linewidth=5,
        alpha=0.8,
        marker=False,
    )
    a.plot(
        [2.332],
        [offset],
        color=darkgrey,
        alpha=1.0,
        marker=diamond,
    )

    plot(
        a,
        distance_doped[this_mask],
        curve[key]["energy_doped"][this_mask],
        offset,
        model_names[key],
        color=model_colors[key],
        linestyle=linestyle,
        alpha=1.0,
    )

    plot(
        b,
        reference_distance_undoped,
        reference_energy_undoped,
        offset,
        color=darkgrey,
        label="Reference" if i == 0 else nolabel,
        linewidth=5,
        alpha=0.8,
        marker=False,
    )
    b.plot(
        [2.190],
        [offset],
        color=darkgrey,
        alpha=1.0,
        marker=diamond,
    )

    plot(
        b,
        distance_undoped[this_mask],
        curve[key]["energy_undoped"][this_mask],
        offset,
        model_names[key],
        color=model_colors[key],
        linestyle=linestyle,
        alpha=1.0,
    )


split = 3
lines, labels = a.get_legend_handles_labels()

lines = [lines[0]] + list(reversed(lines[1:]))
labels = [labels[0]] + list(reversed(labels[1:]))

a.legend(lines[:split], labels[:split], loc="upper center")
b.legend(lines[split:], labels[split:], loc="upper center")

a.set_title(r"Doped")
b.set_title(r"Undoped")


a.set_xlim(2.32 - 0.13, 2.32 + 0.13)
b.set_xlim(2.18 - 0.1, 2.18 + 0.1)
a.set_ylim(-0.005, 0.2)

major_ticks_every(a, 0.02, direction="y")
minor_ticks_every(a, 0.01, direction="y")

from matplotlib.ticker import FuncFormatter

ticks = FuncFormatter(lambda x, pos: "{0:g}".format(x * 1e3) if x <= 0.02 else "")
a.yaxis.set_major_formatter(ticks)

a.set_ylabel(r"$E-E_{\text{min}}$ (meV)", labelpad=-10)
a.set_xlabel("Au-O Distance (Å)")
b.set_xlabel("Au-O Distance (Å)")

major_ticks_every(a, 0.1)
major_ticks_every(b, 0.1)

savefig(fig, "figures/aumgo")
