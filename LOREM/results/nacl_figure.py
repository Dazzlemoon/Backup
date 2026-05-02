from tbx import *
from itertools import cycle
from nacl import *

fig, axs = plt.subplots(
    1, 2, figsize=(textwidth, 2.5), sharey=True, constrained_layout=True
)

a = axs[0]
b = axs[1]


def plot(
    ax,
    distance,
    energy,
    offset,
    label=nolabel,
    color=black,
    linestyle=solid,
    alpha=0.8,
    linewidth=2,
):
    if "4G" in label:
        distance = distance[20:90]
        energy = energy[20:90]

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

    ax.plot(
        [distance[minimum]],
        [value],
        # energy[minimum] + 0.01,
        color=color,
        # linestyle=linestyle,
        # linewidth=linewidth,
        alpha=alpha,
        marker=diamond,
    )


keys = [
    "lorem",
    # "aeres-l0",
    "cace",
    # "SOAPBPNN-LR",
    "4G_NN",
    # "NativePET-LR",
    "MACE",
    "NativePET-SR",
]
keys = reversed(keys)

for i, key in enumerate(keys):
    linestyle = model_linestyles[key]

    offset = i * 0.01

    a.axhline(offset, color=black, alpha=0.1, linewidth=0.5)
    b.axhline(offset, color=black, alpha=0.1, linewidth=0.5)

    plot(
        a,
        reference_distance_na8,
        reference_energy_na8,
        offset,
        color=darkgrey,
        label="Reference" if i == 0 else nolabel,
        linewidth=5,
    )
    plot(
        b,
        reference_distance_na9,
        reference_energy_na9,
        offset,
        color=darkgrey,
        label="Reference" if i == 0 else nolabel,
        linewidth=5,
    )

    plot(
        a,
        distance_na8,
        results_na8[key],
        offset,
        label=model_names[key],
        color=model_colors[key],
        linestyle=linestyle,
        alpha=0.8,
    )

    plot(
        b,
        distance_na9,
        results_na9[key],
        offset,
        label=model_names[key],
        color=model_colors[key],
        linestyle=linestyle,
        alpha=0.8,
    )

split = 3
lines, labels = a.get_legend_handles_labels()

lines = [lines[0]] + list(reversed(lines[1:]))
labels = [labels[0]] + list(reversed(labels[1:]))

# a.legend(lines[:split], labels[:split], loc="upper center")
# b.legend(lines[split:], labels[split:], loc="upper center")

a.set_title(r"Na$_8$Cl$_8^+$")
b.set_title(r"Na$_9$Cl$_8^+$")

# a.set_xlim(3.2, 3.66)
a.set_xlim(3.48 - 0.15, 3.48 + 0.15)
b.set_xlim(3.37 - 0.15, 3.37 + 0.15)
# b.set_xlim(3.2, 3.66)
a.set_ylim(-0.005, 0.071)

a.set_xlabel(r"Na-Na Distance $d$ (Å)")
b.set_xlabel(r"Na-Na Distance $d$ (Å)")

major_ticks_every(a, 0.1)
major_ticks_every(b, 0.1)


major_ticks_every(a, 0.01, direction="y")
minor_ticks_every(a, 0.005, direction="y")

from matplotlib.ticker import FuncFormatter

ticks = FuncFormatter(lambda x, pos: "{0:g}".format(x * 1e3) if x <= 0.01 else "")
a.yaxis.set_major_formatter(ticks)

a.set_ylabel(r"$E-E_{\text{min}}$ (meV)", labelpad=-10, loc="top")


savefig(fig, "figures/nacl")
