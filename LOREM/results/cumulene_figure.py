from tbx import *
from itertools import cycle
from cumulene import results, dihedrals, reference, is_zero

fig, ax = fig_and_ax(figsize=(textwidth, 3))

ax.plot(dihedrals, reference, color=darkgrey, linewidth=4, label="Reference")

linestyles = {
    "lorem": solid,
    "cace": dotted,
    "SOAPBPNN-LR": finedash,
    "MACE": dotted,
    "NativePET-SR": dotted,
}

linewidths = {
    "lorem": 2,
    "aeres-l0": 2,
    "cace": 3,
    "SOAPBPNN-LR": 3,
    "MACE": 2,
    "NativePET-SR": 2,
}

keys = [
    "lorem",
    # "aeres-l0",
    "cace",
    # "SOAPBPNN-LR",
    # "NativePET-LR",
    "MACE",
    "NativePET-SR",
]

zero_labels = []

for i, key in enumerate(keys):
    data = results[key]

    # if key in is_zero:
    #     zero_labels.append(model_names[key])

    # else:
    ax.plot(
        dihedrals,
        data,
        label=model_names[key],
        color=model_colors[key],
        linestyle=linestyles[key],
        alpha=0.8,
        linewidth=linewidths[key],
    )

# label = "\n".join(zero_labels)
# ax.plot(
#     dihedrals,
#     np.zeros_like(dihedrals),
#     label=label,
#     color=black,
#     linewidth=3,
#     zorder=0,
# )

leg = ax.legend(loc="upper left", ncol=2, columnspacing=1.5)
# leg.texts[3].set_y(-1.5)

ax.set_xlim(90, 180)
ax.set_xticks([90, 120, 150, 180])

ax.set_ylabel("Energy (eV)")
ax.set_xlabel(r"Dihedral angle $\theta$ (Degrees)")

savefig(fig, "figures/cumulene")
