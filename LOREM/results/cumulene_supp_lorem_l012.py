from tbx import *
from itertools import cycle
from cumulene import results, dihedrals, reference, is_zero

fig, ax = fig_and_ax(figsize=(textwidth, 3))

ax.plot(dihedrals, reference, color=darkgrey, linewidth=4, label="Reference")


model_names = {
    "lorem-nolr-mp1": r"\textsc{Lorem}, No LR",
    "lorem-l0": r"\textsc{Lorem}, LR $l=0$",
    "lorem-l1": r"\textsc{Lorem}, LR $l=1$",
    "lorem": r"\textsc{Lorem}, LR $l=2$",
}


model_colors = {
    "lorem": red,
    "lorem-l1": teal,
    "lorem-l0": orange,
    "lorem-nolr-mp1": blue,
}


linestyles = {
    "lorem": solid,
    "lorem-l1": dotted,
    "lorem-l0": dashed,
    "lorem-nolr-mp1": finedot,
}

linewidths = {
    "lorem": 2,
    "lorem-nolr-mp1": 4,
    "lorem-l0": 2.5,
    "lorem-l1": 1,
}

keys = model_names.keys()

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

savefig(fig, "figures/cumulene_supp_lorem_l012")
