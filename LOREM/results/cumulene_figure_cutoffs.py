from tbx import *
from itertools import cycle
from cumulene import results, dihedrals, reference, is_zero, get_cutoff_variants_name


def get(cutoff, mp, lr):
    name, label = get_cutoff_variants_name(cutoff, mp, lr)
    data = results[name]

    linestyle = solid if lr else dashed
    color = [red, teal, blue, cyan][mp - 1]

    return data, name, label, linestyle, color


fig, axs = plt.subplots(3, figsize=(textwidth, 12), sharex=True)

for ax in axs:
    ax.plot(dihedrals, reference, color=darkgrey, linewidth=4, label="Reference")

for i, cutoff in enumerate([2.5, 3.0, 3.5]):
    zero_labels = []
    ax = axs[i]

    ax.set_title(r"Cutoff \qty{X}{\angstrom}".replace("X", f"{cutoff:.1f}"))

    lr = True
    for mp in [1, 2]:
        data, name, label, linestyle, color = get(cutoff, mp, lr)
        if name in is_zero:
            zero_labels.append(label)
            continue
        ax.plot(
            dihedrals,
            data,
            label=label,
            color=color,
            linestyle=linestyle,
            alpha=0.8,
            linewidth=2,
        )

    lr = False
    for mp in [1, 2, 3, 4]:
        data, name, label, linestyle, color = get(cutoff, mp, lr)
        if name in is_zero:
            zero_labels.append(label)
            continue
        ax.plot(
            dihedrals,
            data,
            label=label,
            color=color,
            linestyle=linestyle,
            alpha=0.8,
            linewidth=2,
        )

    label = "\n".join(zero_labels)
    ax.plot(
        dihedrals,
        np.zeros_like(dihedrals),
        label=label,
        color=black,
        linewidth=3,
        zorder=0,
    )

    leg = ax.legend(loc="upper left")

    if len(zero_labels) == 3:
        leg.texts[-1].set_y(-1)
    elif len(zero_labels) == 2:
        leg.texts[-1].set_y(-2)

    ax.set_ylabel("Energy (eV)")
    ax.set_xlim(90, 180)
    ax.set_xticks([90, 120, 150, 180])

axs[-1].set_xlabel(r"Dihedral angle $\theta$ (Degrees)")

savefig(fig, "figures/cumulene_cutoffs")
