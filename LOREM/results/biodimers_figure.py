from tbx import *
from biodimers import metrics, classes
from metrics import metrics as all_metrics

keys = [
    "lorem",
    "cace",
    "MACE",
    "NativePET-SR",
]

model_names = {
    "lorem": r"\textsc{Lorem}",
    "cace": r"\textsc{Cace-Les}",
    "MACE": r"\textsc{Mace}",
    "NativePET-SR": r"\textsc{Pet}",
}

num_bars = len(keys)
num_classes = len(classes)

width = 0.15
break_at = 2.75
restart_at = break_at

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(textwidth, 3.5))
fig.subplots_adjust(hspace=0.07)  # adjust space between Axes

patterns = ["XXX", "OOO", "////", "***"]
break_x = []
for i, key in enumerate(keys):
    for j, c in enumerate(classes):
        if j == 0:
            label = model_names[key]
            # ax.axhline(all_metrics["bio_dimers"][key]["energy_mae"])
        else:
            label = nolabel
        x = j - (width * num_bars / 2) + i * width + width / 2
        ax1.bar(
            x,
            metrics[key][c]["forces_mae"],
            width,
            color=model_colors[key],
            hatch=patterns[i],
            label=label,
        )
        ax2.bar(
            x,
            metrics[key][c]["forces_mae"],
            width,
            color=model_colors[key],
            hatch=patterns[i],
            label=label,
        )

        if metrics[key][c]["forces_mae"] > break_at:
            break_x.append(x)



# zoom-in / limit the view to different portions of the data
ax2.set_ylim(0, break_at)  # outliers only
ax1.set_ylim(restart_at, 28)  # most of the data

# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()


d = .24  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
              linestyle="none", color='k', mec='k', mew=1.0, clip_on=False)

ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
              linestyle="none", color='k', mec='k', mew=1.7, clip_on=False)

x = np.array(break_x)
ax2.plot(x, np.ones_like(x) * break_at, **kwargs)
ax1.plot(x, np.ones_like(x) * restart_at, **kwargs)

ax2.set_ylabel("Forces MAE (meV/Å)")
ax2.set_xlabel("Dimer class")
ax2.set_xticks(np.arange(num_classes))
ax2.set_xticklabels(classes)


major_ticks_every(ax1, 5, direction="y")
minor_ticks_every(ax1, 2.5, direction="y")
major_ticks_every(ax2, 1, direction="y")
minor_ticks_every(ax2, 0.5, direction="y")

ax2.yaxis.set_label_coords(-0.07,1)

ax1.legend(loc="upper right")

savefig(fig, "figures/biodimers")
