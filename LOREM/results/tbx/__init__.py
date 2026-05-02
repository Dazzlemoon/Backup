import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import pyplot as plt

from .literals import *

plt.style.use(Path(__file__).parent / "plots.mlpstyle")

datadir = Path(__file__).parent / "../../experiments"
datasetdir = Path(__file__).parent / "../../datasets"


def savefile(table, file):
    if isinstance(table, list):
        text = "\n".join(table)
    else:
        text = table
    with open(file, "w") as f:
        f.write(text)


def savefig(fig, file):
    fig.savefig(file + ".png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(file + ".pdf", bbox_inches="tight", pad_inches=0.02)


def fig_and_ax(figsize=None):
    if figsize:
        fig = plt.figure(figsize=figsize, dpi=200)
    else:
        fig = plt.figure(figsize=(16, 10), dpi=200)
    ax = plt.axes()
    return fig, ax


def minor_ticks_every(ax, spacing, direction="x"):
    from matplotlib.ticker import MultipleLocator

    if direction == "x":
        ax.xaxis.set_minor_locator(MultipleLocator(spacing))
    else:
        ax.yaxis.set_minor_locator(MultipleLocator(spacing))


def major_ticks_every(ax, spacing, direction="x"):
    from matplotlib.ticker import MultipleLocator

    if direction == "x":
        ax.xaxis.set_major_locator(MultipleLocator(spacing))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(spacing))


def scale_labels(ax, scale, direction="x"):
    from matplotlib.ticker import FuncFormatter

    ticks = FuncFormatter(lambda x, pos: "{0:g}".format(x / scale))

    if direction == "x":
        ax.xaxis.set_major_formatter(ticks)
    else:
        ax.yaxis.set_major_formatter(ticks)


def reversed_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), **kwargs)


def no_ticks(ax, direction="x"):
    if direction == "x":
        ax.xaxis.set_major_locator(plt.NullLocator())
    else:
        ax.yaxis.set_major_locator(plt.NullLocator())
