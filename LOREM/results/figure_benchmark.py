from tbx import *

markers = {
    "CsCl": thiamond,
    "NaCl_cubic": diamond,
    "NaCl_primitive": star,
    "cu2o": plus,
    "wigner_bcc": dot,
    "wigner_fcc": bigdot,
    "wurtzite": diamond,
    "zincblende": diamond,
}


def read_yaml(filename):
    import yaml

    with open(filename) as stream:
        dct = yaml.safe_load(stream)

    return dct


lr = read_yaml(datadir / "lorem-benchmark/lr_times.yaml")
nolr = read_yaml(datadir / "lorem-benchmark/nolr_times.yaml")

fig, ax = fig_and_ax(figsize=(textwidth, 3))

for name, value in lr.items():
    n = int(name.split("_")[-1])
    name = "_".join(name.split("_")[:-1])
    ax.scatter([n], [1e3*value], marker=dot, color=red)

for name, value in nolr.items():
    n = int(name.split("_")[-1])
    name = "_".join(name.split("_")[:-1])
    ax.scatter([n], [1e3*value], marker=dot, color=black)

ax.set_xscale("log")
ax.set_yscale("log")

savefig(fig, "figures/benchmark")
