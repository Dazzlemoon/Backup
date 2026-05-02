from tbx import *

def read_yaml(filename):
    import yaml

    with open(filename) as stream:
        dct = yaml.safe_load(stream)

    return dct

_lr = read_yaml(datadir / "lorem-benchmark/lr_times.yaml")
_nolr = read_yaml(datadir / "lorem-benchmark/nolr_times.yaml")

lr = {}
for k, v in _lr.items():
    n = int(k.split("_")[-1])
    name = "_".join(k.split("_")[:-1])

    lr[(n, name)] = v

nolr = {}
for k, v in _nolr.items():
    n = int(k.split("_")[-1])
    name = "_".join(k.split("_")[:-1])

    nolr[(n, name)] = v
