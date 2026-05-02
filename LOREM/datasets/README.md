# Datasets

This contains datasets used in the paper in `.xyz` (extended) format. All units are eV and Å; energies are not modified (composition baseline is removed elsewhere).

## Preparing data for training

`marathon` uses a custom data format for efficient reading with `grain`, based on `mmap`-ed arrays. The `prepare.py` script in this folder runs the necessary processing to generate this from the input `.xyz` files presented here. **Note that it is required for you to set `$DATASETS` to a reasonable folder into which to write the processed datasets.**

## Notes on the datasets

4G datasets (AuMgO, NaCl) are based on https://github.com/BingqingCheng/cace-lr-fit/tree/main, commit `7386d1a` with splits re-generated from the random seed used in the CACE-LES publication.

***

Data for cumulene is obtained from https://zenodo.org/records/14750286. Subsets have been sampled randomly.

***

Biodimers data is likely the split used in the `torch-pme` paper. All structures contain a two-latter label that have been used to stratify the training and test sets. 

The counts for the subsets and the train/valid split are:

```
PA: total=2928. train=2342   valid=585
AA: total=7357. train=5885   valid=1471
PP: total=1128. train=902    valid=225
CA: total=1470. train=1176   valid=293
CC: total=1288. train=1030   valid=257
CP: total=1869. train=1495   valid=373
```

***

The SN2 data is obtained from [Zenodo](https://zenodo.org/records/14750286). Here's the snippet to convert the `.npz` to `ase.Atoms`, reversing the shifts done in the zenodo release:

```python

from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms

data = np.load("sn2_reactions_shifted.npz")
shifts = {
    1: -3.5921101,
    6: -1.71485694,
    9: -0.97762923,
    17: -0.23682422,
    35: 0.08449515,
    53: 0.17498543,
}

total = data["positions"].shape[0]
node_mask = np.array(data["node_mask"])
positions = np.array(data["positions"])
forces = np.array(data["forces"])
energy = np.array(data["energy"])
atomic_numbers = np.array(data["atomic_numbers"])

def get_atoms(i):
    mask = node_mask[i]
    pos = positions[i][mask]
    f = forces[i][mask]
    e = energy[i][0]
    z = atomic_numbers[i][mask]

    offset = sum([shifts[Z] for Z in z])

    a = Atoms(positions=pos, numbers=z)
    c = SinglePointCalculator(a, forces=f, energy=e + offset)
    a.calc = c

    return a

# the split was: 405k training, 5k valid, 42708 test

```

The minimum energy paths are obtained the same way, except that the shift is omitted.

***

To obtain "inner" train/validation splits for AuMgO and NaCl, this snippet was used:

```python
import jax
from marathon.data import get_splits
key = jax.random.key(42)
idx_train, idx_valid, idx_test = get_splits(4500, 4000, 500, 0, key)
```
