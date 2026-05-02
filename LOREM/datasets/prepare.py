import jax
import numpy as np

from marathon.data import datasets
from ase.io import read

from marathon import comms
from marathon.extra.hermes.data_source import prepare, DataSource

# -- AuMgO --
reporter = comms.reporter()
reporter.start("processing AuMgO")

reporter.step("load")
train = read(datasets / "AuMgO_train.xyz", format="extxyz", index=":")
valid = read(datasets / "AuMgO_valid.xyz", format="extxyz", index=":")

reporter.step("process")
prepare(
    train,
    folder=datasets / "AuMgO_train",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
)

prepare(
    valid,
    folder=datasets / "AuMgO_valid",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
)

reporter.done()

# -- bio_dimers --

reporter = comms.reporter()
reporter.start("processing bio_dimers")

reporter.step("load")
train = read(datasets / "bio_dimers_train.xyz", format="extxyz", index=":")
valid = read(datasets / "bio_dimers_valid.xyz", format="extxyz", index=":")

reporter.step("process")
prepare(
    train,
    folder=datasets / "bio_dimers_train",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
)

prepare(
    valid,
    folder=datasets / "bio_dimers_valid",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
)

reporter.done()

# -- cumulene --

reporter = comms.reporter()
reporter.start("processing cumulene")

reporter.step("load")
train = read(datasets / "cumulene_train.xyz", format="extxyz", index=":")
valid = read(datasets / "cumulene_valid.xyz", format="extxyz", index=":")


reporter.step("process")
prepare(
    train,
    folder=datasets / "cumulene_train",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
)

prepare(
    valid,
    folder=datasets / "cumulene_valid",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
)

reporter.done()

# -- NaCl --

reporter = comms.reporter()
reporter.start("processing NaCl")

reporter.step("load")
train = read(datasets / "NaCl_train.xyz", format="extxyz", index=":")
valid = read(datasets / "NaCl_valid.xyz", format="extxyz", index=":")


reporter.step("process")
prepare(
    train,
    folder=datasets / "NaCl_train",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
)

prepare(
    valid,
    folder=datasets / "NaCl_valid",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
)

reporter.done()

# -- SN2 --

reporter = comms.reporter()
reporter.start("processing SN2")

reporter.step("load")
train = read(datasets / "sn2_train.xyz", format="extxyz", index=":")
valid = read(datasets / "sn2_valid.xyz", format="extxyz", index=":")


reporter.step("process")
prepare(
    train,
    folder=datasets / "sn2_train",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
)

prepare(
    valid,
    folder=datasets / "sn2_valid",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
)

reporter.done()
