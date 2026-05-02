# `marathon`: anonymous training infrastructure in `jax`

`marathon` provides the following functionality:

- `marathon.data`: Processing `ase.Atoms` objects first into graphs and then into suitably padded batches of graphs
- `marathon.emit`: Checkpointing and logging (text, W&B), diagnostic plots
- `marathon.evaluate`: Predicting energy, forces, and stress, computing the loss as well as metrics (MAE, RMSE, R2)
- `marathon.elemental`: Computing per-element contributions with linear regression (needed to avoid floating point difficulties)
- `marathon.io`: Reading and writing of `msgpack` and `yaml`, as well as a very minimal way to turn `dataclass` instances into `dicts` and vice versa (to instantiate and store `flax.nn.Module`s)

In addition, `marathon.experimental` contains more advanced tooling:

- `marathon.experimental.hermes` provides tools to build `marathon` training pipelines with [`grain`](https://github.com/google/grain) designed to scale to large-ish datasets (up to millions of samples)

## Installation and dependencies

You'll need `jax`, probably via `pip install "jax[cuda12]"`.

Then, you should be able to run `pip install -e .`, which will install all other dependencies.

`marathon` provides a number of extras, all of which are installable via `pip install -e .[all]`. They are required to run some parts of the code. They are not automatically installed to avoid dependency resolution hell. Check the `pyproject.toml` for a list.

For convenience, `marathon` looks for an environment variable named `DATASETS` and turns it, if it exists, into a `Path` at `marathon.data.datasets`.
