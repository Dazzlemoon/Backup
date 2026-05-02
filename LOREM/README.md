# Learning Long-Range Representations with Equivariant Messages: Anonymised supplemental materials

This archive contains anonymised supplemental material for the TMLR submission "Learning Long-Range Representations with Equivariant Messages" (LOREM) for short. It contains all the code and data required to reconstruct the results from the paper. The repository is assigned the DOI 10.5281/zenodo.17789350.

The archive is structured as follows:

- `lorem/` contains the source code of the model and other related infrastructure scripts
- `experiments/` contains the experiment-specific scripts and results with different models
- `results/` contains scripts that process the data from `experiments/` into tables and figures for the paper
- `marathon/` is the the infrastructure library used for training LOREM

## Setup/Installation

- `jax` must be installed
- https://github.com/lab-cosmo/jax-pme must be installed
- additional dependencies: `scikit-learn grain e3x mmap_ninja wandb`
- `lorem/` must be in the `$PYTHONPATH`
- `marathon/` most be installed with `pip install -e .`
- datasets must be prepared as indicated in `datasets/` and you must export the environment variable `$DATASETS` to point to the `datasets/` folder

## Running training/evaluation

Training is simply `python /path/to/lorem/run.py`, expecting a `settings.yaml` and `model.yaml` in the current working directory. The results are stored in `run/`. Note that you cannot restore the training state from this repo, as it contains personally identifying information and has therefore been purged. Model checkpoints are however intact. For this reason, if you wish to train a model, you need to either rename or remove `run/` to avoid restoring the checkpoint, which will fail.

Computing results or metrics is done via `python /path/to/lorem/{results/metrics}_XXX.py`, from the same working directory. There is also a shell script in `experiments/` that will run all the post-processing.

## Environment for experiments

Experiments were performed with `Python 3.11.7` in the following environment:

```
absl-py                  2.3.1
aiofiles                 24.1.0
annotated-types          0.7.0
array_record             0.8.1
ase                      3.26.0
attrs                    25.3.0
certifi                  2025.8.3
charset-normalizer       3.4.3
chex                     0.1.90
click                    8.2.1
cloudpickle              3.1.1
comms                    0.1.0
contourpy                1.3.3
cycler                   0.12.1
dm-tree                  0.1.9
e3x                      1.0.2
etils                    1.13.0
flax                     0.11.1
fonttools                4.59.1
fsspec                   2025.7.0
gitdb                    4.0.12
GitPython                3.1.45
grain                    0.2.12
humanize                 4.13.0
idna                     3.10
importlib_resources      6.5.2
iniconfig                2.1.0
jax                      0.7.1
jax-cuda12-pjrt          0.7.1
jax-cuda12-plugin        0.7.1
jax-pme                  0.1.0a1
jaxlib                   0.7.1
jaxtyping                0.3.2
joblib                   1.5.1
kiwisolver               1.4.9
llvmlite                 0.44.0
marathon                 0.1.0
markdown-it-py           4.0.0
matplotlib               3.10.5
matscipy                 1.1.1
mdurl                    0.1.2
ml_dtypes                0.5.3
mmap_ninja               0.7.4
more-itertools           10.7.0
mpmath                   1.3.0
msgpack                  1.1.1
nest-asyncio             1.6.0
numba                    0.61.2
numpy                    1.26.4
nvidia-cublas-cu12       12.9.1.4
nvidia-cuda-cupti-cu12   12.9.79
nvidia-cuda-nvcc-cu12    12.9.86
nvidia-cuda-nvrtc-cu12   12.9.86
nvidia-cuda-runtime-cu12 12.9.79
nvidia-cudnn-cu12        9.12.0.46
nvidia-cufft-cu12        11.4.1.4
nvidia-cusolver-cu12     11.7.5.82
nvidia-cusparse-cu12     12.5.10.65
nvidia-nccl-cu12         2.27.7
nvidia-nvjitlink-cu12    12.9.86
nvidia-nvshmem-cu12      3.3.24
opt_einsum               3.4.0
optax                    0.2.5
orbax-checkpoint         0.11.23
packaging                25.0
pillow                   11.3.0
pip                      23.2.1
platformdirs             4.3.8
pluggy                   1.6.0
protobuf                 6.32.0
pydantic                 2.11.7
pydantic_core            2.33.2
Pygments                 2.19.2
pyparsing                3.2.3
pytest                   8.4.1
python-dateutil          2.9.0.post0
PyYAML                   6.0.2
requests                 2.32.5
rich                     14.1.0
ruff                     0.12.10
scikit-learn             1.7.1
scipy                    1.16.1
sentry-sdk               2.35.0
setuptools               65.5.0
simplejson               3.20.1
six                      1.17.0
smmap                    5.0.2
sympy                    1.14.0
tensorstore              0.1.76
threadpoolctl            3.6.0
toolz                    1.0.0
tqdm                     4.67.1
treescope                0.1.10
typing_extensions        4.14.1
typing-inspection        0.4.1
urllib3                  2.5.0
vesin                    0.3.7
wadler_lindig            0.1.7
wandb                    0.21.1
wrapt                    1.17.3
zipp                     3.23.0
```