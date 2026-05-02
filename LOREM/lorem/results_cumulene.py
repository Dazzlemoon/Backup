from pathlib import Path

with open(Path(__file__).parent / "../experiments/results_cumulene.py", "r") as f:
    exec(f.read())

from calculator import Calculator
from marathon.data import datasets

calc = Calculator.from_checkpoint("run/checkpoints/R2_E+F", add_offset=False)

main(calc, datasets)
