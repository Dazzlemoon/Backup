from pathlib import Path

with open(Path(__file__).parent / "../experiments/results_AuMgO-wetting-vs-non-wetting.py", "r") as f:
    exec(f.read())

from calculator import Calculator
from marathon.data import datasets

calc = Calculator.from_checkpoint("run/checkpoints/R2_E+F", add_offset=True)

main(calc, datasets)
