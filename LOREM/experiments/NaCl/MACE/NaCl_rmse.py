from mace.calculators import MACECalculator
import torch
from pathlib import Path
with open("../metrics_NaCl.py", "r") as f:
    exec(f.read())
device = "cuda" if torch.cuda.is_available() else "cpu"
calculator = MACECalculator("best_model.model", 
                            device=device,
                            enable_cueq=False)

main(calculator, Path("../../../datasets/4G/NaCl/"))