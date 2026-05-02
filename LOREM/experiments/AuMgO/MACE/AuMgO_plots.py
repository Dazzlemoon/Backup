from mace.calculators import MACECalculator
import torch
from pathlib import Path
with open("../results_AuMgO.py", "r") as f:
    exec(f.read())
device = "cuda" if torch.cuda.is_available() else "cpu"
calculator = MACECalculator("best_model.model", 
                            device=device,
                            enable_cueq=False) # cuEquivariance acceleration

main(calculator, Path("../../../datasets/4G/Au-MgO-Al/"))