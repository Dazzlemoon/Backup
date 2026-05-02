from metatomic.torch.ase_calculator import MetatomicCalculator
import torch
from pathlib import Path
with open("../results_cumulene.py", "r") as f:
    exec(f.read())
device = "cuda" if torch.cuda.is_available() else "cpu"
calculator = MetatomicCalculator("model.pt", 
                            device=device,)

main(calculator, Path("../../../datasets/cumulene/"))
