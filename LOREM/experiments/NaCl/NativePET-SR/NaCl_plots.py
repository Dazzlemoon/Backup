from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator
import torch
from pathlib import Path
with open("../results_NaCl.py", "r") as f:
    exec(f.read())
device = "cuda" if torch.cuda.is_available() else "cpu"
calculator = MetatensorCalculator("best_model.pt", 
                            device=device,)
                            
main(calculator, Path("../../../datasets/4G/NaCl/"))
