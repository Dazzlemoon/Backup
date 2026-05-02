from cace.calculators import CACECalculator
import torch
from pathlib import Path
with open("../results_cumulene.py", "r") as f:
    exec(f.read())
device = "cuda" if torch.cuda.is_available() else "cpu"
calculator = CACECalculator("best_model.pth", 
                            device=device,
                            energy_key='CACE_energy',
                            forces_key='CACE_forces',)

main(calculator, Path("../../../datasets/cumulene/"))