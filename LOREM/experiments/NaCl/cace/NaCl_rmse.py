from cace.calculators import CACECalculator
import torch
from pathlib import Path
with open("../metrics_NaCl.py", "r") as f:
    exec(f.read())
device = "cuda" if torch.cuda.is_available() else "cpu"
calculator = CACECalculator("best_model.pth", 
                            device=device,
                            energy_key='CACE_energy',
                            forces_key='CACE_forces',
                            atomic_energies={11: -4417.07609365649, 17: -12516.880649933015})

main(calculator, Path("../../../datasets/4G/NaCl/"))