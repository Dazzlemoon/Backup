from cace.calculators import CACECalculator
import torch
from pathlib import Path
with open("../metrics_AuMgO.py", "r") as f:
    exec(f.read())
device = "cuda" if torch.cuda.is_available() else "cpu"
calculator = CACECalculator("best_model.pth", 
                                device=device,
                                energy_key='CACE_energy',
                                forces_key='CACE_forces',
                                atomic_energies={8: -18599.43617104475, 
                                                 12: -8721.75974245582, 
                                                 13: -9877.676428588728, 
                                                 79: -688.8680063349827})

main(calculator, Path("../../../datasets/4G/Au-MgO-Al/"))
