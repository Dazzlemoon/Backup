import sys
sys.path.append('../')

import torch
import les
from les.module import BEC

# set the same random seed for reproducibility
torch.manual_seed(0)
r = torch.rand(10, 3) * 10  # Random positions in a 10x10x10 box
r.requires_grad_(requires_grad=True)
q = torch.rand(10) * 2 - 1 # Random charges

box = torch.tensor([10.0, 10.0, 10.0])  # Box dimensions
box_full = torch.tensor([
    [10.0, 0,0],
    [0,10.0, 0], 
    [0,0,10.0]])  # Box dimensions

bec = BEC(remove_mean=False)

test = bec(q=q,
    r=r,
    cell=box_full.unsqueeze(0),
    batch=None,
    output_index=None
   )

print("BEC output shape:", test.shape)
print("BEC output:", test)
print("q", q)
