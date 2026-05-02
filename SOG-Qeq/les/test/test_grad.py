import sys
sys.path.append('../')

import torch
import les
from les.util import grad

# set the same random seed for reproducibility
torch.manual_seed(0)
r = torch.rand(8, 3) * 10  
r.requires_grad_(requires_grad=True)

y1 = torch.sum(r, dim=0)
y2 = torch.sum(r**2., dim=0)
y = torch.stack([y1, y2]).T

# test the gradient
g = grad(x=r, y=y)

print('Gradient of y1 with respect to r:')
print(g)
