import torch
from torch.nn import Parameter

a = Parameter(torch.tensor([0.]))

v = torch.log(a)
v.backward()
print(a.grad)
