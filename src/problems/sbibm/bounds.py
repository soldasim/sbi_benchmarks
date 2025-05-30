from sbibm.visualisation.posterior import _LIMITS_
import torch

def get_bounds(task):
    name = process_name(task.name)
    limits = _LIMITS_[name]
    lb = torch.tensor([bounds[0] for bounds in limits])
    ub = torch.tensor([bounds[1] for bounds in limits])
    return lb, ub

def process_name(name):
    return name.split('-')[0]
