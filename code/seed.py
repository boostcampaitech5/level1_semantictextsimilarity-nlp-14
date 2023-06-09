import random
import torch
import pytorch_lightning as pl
import numpy as np
import random


# get 6 random seeds
def get_seed():
    return [random.randint(0, 2**32 - 1) for _ in range(6)]


# set seeds
def set_seed(a, b, c, d, e, f):
    torch.manual_seed(a)
    torch.cuda.manual_seed(b)
    torch.cuda.manual_seed_all(c)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(d)
    random.seed(e)
    pl.seed_everything(f, workers=True)
