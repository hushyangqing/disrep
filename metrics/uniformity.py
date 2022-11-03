import torch
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from models.utils import model_loader, sprites_label_to_action
from datasets.datasets import datasets, dataset_meta, set_to_loader
from torch.nn import functional as F
from torch import nn, optim
import tqdm
import collections
import itertools
from torch.utils.data import DataLoader, Dataset
import random
import gc

def getRandomSamplesOnNSphere(N , R , numberOfSamples):
    # Return 'numberOfSamples' samples of vectors of dimension N 
    # with an uniform distribution on the (N-1)-Sphere surface of radius R.
    # RATIONALE: https://mathworld.wolfram.com/HyperspherePointPicking.html
    
    X = np.random.default_rng().normal(size=(numberOfSamples , N))

    return R / np.sqrt(np.sum(X**2, 1, keepdims=True)) * X

def distance(data):
    data = torch.tensor(data)
    res = torch.cdist(data, data).reshape(-1)
    exp = [torch.exp(-x) for x in res if x!=0]
    exp = torch.tensor(exp)
    return exp.mean()

class Uniformity:
    def __init__(self, ds, val_ds=None, bs=1024, nactions=4, ntrue_actions=4, verbose=False):
        self.ds = ds
        self.val_ds = val_ds if val_ds is not None else ds
        self.bs = bs
        self.nactions = nactions
        self.ntrue_actions = ntrue_actions
        self.verbose = verbose
    
    def __call__(self, pymodel):
        gc.collect()
        pymodel_state = pymodel.training
        pymodel.eval()
        valloader = DataLoader(self.ds, self.bs, shuffle=True, num_workers=7)
        out = measure_uniformity(pymodel, self.ntrue_actions, self.verbose, valloader)
        gc.collect()
        return out

def measure_uniformity(vae, ntrue_actions, verbose, valloader):
    z_mean = []
    z_dist = []
    for i, data in enumerate(valloader):
        with torch.no_grad():
            (img, label), target = data
            img = img.cuda()
            z = vae.unwrap(vae.encode(img))[0].detach()
            z_norm = torch.linalg.norm(z, dim=1)
            z_mean.append(z_norm.mean())
            z_dist.append(distance(z))
    zmean = torch.tensor(z_mean).mean()
    base_data = getRandomSamplesOnNSphere(ntrue_actions, zmean.cpu(), len(valloader))
    base_dis = distance(base_data)
    zdist = torch.tensor(z_dist).mean()
    print('z_exp_', zdist)
    print('base_exp_', base_dis)
    print('uni_',zdist/base_dis)

    return {'uniformity': z_dist/base_dis,
        'z_exp': zdist,
        'base_exp': basedis}