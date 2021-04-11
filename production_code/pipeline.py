import numpy as np
import matplotlib.pyplot as plt

import genetic_gaussian_search as search

def load_seed(filename):
    s = np.load(filename)
    return s

def QCTRL_loss(x):
    return x.sum()

#initialize parameters
seed = load_seed("H_START.npy") #load_seed("NOT_START.npy")
segment_count = seed.shape[0]
params = {
    "gaussian_params" : {
        "amp_mean": 0,
        "ph_mean": 0,
        "amp_sd": 1,
        "ph_sd": 1,
    },
    "iterations": 100,
    "population size": 10,
    "crossover prob": 0.9,
    "mutation prob amp": 1/(segment_count*2),
    "mutation prob phase": 1/(segment_count*2)
}

best, score = search.genetic_gaussian_search(QCTRL_loss, seed, params)
print('Done! THE BEST IS:')
print('f(%s \n) = %f' % (search.np_2d_print(best), score))