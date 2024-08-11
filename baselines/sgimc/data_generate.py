import os
import numpy as np
from sgimc.utils import make_imc_data
import gzip
import pickle

PATH_DATA = 'data'

if not os.path.exists(PATH_DATA):
    os.makedirs(PATH_DATA)

# Random state.
random_state = np.random.RandomState(0x0BADCAFE)

# Data configuration.
elements = np.arange(0.001, 0.02, 0.0015)
n_samples, n_objects = 800, 1600
n_rank = 25
n_features = 100

scale = 0.05
noise = 0.10

# Making artificial data.
X, W_ideal, Y, H_ideal, R_noisy, R = make_imc_data(
    n_samples, n_features, n_objects, n_features,
    n_rank, scale=(scale, scale), noise=scale*noise,
    binarize=False,
    random_state=random_state,
    return_noisy_only=False)

data = (X, Y, R, R_noisy)

# Saving data.
filename = os.path.join(PATH_DATA, 'data.gz')
with gzip.open(filename, "wb+", 4) as fout:
    pickle.dump(data, fout)
