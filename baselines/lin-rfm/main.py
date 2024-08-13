import torch
import numpy as np
import random
from data_loading import get_data
from linear_rfm import linear_rfm
from svd_free_lin_rfm import rfm

SEED = 6

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)


def main():

    d = 500
    r = 5

    NUM_RFM_ITERS = 3000

    num_obs = 10000
    reg = 5e-2

    Y, unmasked = get_data(d, r, num_obs)

    loss = linear_rfm(Y, unmasked, NUM_RFM_ITERS,
                                 reg=reg)
    print("Linear RFM Alpha = 1: ", loss)

    loss = rfm(Y, unmasked, NUM_RFM_ITERS, reg=reg)
    print("Linear RFM Alpha = 1/2: ", loss)

if __name__ == "__main__":
    main()