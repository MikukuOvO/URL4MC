import random
import numpy as np
import torch
from numpy import ndarray
from csmc.settings import T
import scipy

def incoherent_subspace(n: int, r: int) -> ndarray:
    """Create incoherent subspace with Hadamard transform."""
    return scipy.linalg.hadamard(n)[:, :r]

def incoherent_matrix(n1: int, n2: int, r: int) -> ndarray:
    """Create incoherent matrix."""
    U = incoherent_subspace(n1, r)
    V = incoherent_subspace(n2, r)
    output = np.zeros((n1, n2))
    for i in range(r):
        sigma = random.uniform(0, 100)
        output += sigma * np.outer(U[:, i], V[:, i])
    return output

def create_rank_k_dataset(
        n_rows: int = 5,
        n_cols: int = 5,
        k: int = 3,
        fraction_missing: float = 0.7,
        symmetric: bool = False,
        gaussian: bool = False,
        with_replacement: bool = False,
        noise: float = 0,
        numlib: str = "numpy") -> tuple:
    """Generate synthetic data."""
    if gaussian:
        x = torch.randn(n_rows, k)
        y = torch.randn(k, n_cols)
        if noise:
            noise_x = torch.empty(n_rows, k)
            noise_y = torch.empty(k, n_cols)
            torch.nn.init.sparse_(noise_x, sparsity=1 - noise)
            torch.nn.init.sparse_(noise_y, sparsity=1 - noise)
            x += noise_x
            y += noise_y
        XY = x @ y
    else:
        XY = torch.tensor(incoherent_matrix(n_rows, n_cols, k), dtype=torch.float32)

    indices = torch.cartesian_prod(torch.arange(XY.shape[0]), torch.arange(XY.shape[1]))

    if with_replacement:
        omega = indices[torch.randint(len(indices), (int((1 - fraction_missing) * len(indices)),))]
    else:
        omega = indices[torch.randperm(len(indices))[:int((1 - fraction_missing) * len(indices))]]

    mask_array = torch.zeros(XY.shape, dtype=torch.int)

    if symmetric:
        assert n_rows == n_cols
        XY = 0.5 * XY + 0.5 * XY.T

    XY_incomplete = torch.zeros(XY.shape)
    for idx in omega:
        XY_incomplete[idx[0], idx[1]] = XY[idx[0], idx[1]]
        mask_array[idx[0], idx[1]] = 1

    mask_array = mask_array.bool()
    missing_mask = ~mask_array
    XY_incomplete[missing_mask] = float('nan')
    return XY, XY_incomplete, omega, mask_array

def remove_pixels_uniformly(
        X: T,
        missing_part: float = 0.9,
        random_seed=0) -> np.ndarray:
    if isinstance(X, np.ndarray):
        X_missing = np.copy(X).astype("float32")
        index_nan = np.random.choice(X.size, int(missing_part * X.size), replace=False)
        X_missing.ravel()[index_nan] = np.nan
    else:
        X_missing = torch.clone(X)
        missing_mask = torch.ones(X_missing.shape)
        m = torch.nn.Dropout(p=missing_part)
        missing_mask = m(missing_mask).type(torch.bool)
        X_missing[missing_mask] = torch.nan
    return X_missing