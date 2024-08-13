import torch
from typing import Optional

class NuclearNormMin:
    """Class for completing matrix using nuclear norm minimization."""

    def __init__(self, M_incomplete: torch.Tensor, max_iterations: int = 1000, tolerance: float = 1e-4) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def fit_transform(self, M_incomplete: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        """Matrix completion logic."""
        return self.nn_complete(M_incomplete, missing_mask)

    def nn_complete(self, M_incomplete: torch.Tensor, missing_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Fill M_incomplete with nuclear norm minimization."""
        if missing_mask is None:
            missing_mask = torch.isnan(M_incomplete)
        
        M_incomplete = M_incomplete.clone()
        M_incomplete[missing_mask] = 0

        U, S, V = torch.svd(M_incomplete)
        M_filled = M_incomplete.clone()

        for _ in range(self.max_iterations):
            M_prev = M_filled.clone()

            # Soft-thresholding of singular values
            U, S, V = torch.svd(M_filled)
            S = torch.clamp(S - self.tolerance, min=0)
            M_filled = torch.mm(torch.mm(U, torch.diag(S)), V.t())

            # Project onto the known entries
            M_filled[~missing_mask] = M_incomplete[~missing_mask]

            # Check for convergence
            if torch.norm(M_filled - M_prev) < self.tolerance:
                break

        return M_filled