import numpy as np
import torch
import torch.nn as nn
from cvxpy import *
from utils import build_args, load_best_configs, fill_missing_values
from data_loading import load_data
from torch.utils.data import DataLoader
from tqdm import tqdm

def inference_batch(input_matrices, missing_masks):
    batch_size, n_1, n_2 = input_matrices.shape
    completed_matrices = []

    for i in tqdm(range(batch_size)):
        input_matrix = input_matrices[i].cpu().numpy()
        missing_mask = missing_masks[i].cpu().numpy()
        
        input_matrix = fill_missing_values(input_matrix, missing_mask)
        omega = np.argwhere(~missing_mask)

        X_ = Variable((n_1 + n_2, n_1 + n_2), PSD=True)
        objective = Minimize(trace(X_))

        constraints = [X_ == X_.T]
        constraints += [X_[x, y + n_1] == input_matrix[x, y] for x, y in omega]

        problem = Problem(objective, constraints)
        problem.solve(solver=CVXOPT)

        completed_matrix = X_.value[:n_1, n_1:]
        completed_matrices.append(completed_matrix)

    return torch.tensor(np.array(completed_matrices), dtype=torch.float32, device=input_matrices.device)


def main(args):
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    train_dataset, test_dataset = load_data(args.dataset, args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.MSELoss()
    total_mse_loss = 0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (feature_matrices, true_matrices, missing_masks) in enumerate(test_loader):
            feature_matrices = feature_matrices.to(device)
            true_matrices = true_matrices.to(device)
            missing_masks = missing_masks.to(device)

            predicted_matrices = inference_batch(feature_matrices, missing_masks)
            predicted_matrices = predicted_matrices.to(device)

            mse_loss = criterion(predicted_matrices, feature_matrices)
            total_mse_loss += mse_loss.item() * feature_matrices.size(0)
            num_samples += feature_matrices.size(0)
    
    avg_mse_loss = total_mse_loss / num_samples
    print(f"\Average MSE Loss: {avg_mse_loss}")
    
if __name__ == "__main__":
    args = build_args()
    args = load_best_configs(args, "configs/nuclear_configs.yaml")
    print(args)
    main(args)