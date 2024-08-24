import numpy as np
import torch
import torch.nn as nn
from cvxpy import *
from utils import build_args, load_best_configs, fill_missing_values
from data_loading import load_sync_data, get_matrix_mask

def inference(input_matrix, missing_mask):
    input_matrix = fill_missing_values(input_matrix, missing_mask)
    omega = [(i, j) for i in range(input_matrix.shape[0]) for j in range(input_matrix.shape[1]) if missing_mask[i, j] == False]

    n_1 = input_matrix.shape[0]
    n_2 = input_matrix.shape[1]
    X_ = Variable((n_1 + n_2, n_1 + n_2), PSD=True)
    objective = Minimize(trace(X_))

    constraints = [(X_ == X_.T)]
    for i, j in omega:
        constr = (X_[i, j + n_1] == input_matrix[i, j])
        constraints.append(constr)
    problem = Problem(objective, constraints)
    problem.solve(solver=CVXOPT)

    X0 = X_.value
    completed_matrix = X_.value[:n_1, n_1:]

    return completed_matrix

def main(args):
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    print(device)
    feature_matrix, missing_mask = load_sync_data(args.num_nodes, args.rank_k, args.missing_rate)
    input_matrix = get_matrix_mask(feature_matrix, missing_mask)
    predicted_matrix = inference(input_matrix, missing_mask)

    feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32).to(device)
    predicted_matrix = torch.tensor(predicted_matrix, dtype=torch.float32).to(device)
    
    criterion = nn.MSELoss()
    mse_loss = criterion(predicted_matrix, feature_matrix)
    print(f"Inference MSE Loss: {mse_loss.item()}")
    
if __name__ == "__main__":
    args = build_args()
    args = load_best_configs(args, "configs/nuclear_configs.yaml")
    print(args)
    main(args)