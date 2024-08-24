import argparse
import yaml
import logging
import torch
import numpy as np

def build_args():
    parser = argparse.ArgumentParser(description="URL4MC")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--missing_rate", type=float, default=0.3)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--token_id_start", type=int, default=1)
    parser.add_argument("--mask_token_id", type=int, default=0)
    parser.add_argument("--matrix_range", type=tuple, default=(-10, 10))
    parser.add_argument("--rank_k", type=int, default=2)
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=10000)
    args = parser.parse_args()
    return args

def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args

def discretize_matrix(matrix, epsilon):
    return torch.round(matrix / epsilon) * epsilon

def vectorize_matrix(matrix):
    return matrix.view(-1)

def fill_missing_values(input_matrix, missing_mask):
    M_abs_max = np.nanmax(np.abs(input_matrix))
    np.place(input_matrix, missing_mask, M_abs_max * M_abs_max)
    return input_matrix