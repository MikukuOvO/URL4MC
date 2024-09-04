import argparse
import yaml
import logging
import torch
import numpy as np

def build_args():
    parser = argparse.ArgumentParser(description="URL4MC")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="synthetic")
    parser.add_argument("--missing_rate", type=float, default=0.3)
    parser.add_argument("--matrix_range", type=tuple, default=(-10, 10))
    parser.add_argument("--rank_k", type=int, default=2)
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--num_features", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--max_device_batch_size", type=int, default=4000)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--warmup_epoch", type=int, default=200)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layers", type=int, default=12)
    parser.add_argument("--encoder_heads", type=int, default=3)
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--decoder_heads", type=int, default=3)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument('--scale_lo', type=float, default=0.3)
    parser.add_argument('--scale_high', type=float, default=1.0)
    parser.add_argument('--ratio_lo', type=float, default=1.0)
    parser.add_argument('--ratio_high', type=float, default=1.0)
    parser.add_argument('--label_file', type=str, default='tmp/tmp_param/label_perm_true.npy')
    parser.add_argument('--emb_dim', type=int, default=192)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    parser.add_argument('--num_groups', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--flip', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
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

def add_dimension(batch):
    return batch.unsqueeze(0)

def fill_missing_values(input_matrix, missing_mask):
    M_abs_max = np.nanmax(np.abs(input_matrix))
    np.place(input_matrix, missing_mask, M_abs_max * M_abs_max)
    return input_matrix