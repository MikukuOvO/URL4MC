import torch
import torch.nn as nn

from utils import build_args, load_best_configs
from data_loading import load_data

def main(args):
    device = args.device if args.device >= 0 else "cpu"
    print(device)
    train_dataset, test_dataset = load_data(args.dataset, args)
    

if __name__ == "__main__":
    args = build_args()
    args = load_best_configs(args, "configs/transformer_configs.yaml")
    print(args)
    main(args)