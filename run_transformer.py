import torch
import wandb
import torch.nn as nn
from tqdm import tqdm

from utils import build_args, load_best_configs
from data_loading import load_data
from torch.utils.data import DataLoader
from models.transformer import Transformer

def evaluate(model, data_loader, args):
    model.eval()
    total_mse = 0
    num_samples = 0
    
    with torch.no_grad():
        for data in data_loader:
            feature_vec, true_vec, missing_vec = data
            feature_vec = feature_vec.to(args.device)
            true_vec = true_vec.to(args.device)
            missing_vec = missing_vec.to(args.device)
            
            complete_vec, _ = model(feature_vec)  # 忽略返回的mask
            
            mse = nn.MSELoss(reduction='none')(complete_vec, true_vec)
            
            # 对每个样本取平均
            mse = mse.sum(dim=1) / len(complete_vec[0])
            
            total_mse += mse.sum().item()
            num_samples += feature_vec.size(0)
    
    avg_mse = total_mse / num_samples
    return avg_mse

def train(model, optim, data_loader, args):
    epoch_bar = tqdm(range(args.num_epochs), desc="Training")
    for epoch in epoch_bar:
        model.train()
        losses = []
        for data in data_loader:
            feature_vec, true_vec, missing_vec = data
            feature_vec = feature_vec.to(args.device)
            true_vec = true_vec.to(args.device)
            missing_vec = missing_vec.to(args.device)
            optim.zero_grad()
            complete_vec, mask = model(feature_vec)
            calc_mask = (1 - missing_vec) * (1 - mask)
            loss = nn.MSELoss()(complete_vec * calc_mask, true_vec * calc_mask)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        epoch_bar.set_postfix({"Loss": sum(losses) / len(losses)})
        wandb.log({"Loss": sum(losses) / len(losses)})

        if epoch % 200 == 0:
            mse = evaluate(model, data_loader, args)
            print(f"Epoch {epoch}: {mse}")
            wandb.log({"MSE": mse})

    return model
    
def main(args):
    device = args.device if args.device >= 0 else "cpu"
    print(device)
    data_loader = load_data(args.dataset, args)

    model = Transformer(args.num_nodes, args.hidden_size, args.encoder_heads, args.encoder_layers, args.mask_ratio) 
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model = train(model, optim, data_loader, args)
    mse = evaluate(model, data_loader, args)
    print(f"Final Acc: {mse}")
    wandb.log({"Final Acc": mse})

if __name__ == "__main__":
    args = build_args()
    args = load_best_configs(args, "configs/transformer_configs.yaml")
    print(args)
    wandb.init(project="URL4MC", config=args)
    main(args)