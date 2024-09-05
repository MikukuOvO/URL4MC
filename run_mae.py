import torch
import wandb
import random
import math
import torch.nn as nn
from tqdm import tqdm

from utils import build_args, load_best_configs
from data_loading import load_data
from torch.utils.data import DataLoader
from models.mae import MAE_ViT

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
            
            complete_vec = model(feature_vec)  # 忽略返回的mask
            
            mse = nn.MSELoss(reduction='none')(complete_vec, true_vec)
            
            # 对每个样本取平均
            mse = mse.sum(dim=1) / len(complete_vec[0])
            
            total_mse += mse.sum().item()
            num_samples += feature_vec.size(0)
    
    avg_mse = total_mse / num_samples
    return avg_mse

def train(model, optim, data_loader, lr_scheduler, args):
    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    epoch_bar = tqdm(range(args.num_epochs), desc="Training")
    optim.zero_grad()
    step_count = 0
    for epoch in epoch_bar:
        model.train()
        losses = []
        for data in data_loader:
            step_count += 1
            feature_vec, true_vec, missing_vec = data
            feature_vec = feature_vec.to(args.device)
            true_vec = true_vec.to(args.device)
            missing_vec = missing_vec.to(args.device)
            complete_vec, mask = model(feature_vec)
            calc_mask = (1 - missing_vec) * (1 - mask)
            loss = nn.MSELoss()(complete_vec * calc_mask, true_vec * calc_mask)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        epoch_bar.set_postfix({"Loss": sum(losses) / len(losses)})
        wandb.log({"Loss": sum(losses) / len(losses)})

        if epoch % 50 == 0:
            mse = evaluate(model, data_loader, args)
            print(f"Epoch {epoch}: {mse}")
            wandb.log({"MSE": mse})

    return model
    
def main(args):
    device = args.device if args.device >= 0 else "cpu"
    print(device)
    data_loader = load_data(args.dataset, args)

    model = MAE_ViT(args.label_file,patch_size=args.patch_size,mask_ratio=args.mask_ratio,
                        mlp_ratio=args.mlp_ratio,emb_dim=args.emb_dim,decoder_layer=args.decoder_layers,
                        num_groups=args.num_groups)  
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.num_epochs * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=False)
    optim.zero_grad()
    model = train(model, optim, data_loader, lr_scheduler, args)
    final_mse = evaluate(model, data_loader, args)
    print(f"Final Acc: {final_mse}")
    wandb.log({"Final Acc": final_mse})

if __name__ == "__main__":
    args = build_args()
    args = load_best_configs(args, "configs/mae_configs.yaml")
    print(args)
    random.seed(args.seed)
    wandb.init(project="URL4MC", config=args)
    main(args)