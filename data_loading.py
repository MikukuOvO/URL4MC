import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, RandomResizedCrop


class ColumnwiseDataset(Dataset):
    def __init__(self, num_nodes, num_features, rank_k, missing_rate=0.1, seed=42):
        self.datasetname = "synthetic"
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.rank_k = rank_k
        self.missing_rate = missing_rate
        self.seed = seed
        np.random.seed(seed)
        self.load_or_generate()

    def load_or_generate(self):
        dataset_path = self._get_dataset_path()
        if os.path.exists(dataset_path):
            print(f"加载数据集：{dataset_path}")
            self.load_dataset(dataset_path)
        else:
            print(f"在 {dataset_path} 未找到数据集，正在生成新数据集")
            self.generate()
            self.save_dataset(dataset_path)

    def _get_dataset_path(self):
        return os.path.join("dataset", self.datasetname, 
                            f"{self.datasetname}_{self.num_nodes}_{self.num_features}_{self.rank_k}_{self.missing_rate}.pt")

    def generate(self):
        U = np.random.randn(self.num_nodes, self.rank_k).astype(np.float32)
        V = np.random.randn(self.rank_k, self.num_features).astype(np.float32)
        true_matrix = np.dot(U, V)
        missing_mask = (np.random.rand(self.num_nodes, self.num_features) < self.missing_rate).astype(np.float32)
        feature_matrix = true_matrix.copy()
        feature_matrix[missing_mask.astype(bool)] = 0

        self.true_matrix = torch.from_numpy(true_matrix)
        self.feature_matrix = torch.from_numpy(feature_matrix)
        self.missing_mask = torch.from_numpy(missing_mask)

    def save_dataset(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'true_matrix': self.true_matrix,
            'feature_matrix': self.feature_matrix,
            'missing_mask': self.missing_mask
        }, path)
        print(f"新数据集已保存至 {path}")

    def load_dataset(self, path):
        data = torch.load(path)
        self.true_matrix = data['true_matrix']
        self.feature_matrix = data['feature_matrix']
        self.missing_mask = data['missing_mask']

    def __getitem__(self, index):
        data = self.feature_matrix[:, index].reshape(1, 32, 32)
        # 填充到 3 个通道
        feature = torch.cat([data, data, data], dim=0)
        # 对 true_value 和 mask 进行相同操作
        true_value = torch.cat([self.true_matrix[:, index].reshape(1, 32, 32)] * 3, dim=0)
        mask = torch.cat([self.missing_mask[:, index].reshape(1, 32, 32)] * 3, dim=0)
        return (feature, true_value, mask)

    def __len__(self):
        return self.num_features

class MissingValueDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, missing_rate=0.2):
        self.original_dataset = original_dataset
        self.missing_rate = missing_rate

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        
        # Generate missing mask with the same shape as the original image
        mask = torch.FloatTensor(img.shape).uniform_() > self.missing_rate
        
        # Create missing feature matrix
        missing_features = img.clone()
        missing_features[~mask] = float(0)  # Set missing positions to 0.0
        mask = mask.float()
        
        return missing_features, img, mask

def load_data(datasetname, args):
    if datasetname == "synthetic":
        dataset = ColumnwiseDataset(args.num_nodes, args.num_features, args.rank_k, args.missing_rate)
        dataloader = torch.utils.data.DataLoader(dataset, args.max_device_batch_size, shuffle=True, num_workers=4,drop_last=True)
    elif datasetname == "cifar10":
        raw_dataset = torchvision.datasets.CIFAR10('dataset', train=True, download=True, transform=
                                                     Compose([ToTensor(), 
                                                              RandomResizedCrop(32, scale=(args.scale_lo, args.scale_high), ratio=(args.ratio_lo, args.ratio_high), antialias=True), 
                                                              transforms.RandomHorizontalFlip(p=args.flip),
                                                              Normalize(0.5, 0.5)]))
        dataset = MissingValueDataset(raw_dataset, args.missing_rate)
        dataloader = torch.utils.data.DataLoader(dataset, args.max_device_batch_size, shuffle=True, num_workers=4,drop_last=True)
    else:
        raise ValueError("未找到数据集")
    
    return dataloader