import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split

class SyncDataset(Dataset):
    def __init__(self, num_samples, num_nodes, rank_k, missing_rate=0.1):
        self.datasetname = "synthetic"
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.rank_k = rank_k
        self.missing_rate = missing_rate
        self.dataset = self._load_or_create_dataset()

    def _load_sync_data(self):
        U = np.random.randn(self.num_nodes, self.rank_k)
        V = np.random.randn(self.rank_k, self.num_nodes)
        true_matrix = np.dot(U, V)
        missing_mask = np.random.rand(self.num_nodes, self.num_nodes) < self.missing_rate
        feature_matrix = true_matrix.copy()
        feature_matrix[missing_mask] = 0
        return true_matrix, feature_matrix, missing_mask

    def _create_dataset(self):
        return [self._load_sync_data() for _ in range(self.num_samples)]

    def _get_dataset_path(self):
        return os.path.join("dataset", self.datasetname, 
                            f"{self.datasetname}_{self.num_samples}_{self.num_nodes}_{self.rank_k}_{self.missing_rate}.npz")

    def _load_or_create_dataset(self):
        dataset_path = self._get_dataset_path()
        
        if os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_path}")
            return self._load_dataset(dataset_path)
        else:
            print(f"No dataset found at {dataset_path}, creating a new one")
            dataset = self._create_dataset()
            self._save_dataset(dataset, dataset_path)
            return dataset

    def _load_dataset(self, filename):
        loaded_data = np.load(filename, allow_pickle=True)
        return [
            (torch.tensor(item[0], dtype=torch.float32),
             torch.tensor(item[1], dtype=torch.float32),
             torch.tensor(item[2], dtype=torch.bool))
            for item in loaded_data['data']
        ]

    def _save_dataset(self, dataset, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez_compressed(filename, data=dataset)
        print(f"New dataset saved at {filename}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        true_matrix, feature_matrix, missing_mask = self.dataset[idx]
        return feature_matrix, true_matrix, missing_mask

def load_data(datasetname, args):
    if datasetname == "synthetic":
        dataset = SyncDataset(args.num_samples, args.num_nodes, args.rank_k, args.missing_rate)
    else:
        raise ValueError("Dataset not found")
    
    train_size = int(args.train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    return train_dataset, test_dataset