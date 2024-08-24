import dgl
from dgl.data import CoraGraphDataset
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import vectorize_matrix

def load_graph_data(dataset_name, missing_rate=0.1, random_state=42):
    if dataset_name == "cora":
        dataset = CoraGraphDataset()
    else:
        raise ValueError("Dataset not supported")
    
    graph = dataset[0]
    np.random.seed(random_state)
    feature_matrix = graph.ndata['feat']
    num_nodes, num_features = feature_matrix.shape
    missing_mask = np.random.rand(num_nodes, num_features) < missing_rate
    
    return feature_matrix, missing_mask

def load_sync_data(num_nodes, rank_k, missing_rate=0.1):
    U = np.random.randn(num_nodes, rank_k)
    V = np.random.randn(rank_k, num_nodes)
    feature_matrix = np.dot(U, V)
    missing_mask = np.random.rand(num_nodes, num_nodes) < missing_rate
    return feature_matrix, missing_mask


def get_matrix_mask(feature_matrix, missing_mask):
    input_matrix = np.where(missing_mask, np.nan, feature_matrix)
    return input_matrix

def create_dataloader(input_matrix, ground_truth_matrix, missing_mask, batch_size=512):
    vectorized_input = vectorize_matrix(input_matrix)
    vectorized_ground_truth = vectorize_matrix(ground_truth_matrix)
    vectorized_mask = vectorize_matrix(missing_mask)
    dataset = TensorDataset(vectorized_input, vectorized_ground_truth, vectorized_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader