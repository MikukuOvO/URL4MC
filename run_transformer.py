import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import build_args, load_best_configs, discretize_matrix, vectorize_matrix
from data_loading import get_matrix_mask, load_sync_data, create_dataloader
from models.BERT import BertForMatrixCompletion, CustomTokenizer

def train(model, data_loader, criterion, optimizer, epsilon, tokenizer, epochs, batch_size):
    model.train()
    progress_bar = tqdm(range(epochs))
    for epoch in progress_bar:
        epoch_loss = 0
        for batch in data_loader:
            input_matrix, ground_truth_matrix, missing_mask = batch
            input_matrix = discretize_matrix(input_matrix, epsilon)
            tokenized_input = tokenizer.tokenize(input_matrix)
            input_ids = torch.tensor(tokenized_input).unsqueeze(0).to(input_matrix.device)

            optimizer.zero_grad()
            outputs = model(input_ids).squeeze(0)
            known_indices = ~missing_mask
            predicted_vector = outputs[known_indices]

            loss = criterion(predicted_vector, ground_truth_matrix[known_indices])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        if epoch % 100 == 0:
            predicted_matrix = inference(model, input_matrix, tokenizer, epsilon, batch_size)
            mse_loss = criterion(predicted_matrix, ground_truth_matrix)
            print(f"Epoch: {epoch}, Loss: {mse_loss.item()}")

        avg_loss = epoch_loss / len(data_loader)
        progress_bar.set_postfix(loss=avg_loss)

def inference(model, input_matrix, tokenizer, epsilon, batch_size):
    model.eval()
    with torch.no_grad():
        input_matrix = discretize_matrix(input_matrix, epsilon)
        vectorized_input = vectorize_matrix(input_matrix)
        tokenized_input = tokenizer.tokenize(vectorized_input)

        outputs_list = []
        for i in range(0, len(tokenized_input), batch_size):
            input_ids = torch.tensor(tokenized_input[i : i + batch_size]).unsqueeze(0).to(input_matrix.device)
            outputs = model(input_ids)
            outputs_list.append(outputs.squeeze(0))

        predicted_vector = torch.cat(outputs_list)
        predicted_matrix = predicted_vector.view(input_matrix.shape)
        
        return predicted_matrix

def main(args):
    device = args.device if args.device >= 0 else "cpu"
    print(device)
    feature_matrix, missing_mask = load_sync_data(args.num_nodes, args.rank_k, args.missing_rate)
    input_matrix = get_matrix_mask(feature_matrix, missing_mask)
    feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32).to(device)
    input_matrix = torch.tensor(input_matrix, dtype=torch.float32).to(device)
    missing_mask = torch.tensor(missing_mask, dtype=torch.bool).to(device)

    dataloader = create_dataloader(input_matrix, feature_matrix, missing_mask, args.batch_size)

    model = BertForMatrixCompletion()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    tokenizer = CustomTokenizer(args.epsilon, args.token_id_start, args.mask_token_id, args.matrix_range)
    model.to(device)
    train(model, dataloader, criterion, optimizer, args.epsilon, tokenizer, args.num_epochs, args.batch_size)

    predicted_matrix = inference(model, input_matrix, tokenizer, args.epsilon, args.batch_size)
    mse_loss = criterion(predicted_matrix, feature_matrix)
    print(f"Inference MSE Loss: {mse_loss.item()}")


if __name__ == "__main__":
    args = build_args()
    args = load_best_configs(args, "configs/transformer_configs.yaml")
    print(args)
    main(args)