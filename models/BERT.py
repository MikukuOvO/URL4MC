import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig

class BertForMatrixCompletion(nn.Module):
    def __init__(self):
        super(BertForMatrixCompletion, self).__init__()
        config = BertConfig(
            hidden_dropout_prob=0.0,
            num_hidden_layers=4,
            num_attention_heads=8
        )
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        sequence_output = outputs.last_hidden_state
        predictions = self.linear(sequence_output).squeeze(-1)
        return predictions
    
class CustomTokenizer:
    def __init__(self, epsilon, token_id_start, mask_token_id, matrix_range):
        self.epsilon = epsilon
        self.token_id_start = token_id_start
        self.mask_token_id = mask_token_id
        self.matrix_range = matrix_range
        self.token_to_id = {}
        self.id_to_token = {}
        self._create_token_mapping()

    def _create_token_mapping(self):
        value = self.matrix_range[0]
        token_id = self.token_id_start
        while value <= self.matrix_range[1]:
            token_str = f"{value:.2f}"
            self.token_to_id[token_str] = token_id
            self.id_to_token[token_id] = token_str
            value += self.epsilon
            token_id += 1
        self.token_to_id["MASK"] = self.mask_token_id

    def tokenize(self, vector):
        tokens = []
        for val in vector:
            token_str = f"{val:.2f}"
            tokens.append(self.token_to_id.get(token_str, self.mask_token_id))
        return tokens