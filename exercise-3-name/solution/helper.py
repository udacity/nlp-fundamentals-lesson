from __future__ import annotations
from collections.abc import Sequence

import time
import math
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset,
    # DataLoader,
)

class ShakespeareDataset(Dataset):
    def __init__(self, encoded_text: Sequence, sequence_length: int):
        self.encoded_text = encoded_text
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.encoded_text) - self.sequence_length

    def __getitem__(self, index):
        x = torch.tensor(
            self.encoded_text[index: (index+self.sequence_length)],
            dtype=torch.long,
        )
        # Target is shifted by one character/token
        y = torch.tensor(
            self.encoded_text[(index+1): (index+self.sequence_length+1)],
            dtype=torch.long,
        )
        return x, y

class ShakespeareModel(nn.Module):
    def __init__(self, n_tokens: int, embedding_dim: int, hidden_dim: int):
        super(ShakespeareModel, self).__init__()
        self.embedding = nn.Embedding(n_tokens, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_tokens)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


def start_time() -> float:
    '''Returns a start time from right now'''
    return time.time()

def time_since(start: float) -> str:
    '''Return string of time since given time (float) in "{m:02}m {s:02.1f}s"'''
    s = time.time() - start
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m:02}m {s:02.1f}s'

def build_model(
    n_tokens: int,
    embedding_dim: int = 16,
    hidden_dim: int = 32,
):
    '''Create basic RNN-based model (embeddings -> RNN -> Linear/Dense layer)'''
    return ShakespeareModel(n_tokens, embedding_dim, hidden_dim)

def tokens_to_index_tensor(
    tokens: list[str],
    token_index_mapping: dict[str, int],
) -> torch.Tensor:
    '''Create PyTorch Tensor from token to index based on given mapping'''
    index_tensor = (
        torch.tensor(
            [token_index_mapping[token] for token in tokens],
            dtype=torch.long,
        )
        .unsqueeze(0)
    )
    return index_tensor

def tokenize_text(
    tokenizer,
    text: str,
) -> list[str]:
    '''Return list of token (strings) of text using given tokenizer'''
    max_seq_length = tokenizer.model_max_length
    # Chunk the string so tokenizer can take in full input
    chunks_generator = (
        text[i:i+max_seq_length]
        for i in range(0, len(text), max_seq_length)
    )
    # Special tokens to ignore
    ignore_tokens = (
        tokenizer.cls_token,
        tokenizer.sep_token,
    )
    # Get list of tokens (one chunk at a time)
    tokenized_text = [
        token
        for chunk in chunks_generator
        for token in tokenizer(chunk).tokens()
        if (
            token not in ignore_tokens
        )
    ]

    return tokenized_text