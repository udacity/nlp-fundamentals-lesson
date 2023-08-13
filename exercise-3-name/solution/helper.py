from __future__ import annotations
from collections.abc import Sequence

import time
import math
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset,
)
from collections import Counter


class TokenMapping():
    def __init__(
        self,
        text_as_list: list[str],
        not_found_token: str = 'TOKEN_NOT_FOUND',
        not_found_id: int | None = None,
    ):
        self.counter = Counter(text_as_list)
        # Includes `not found token`
        self.n_tokens: int = len(self.counter) + 1
        # Token to ID mapping
        self._token2id: dict[str, int] = {
            token: idx
            for idx, (token, _) in enumerate(self.counter.items())
        }
        # Reverse mapping: ID to token mapping
        self._id2token: dict[int, str] = {
            idx: token
            for token, idx in self._token2id.items()
        }
        # Token representing not found 
        self._not_found_token = not_found_token
        # Not found ID defaults to next available number
        if not_found_id is None:
            self._not_found_id = max(self._token2id.values()) + 1
        else:
            self._not_found_id = not_found_id
    
    def encode(self, text_list: list[str]) -> list[int]:
        '''Encodes list of tokens (strings) into list of IDs (integers)'''
        encoded = [
            self.token2id(token)
            for token in text_list
        ]
        # Include the not found ID if it wasn't included yet
        if self._not_found_id not in encoded:
            encoded += [self._not_found_id]
        return encoded

    def token2id(self, token: str):
        '''Returns ID for given token (even if token not found)'''
        return self._token2id.get(token, self._not_found_id)
    
    def id2token(self, idx: int):
        '''Returns token for given ID or the not found token'''
        return self._id2token.get(idx, self._not_found_token)


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


def tokens_to_id_tensor(
    tokens: list[str],
    token_id_mapping: dict[str, int],
) -> torch.Tensor:
    '''Create PyTorch Tensor from token to ID based on given mapping'''
    id_tensor = (
        torch.tensor(
            [token_id_mapping(token) for token in tokens],
            dtype=torch.long,
        )
        .unsqueeze(0)
    )
    return id_tensor


def tokenize_text_from_tokenizer(
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