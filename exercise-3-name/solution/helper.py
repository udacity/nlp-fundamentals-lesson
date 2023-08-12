from __future__ import annotations
from collections.abc import Sequence

import time
import math
import torch
from torch.utils.data import (
    Dataset,
)

def start_time() -> float:
    '''Returns a start time from right now'''
    return time.time()

def time_since(start: float) -> str:
    '''Return string of time since given time (float) in "{m:02}m {s:02.1f}s"'''
    s = time.time() - start
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m:02}m {s:02.1f}s'

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
        # Target is shifted by one character
        y = torch.tensor(
            self.encoded_text[(index+1): (index+self.sequence_length+1)],
            dtype=torch.long,
        )
        return x, y