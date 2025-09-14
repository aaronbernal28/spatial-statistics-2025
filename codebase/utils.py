import torch
from torch.utils.data import Dataset
from typing import List
import pandas as pd
import types

def load_samples(filename: str):
    import sys

    # Create codebase package
    codebase = types.ModuleType('codebase')
    codebase.__path__ = [] 
    sys.modules['codebase'] = codebase

    # Create codebase.utils submodule
    codebase_utils = types.ModuleType('codebase.utils') 
    codebase_utils.DatasetSismos = DatasetSismos
    sys.modules['codebase.utils'] = codebase_utils

    # Link them
    codebase.utils = codebase_utils
    return torch.load(filename, weights_only=False)

def save_samples(filename: str, list_samples: List[pd.DataFrame]):
    processed_samples = []
    
    for df in list_samples:
        if len(df) < 2:
            continue # skip
            
        # Sort by time (salvo el primero)
        reference_row = df.iloc[0:1]
        data_rows = df.iloc[1:].sort_values('t')
        features = pd.concat([reference_row, data_rows], ignore_index=True).values

        # Convierte a tensor
        sample_tensor = torch.tensor(features)
        processed_samples.append(sample_tensor)

    torch.save(DatasetSismos(processed_samples), filename)

class DatasetSismos(Dataset):
    """
    Dataset class for sequential data from circular subsets.
    Each sample is a sequence ordered by time, with the last element as target.
    """
    def __init__(self, samples: List[torch.Tensor]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input = sample[:-1, :]  # (seq_len-1, 5)
        target = sample[-1, :] # (1, 5)
        return {'input': input, 'target': target}