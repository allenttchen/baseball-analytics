import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


class PreprocessedDataset(Dataset):
    def __init__(self, dataset_filepath: str, target_name: str):
        self.dataset = pd.read_csv(dataset_filepath)
        self.target_name = target_name
        self.feature_names = [name for name in self.dataset.columns.tolist() if name != self.target_name]
        self.features = self.dataset[self.feature_names]
        self.target = np.array(self.dataset[self.target_name])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features.iloc[idx, :].values, dtype=torch.float32),
            torch.tensor(self.target[idx], dtype=torch.int)
        )
