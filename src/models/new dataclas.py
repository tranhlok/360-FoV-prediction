import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils


def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data), scaler


class GazeDirectionDataset(Dataset):
    def __init__(self, csv_file, sequence_length=10):
        self.csv_file = csv_file
        self.sequence_length = sequence_length

        data = pd.read_csv(self.csv_file)
        gaze_data = data[['REyeRX', 'REyeRY', 'REyeRZ']].values

        # Normalize the data
        self.normalized_gaze_data, self.scaler = normalize_data(gaze_data)

    def __len__(self):
        return len(self.normalized_gaze_data) - self.sequence_length - 1

    def __getitem__(self, idx):
        input_seq = self.normalized_gaze_data[idx:idx + self.sequence_length]
        target_seq = self.normalized_gaze_data[idx + 1:idx + self.sequence_length + 1]

        input_seq = torch.tensor(input_seq, dtype=torch.float32)
        target_seq = torch.tensor(target_seq, dtype=torch.float32)

        return input_seq, target_seq