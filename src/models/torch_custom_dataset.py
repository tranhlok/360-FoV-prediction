'''
Modify the dataset here if needed.
'''

import torch
from torch.utils.data import Dataset
import os
import pandas as pd

class GazeDataSet(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [file for file in os.listdir(data_dir) if file.endswith('.csv')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)

        # Load the data from the text file with the correct delimiter
        data = pd.read_csv(file_path, delimiter=',')  # Assuming comma-separated values
        # print("Column Names:", len(data.columns))
        # Replace NaN values with the mean of each column
        # data = data.fillna(data.mean())

        # Extract features and labels (assuming 'GazeDirection' is the column to predict)
        features = data.iloc[:, 1:21].values  # Exclude 'Timer' and gaze direction columns
        # print("Column Names:", features)
        gaze_direction = data[['LEyeRX', 'LEyeRY', 'LEyeRZ', 'REyeRX', 'REyeRY', 'REyeRZ']].values
        # print(features)
        # print(gaze_direction)
        # Convert to PyTorch tensors
        features = torch.tensor(features, dtype=torch.float32)
        gaze_direction = torch.tensor(gaze_direction, dtype=torch.float32)

        return features, gaze_direction


class GazeDataSet_OnlyHead(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [file for file in os.listdir(data_dir) if file.endswith('.csv')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)

        # Load the data from the text file with the correct delimiter
        data = pd.read_csv(file_path, delimiter=',')  # Assuming comma-separated values

        # Extract features and labels (assuming 'GazeDirection' is the column to predict)
        features = data[['HeadRX', 'HeadRY', 'HeadRZ']].values  # Exclude 'Timer' and gaze direction columns
        gaze_direction = data[['REyeRX', 'REyeRY', 'REyeRZ']].values

        # Convert to PyTorch tensors
        features = torch.tensor(features, dtype=torch.float32)
        gaze_direction = torch.tensor(gaze_direction, dtype=torch.float32)

        return features, gaze_direction