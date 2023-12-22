'''
Modify the datasets here if needed.
All the dataset are here

'''

import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from sklearn import preprocessing
import numpy as np



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
        features = data[['HeadRX', 'HeadRY', 'HeadRZ']].values[::100]   # Exclude 'Timer' and gaze direction columns
        # features = data[['HeadX', 'HeadY', 'HeadZ','HeadRX', 'HeadRY', 'HeadRZ']].values  # Exclude 'Timer' and gaze direction columns

        gaze_direction = data[['REyeRX', 'REyeRY', 'REyeRZ']].values[::100] 

        # Convert to PyTorch tensors
        features = torch.tensor(features, dtype=torch.float32)
        gaze_direction = torch.tensor(gaze_direction, dtype=torch.float32)

        return features, gaze_direction
    

class GazeDataSet_Experiment(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
        self.min_max_scaler_features = preprocessing.MinMaxScaler()
        self.min_max_scaler_gaze = preprocessing.MinMaxScaler()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)

        # Load the data from the text file with the correct delimiter
        data = pd.read_csv(file_path, delimiter=',')  # Assuming comma-separated values
        data = data.interpolate(method='linear', limit_direction='both')

        # Extract features and labels (assuming 'GazeDirection' is the column to predict)
        # features = data[['HeadRX', 'HeadRY', 'HeadRZ']].values  # Exclude 'Timer' and gaze direction columns
        features = data[['HeadX', 'HeadY', 'HeadZ']].values[::100] # Exclude 'Timer' and gaze direction columns

        gaze_direction = data[['REyeRX', 'REyeRY', 'REyeRZ']].values[::100] 

        # Convert to PyTorch tensors
        normalized_features = self.min_max_scaler_features.fit_transform(features)
        normalized_gaze_direction = self.min_max_scaler_gaze.fit_transform(gaze_direction)

        # Convert to PyTorch tensors
        features = torch.tensor(normalized_features, dtype=torch.float32)
        gaze_direction = torch.tensor(normalized_gaze_direction, dtype=torch.float32)


        return features, gaze_direction
    
class OneViewDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = pd.read_csv(self.data_dir, delimiter=',')  # Assuming comma-separated values

        self.min_max_scaler_features = preprocessing.MinMaxScaler()
        self.min_max_scaler_gaze = preprocessing.MinMaxScaler()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the data from the text file with the correct delimiter
        self.data = self.data.interpolate(method='linear', limit_direction='both')

        # Extract features and labels (assuming 'GazeDirection' is the column to predict)
        # features = data[['HeadRX', 'HeadRY', 'HeadRZ']].values  # Exclude 'Timer' and gaze direction columns
        features = self.data[['HeadX', 'HeadY', 'HeadZ']].values # Exclude 'Timer' and gaze direction columns

        gaze_direction = self.data[['REyeRX', 'REyeRY', 'REyeRZ']].values

        # Convert to PyTorch tensors
        normalized_features = self.min_max_scaler_features.fit_transform(features)
        normalized_gaze_direction = self.min_max_scaler_gaze.fit_transform(gaze_direction)

        # Convert to PyTorch tensors
        features = torch.tensor(normalized_features, dtype=torch.float32)
        gaze_direction = torch.tensor(normalized_gaze_direction, dtype=torch.float32)


        return features[idx], gaze_direction[idx]

   
    
class GazeDataSet_Movement(Dataset):
    def __init__(self, data_dir, mov):
        self.data_dir = data_dir
        self.file_list = [file for file in os.listdir(data_dir) if (file.endswith('.csv') and (mov in file))]
        self.min_max_scaler_features = preprocessing.MinMaxScaler()
        self.min_max_scaler_gaze = preprocessing.MinMaxScaler()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)

        # Load the data from the text file with the correct delimiter
        data = pd.read_csv(file_path, delimiter=',')  # Assuming comma-separated values
        data = data.interpolate(method='linear', limit_direction='both')

        # Extract features and labels (assuming 'GazeDirection' is the column to predict)
        # features = data[['HeadRX', 'HeadRY', 'HeadRZ']].values  # Exclude 'Timer' and gaze direction columns
        features = data[['HeadX', 'HeadY', 'HeadZ']].values[::100] # Exclude 'Timer' and gaze direction columns

        gaze_direction = data[['REyeRX', 'REyeRY', 'REyeRZ']].values[::100] 

        # Convert to PyTorch tensors
        normalized_features = self.min_max_scaler_features.fit_transform(features)
        normalized_gaze_direction = self.min_max_scaler_gaze.fit_transform(gaze_direction)

        # Convert to PyTorch tensors
        features = torch.tensor(normalized_features, dtype=torch.float32)
        gaze_direction = torch.tensor(normalized_gaze_direction, dtype=torch.float32)


        return features, gaze_direction    