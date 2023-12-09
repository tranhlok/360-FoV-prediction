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
        print("Column Names:", data.columns)
        # Replace NaN values with the mean of each column
        # data = data.fillna(data.mean())

        # Extract features and labels (assuming 'GazeDirection' is the column to predict)
        features = data.iloc[:, 1:21].values  # Exclude 'Timer' and gaze direction columns
        print("Column Names:", features)
        gaze_direction = data[['LEyeRX', 'LEyeRY', 'LEyeRZ', 'REyeRX', 'REyeRY', 'REyeRZ']].values
        print(features)
        print(gaze_direction)
        # Convert to PyTorch tensors
        features = torch.tensor(features, dtype=torch.float32)
        gaze_direction = torch.tensor(gaze_direction, dtype=torch.float32)

        return features, gaze_direction

# Example usage:
data_directory = os.path.expanduser("~/360-FoV-prediction/data/processed")
custom_dataset = GazeDataSet(data_directory)

# Accessing a specific sample
sample_idx = 0
sample_data, sample_gaze_direction = custom_dataset[sample_idx]

# Print the features and gaze direction for the sample after handling missing values
print("Features:")
print(sample_data.shape)

print("\nGaze Direction:")
print(sample_gaze_direction.shape)
