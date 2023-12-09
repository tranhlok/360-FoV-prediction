# import os
# import torch
# import pandas as pd
# from torch.utils.data import Dataset

# class HeadGazeDataset(Dataset):
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         file_name = self.file_list[idx]
#         file_path = os.path.join(self.data_dir, file_name)

#         # Load data from the text file
#         data = pd.read_csv(file_path, sep='\t', index_col=0)

#         # You may need to preprocess your data, normalize, or perform other transformations here

#         # Convert data to PyTorch tensors
#         head_data = torch.tensor(data[['HeadX', 'HeadY', 'HeadZ', 'HeadRX', 'HeadRY', 'HeadRZ']].values, dtype=torch.float32)
#         gaze_data = torch.tensor(data[['LEyeRX', 'LEyeRY', 'REyeRX', 'REyeRY']].values, dtype=torch.float32)

#         # You can modify this based on your specific needs
#         sample = {'head': head_data, 'gaze': gaze_data}

#         return sample

# # Example usage:
# data_dir = '/Users/lok/code/college/360-FoV-prediction/data/user_movement'
# head_gaze_dataset = HeadGazeDataset(data_dir)

# # Accessing a sample
# sample = head_gaze_dataset[0]
# print("Head data:", sample['head'])
# print("Gaze data:", sample['gaze'])

import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class HeadGazeDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

        # Initialize empty lists to store data
        self.head_data_list = []
        self.gaze_data_list = []

        # Load data from all text files
        for file_name in self.file_list:
            file_path = os.path.join(self.data_dir, file_name)
            data = pd.read_csv(file_path, sep='\t', index_col=0)

            # Append data to lists
            self.head_data_list.append(torch.tensor(data[['HeadX', 'HeadY', 'HeadZ', 'HeadRX', 'HeadRY', 'HeadRZ']].values, dtype=torch.float32))
            self.gaze_data_list.append(torch.tensor(data[['LEyeRX', 'LEyeRY', 'REyeRX', 'REyeRY']].values, dtype=torch.float32))

    def __len__(self):
        # Return the total number of samples across all files
        return sum(len(data) for data in self.head_data_list)

    def __getitem__(self, idx):
        # Find the corresponding file and index within that file
        file_idx, sample_idx = self.find_file_and_index(idx)

        # Retrieve data from the selected file
        head_data = self.head_data_list[file_idx][sample_idx]
        gaze_data = self.gaze_data_list[file_idx][sample_idx]

        # You can modify this based on your specific needs
        sample = {'head': head_data, 'gaze': gaze_data}

        return sample

    def find_file_and_index(self, idx):
        # Find the file containing the sample at index idx
        for file_idx, data in enumerate(self.head_data_list):
            if idx < len(data):
                return file_idx, idx
            idx -= len(data)

# Example usage:
data_dir = '/Users/lok/code/college/360-FoV-prediction/data/user_movement'
head_gaze_dataset = HeadGazeDataset(data_dir)

# Accessing a sample
sample = head_gaze_dataset[0]
print("Head data:", sample['head'])
print("Gaze data:", sample['gaze'])
