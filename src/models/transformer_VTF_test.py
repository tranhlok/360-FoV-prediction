#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:51:37 2023

@author: taishanzhao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
import pandas as pd
import numpy as np
from torch_custom_dataset import GazeDataSet_Past_Future
from torch.utils.data import Dataset
# from data.torch_custom_dataset import GazeDataSet # for running in the main dir
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from collate import collate_fn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data), scaler
# Assuming HeadToGazeDataset is already defined as you provided
class GazeDirectionDataset(Dataset):
    def __init__(self, folder_path, sequence_length):
        self.sequence_length = sequence_length
        self.data = []

        # Load and aggregate data from all CSV files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                data = pd.read_csv(file_path)
                gaze_data = data[['HeadX', 'HeadY', 'HeadZ','REyeRX', 'REyeRY', 'REyeRZ']].values
                self.data.append(gaze_data)

        # Concatenate and normalize the data
        all_data = np.concatenate(self.data, axis=0)
        self.normalized_gaze_data, self.scaler = normalize_data(all_data)

    def __len__(self):
        return len(self.normalized_gaze_data) - self.sequence_length - 1

    def __getitem__(self, idx):
        input_seq = self.normalized_gaze_data[idx:idx + self.sequence_length]
        target_seq = self.normalized_gaze_data[idx + self.sequence_length : idx + self.sequence_length + self.sequence_length]

        input_seq = torch.tensor(input_seq, dtype=torch.float32)
        target_seq = torch.tensor(target_seq, dtype=torch.float32)

        return input_seq, target_seq
    
# Define a simple Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, src, tgt):
        out = self.transformer(src, tgt)
        return self.linear(out)

# Parameters
input_dim = 6  # As we have 3 features (REyeRX, REyeRY, REyeRZ)
sequence_length = 30
num_layers = 2
num_heads = 6
dim_feedforward = 512
batch_size = 16
epochs = 30

# Load dataset
dataset = GazeDirectionDataset("/Users/taishanzhao/Desktop/Image_Processing/360-FoV-prediction/data/processed_by_activity/chatting", sequence_length=sequence_length)

train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
val_dataset, test_dataset = train_test_split(dataset, test_size=0.5, random_state=42)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,  collate_fn=collate_fn, shuffle=True)

# Initialize model
model = TransformerModel(input_dim, num_layers, num_heads, dim_feedforward)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_losses = []
val_losses = []
# Training loop
for epoch in range(epochs):
    total_train_loss = 0
    model.train()
    for i, batch in enumerate(train_dataloader):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs, targets)  # Using inputs as both src and tgt for simplicity
        loss = loss_function(output, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        
    with torch.no_grad():
        total_val_loss = 0.0
        for batch in val_dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs, targets)  # Using inputs as both src and tgt for simplicity
            loss = loss_function(output, targets)
            total_val_loss += loss.item()

    average_train_loss = total_train_loss / len(train_dataloader)
    average_val_loss = total_val_loss / len(val_dataloader)
    train_losses.append(average_train_loss)
    val_losses.append(average_val_loss)

    print(f'Epoch [{epoch+1}/{epochs}], Average Training Loss: {average_train_loss}, Average Validation Loss: {average_val_loss}')

# Save the model
torch.save(model.state_dict(), 'transformer_model.pth')
epochs_range = range(1, epochs + 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Transformer: Sweep')
plt.legend()
plt.show()