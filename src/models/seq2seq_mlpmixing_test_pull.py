#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 23:10:57 2023

@author: taishanzhao
"""

import torch
import torch.nn as nn
import os
import sys
from torch_custom_dataset import GazeDataSet_Movement
# from data.torch_custom_dataset import GazeDataSet # for running in the main dir
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from collate import collate_fn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


class Seq2Seq_nlpmixing(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq_nlpmixing, self).__init__()
        self.encoder1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.encoder2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder_lstm1 = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder_dense = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_input, decoder_input, encoder_lengths=None):
        encoder1_outputs, _ = self.encoder1(encoder_input)
        encoder2_outputs, _ = self.encoder2(encoder1_outputs)

        decoder1_outputs, _ = self.decoder_lstm1(decoder_input)
        decoder2_outputs, _ = self.decoder_lstm2(decoder1_outputs)

        output = self.decoder_dense(decoder2_outputs)

        return output


# Split the dataset into training and validation sets
data_directory = os.path.expanduser("~/Desktop/Image_Processing/360-FoV-prediction/data/processed")
custom_dataset = GazeDataSet_Movement(data_directory, 'Pulling')
print("Should be 95", len(custom_dataset))
print(custom_dataset.file_list)

train_dataset, val_dataset = train_test_split(custom_dataset, test_size=0.2, random_state=42)
val_dataset, test_dataset = train_test_split(val_dataset, test_size=0.5, random_state=42)

input_size = 3  # Update based on your input features
hidden_size = 128
output_size = 3 # Update based on your output features
seq2seq_model = Seq2Seq_nlpmixing(input_size, hidden_size, output_size)
learning_rate = 0.001
batch_size = 32


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,  collate_fn=collate_fn, shuffle=False)


# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(seq2seq_model.parameters(), lr=learning_rate)
# check back if scheduler is needed
#scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True)
# 
device = torch.device('cuda' if torch.cuda.device_count() > 0 else 'cpu')
seq2seq_model.to(device)

# Training loop
num_epochs = 100
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    total_train_loss = 0.0
    seq2seq_model.train()

    for i, batch in enumerate(train_dataloader):
        features, gaze_direction = batch
        features, gaze_direction = features.to(device), gaze_direction.to(device)
        # Assuming decoder_input is the same as target for simplicity (modify as needed)
        optimizer.zero_grad()
        output = seq2seq_model(features, gaze_direction)
        # output = seq2seq_model(gaze_direction)

        loss = criterion(output, gaze_direction)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        # print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_dataloader)}], Training Loss: {loss.item()}')
    # Validation
    seq2seq_model.eval()

    with torch.no_grad():
        total_val_loss = 0.0
        for batch in val_dataloader:
            features, gaze_direction = batch
            features, gaze_direction = features.to(device), gaze_direction.to(device)
            decoder_input = gaze_direction
            val_output = seq2seq_model(features, decoder_input)
            total_val_loss += criterion(val_output, gaze_direction).item()

    average_train_loss = total_train_loss / len(train_dataloader)
    average_val_loss = total_val_loss / len(val_dataloader)
    train_losses.append(average_train_loss)
    val_losses.append(average_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {average_train_loss}, Average Validation Loss: {average_val_loss}')
    scheduler.step(average_val_loss)

# Save the model if needed
torch.save(seq2seq_model.state_dict(), 'seq2seq_mlpmixing_test.pth')
epochs_range = range(1, num_epochs + 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('seq2seq + MLP mixing: Pulling Trolley')
plt.legend()
plt.show()