'''
testting file
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pickle
import os
from collate import collate_fn
from torch_custom_dataset import GazeDataSet

# Define the Seq2Seq model in PyTorch
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.LSTM(output_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, target_seq):
        _, (hidden_state, cell_state) = self.encoder(input_seq)
        output, _ = self.decoder(target_seq, (hidden_state, cell_state))
        output = self.fc(output)
        return output

# Hyperparametersx
hidden_size = 128
learning_rate = 0.001
batch_size = 32
epochs = 30  # Adjust as needed

# Instantiate the model, loss function, and optimizer
model = Seq2Seq(input_size=20, hidden_size=hidden_size, output_size=6)  # Assuming 20 features as input
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Dataset and DataLoader
data_directory = os.path.expanduser("~/360-FoV-prediction/data/processed")
custom_dataset = GazeDataSet(data_directory)

# Split dataset into train and test
train_size = int(0.8 * len(custom_dataset))
test_size = len(custom_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size,collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,collate_fn=collate_fn, shuffle=False)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        features, gaze_direction = batch
        optimizer.zero_grad()

        # Pad sequences for variable lengths
        features = pad_sequence(features, batch_first=True)
        gaze_direction = pad_sequence(gaze_direction, batch_first=True)

        # Forward pass
        output = model(features, gaze_direction)

        # Compute the loss
        loss = criterion(output, gaze_direction)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), 'gaze_model.pth')