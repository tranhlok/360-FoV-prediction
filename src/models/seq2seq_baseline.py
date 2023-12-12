'''
The baseline seq2seq model, make adjustment to the 
custom dataset in torch_cutom_dataset.py
collate_fn can be found in collate.py
'''

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from collate import collate_fn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR, EarlyStopping, ReduceLROnPlateau
from torch_custom_dataset import GazeDataSet_OnlyHead
import matplotlib.pyplot as plt

# # Define the PyTorch Seq2Seq model
# class Seq2Seq_Baseline1(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Seq2Seq_Baseline, self).__init__()
#         self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.decoder_lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
#         self.decoder_dense = nn.Linear(hidden_size, output_size)

#     def forward(self, encoder_input, decoder_input):
#         _, (encoder_hidden, encoder_cell) = self.encoder(encoder_input)

#         # Use the last hidden and cell states of the encoder as initial states for the decoder
#         decoder_outputs, _ = self.decoder_lstm(decoder_input, (encoder_hidden, encoder_cell))

#         # Apply the dense layer to get the final output
#         decoder_outputs = self.decoder_dense(decoder_outputs)

#         return decoder_outputs


class Seq2Seq_Baseline(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq_Baseline, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size)

        # Decoder
        self.decoder = nn.LSTM(output_size, hidden_size)

        # Output layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_input, decoder_input):
        # Encoder forward pass
        _, (encoder_hidden, encoder_cell) = self.encoder(encoder_input)

        # Decoder forward pass
        decoder_output, _ = self.decoder(decoder_input, (encoder_hidden, encoder_cell))

        # Linear layer to get the output
        output = self.linear(decoder_output)

        return output

# Split the dataset into training and validation sets
data_directory = os.path.expanduser("~/360-FoV-prediction/data/processed")
custom_dataset = GazeDataSet_OnlyHead(data_directory)
print("Should be 95", len(custom_dataset))

train_dataset, val_dataset = train_test_split(custom_dataset, test_size=0.2, random_state=42)
val_dataset, test_dataset = train_test_split(val_dataset, test_size=0.5, random_state=42)
# Data loaders
# Instantiate the model
input_size = 3  # Update based on your input features
hidden_size = 64
output_size = 3 # Update based on your output features
seq2seq_model = Seq2Seq_Baseline(input_size, hidden_size, output_size)
learning_rate = 0.001
batch_size = 32


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,  collate_fn=collate_fn, shuffle=False)


# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(seq2seq_model.parameters(), lr=learning_rate)
# check back if scheduler is needed
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-6, verbose=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=5)

# Early stopping parameters


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
seq2seq_model.to(device)

# Training loop
num_epochs = 200
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
    # scheduler.step()
    scheduler.step(average_val_loss)

    # Early stopping check
    early_stopping.step(average_val_loss)

    if early_stopping.early_stop:
        print("Early stopping triggered at epoch", epoch)
        break

# Save the model if needed
torch.save(seq2seq_model.state_dict(), 'seq2seq_baseline.pth')

epochs_range = range(1, num_epochs + 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()