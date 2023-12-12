from collate import collate_fn

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the PyTorch Seq2Seq model
class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.decoder_dense = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_input, decoder_input):
        _, (encoder_hidden, encoder_cell) = self.encoder(encoder_input)

        # Use the last hidden and cell states of the encoder as initial states for the decoder
        decoder_outputs, _ = self.decoder_lstm(decoder_input, (encoder_hidden, encoder_cell))

        # Apply the dense layer to get the final output
        decoder_outputs = self.decoder_dense(decoder_outputs)

        return decoder_outputs

# Define the dataset class
class GazeDataSet(Dataset):
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
        features = data.iloc[:, 1:21].values  # Exclude 'Timer' and gaze direction columns
        gaze_direction = data[['LEyeRX', 'LEyeRY', 'LEyeRZ', 'REyeRX', 'REyeRY', 'REyeRZ']].values

        # Convert to PyTorch tensors
        features = torch.tensor(features, dtype=torch.float32)
        gaze_direction = torch.tensor(gaze_direction, dtype=torch.float32)

        return features, gaze_direction

# Split the dataset into training and validation sets
data_directory = os.path.expanduser("~/360-FoV-prediction/data/processed")
custom_dataset = GazeDataSet(data_directory)

train_dataset, val_dataset = train_test_split(custom_dataset, test_size=0.2, random_state=42)
val_dataset, test_dataset = train_test_split(val_dataset, test_size=0.5, random_state=42)
# Data loaders
train_dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16,  collate_fn=collate_fn, shuffle=False)

# Instantiate the model
input_size = 20  # Update based on your input features
hidden_size = 64
output_size = 6  # Update based on your output features
seq2seq_model = Seq2SeqModel(input_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(seq2seq_model.parameters())

# 
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
seq2seq_model.to(device)

# Training loop
num_epochs = 10

train_loss_list = []
val_loss_list = []
val_dice_list = []
test_dice_list = []
for epoch in range(num_epochs):
    total_train_loss = 0.0
    seq2seq_model.train()

    for i, batch in enumerate(train_dataloader):
        features, gaze_direction = batch
        features, gaze_direction = features.to(device), gaze_direction.to(device)
        # Assuming decoder_input is the same as target for simplicity (modify as needed)
        optimizer.zero_grad()
        output = seq2seq_model(features, gaze_direction)
        loss = criterion(output, gaze_direction)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_dataloader)}], Training Loss: {loss.item()}')
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

    print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {average_train_loss}, Average Validation Loss: {average_val_loss}')

# Save the model if needed
torch.save(seq2seq_model.state_dict(), 'seq2seq_model.pth')

