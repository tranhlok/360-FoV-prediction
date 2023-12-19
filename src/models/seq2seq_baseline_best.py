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

class HeadToGazeDataset(Dataset):
    def __init__(self, csv_file, sequence_length=5):
        self.csv_file = csv_file
        self.sequence_length = sequence_length

        data = pd.read_csv(self.csv_file)
        features = data[['HeadX', 'HeadY', 'HeadZ', 'REyeRX', 'REyeRY', 'REyeRZ']].values
        labels = data[['REyeRX', 'REyeRY', 'REyeRZ']].values

        self.features_scaler = MinMaxScaler(feature_range=(0, 1))
        self.labels_scaler = MinMaxScaler(feature_range=(0, 1))
        self.normalized_features = self.features_scaler.fit_transform(features)
        self.normalized_labels = self.labels_scaler.fit_transform(labels)

    def __len__(self):
        return len(self.normalized_features) - self.sequence_length

    def __getitem__(self, idx):
        input_seq = self.normalized_features[idx:idx + self.sequence_length]
        target_seq = self.normalized_labels[idx + 1:idx + self.sequence_length + 1]

        input_seq = torch.tensor(input_seq, dtype=torch.float32)  # Adding batch dimension
        target_seq = torch.tensor(target_seq, dtype=torch.float32)

        return input_seq, target_seq
    

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
data_directory = os.path.expanduser("~/360-FoV-prediction/data/processed/yuchen_presenting.csv")
# custom_dataset = GazeDirectionDataset(data_directory)
custom_dataset = HeadToGazeDataset(data_directory)
print("===========================dataset length===========================", len(custom_dataset))

train_dataset, val_dataset = train_test_split(custom_dataset, test_size=0.3, random_state=42, shuffle=True)
val_dataset, test_dataset = train_test_split(val_dataset, test_size=0.5, random_state=42)
# Data loaders
# Instantiate the model
input_size = 6  # Update based on your input features
hidden_size = 64
output_size = 3 # Update based on your output features
seq2seq_model = Seq2Seq_Baseline(input_size, hidden_size, output_size) 
learning_rate = 0.0001
batch_size = 32


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(seq2seq_model.parameters(), lr=learning_rate)
# check back if scheduler is needed
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-6, verbose=True)

# Early stopping parameters
early_stopping_patience = 5
early_stopping_counter = 0
best_val_loss = float('inf')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# cuda
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seq2seq_model.to(device)

# Training loop
num_epochs = 30
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
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
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

