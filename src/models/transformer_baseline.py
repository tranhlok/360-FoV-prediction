#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:23:19 2023

@author: taishanzhao
"""

"""
Created on Sun Dec 17 23:14:46 2023

@author: taishanzhao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
import pandas as pd
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

class HeadToGazeDataset(Dataset):
    def __init__(self, csv_file, sequence_length=5):
        self.csv_file = csv_file
        self.sequence_length = sequence_length

        data = pd.read_csv(self.csv_file)
        features = data[['REyeRX', 'REyeRY', 'REyeRZ']].values
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
    

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer's Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Transformer's Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)
        
        # Embedding layers for input and output sequences
        self.encoder_embedding = nn.Embedding(input_size, d_model)
        self.decoder_embedding = nn.Embedding(output_size, d_model)
        
        # Output linear layer
        self.out = nn.Linear(d_model, output_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # Change the shape of the source and target to (seq_length, batch_size) for the transformer
        src = src.permute(5, 32)
        tgt = tgt.permute(5, 32)
        
        # Apply the embeddings
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        
        # Apply positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Pass through the transformer encoder
        memory = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Pass through the transformer decoder
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        
        # Pass through the final linear layer
        output = self.out(output)
        
        # Change the shape back to (batch_size, seq_length, output_size)
        output = output.permute(32, 5, 3)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)




# Split the dataset into training and validation sets
data_directory = os.path.expanduser("~/Desktop/Image_Processing/360-FoV-prediction/data/processed/yuchen_presenting.csv")
custom_dataset = HeadToGazeDataset(data_directory)
print("Should be 95", len(custom_dataset))
#print(custom_dataset.file_list)

train_dataset, val_dataset = train_test_split(custom_dataset, test_size=0.2, random_state=42)
val_dataset, test_dataset = train_test_split(val_dataset, test_size=0.5, random_state=42)

# Model hyperparameters
input_size = 6  # Number of features
output_size = 3 # Number of targets
d_model = 128  # Size of the internal representations of the model
nhead = 4
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 256
max_seq_length = custom_dataset.sequence_length
dropout = 0.1

# Instantiate the model
model = TransformerModel(input_size, output_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout)
learning_rate = 0.005
batch_size = 32


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,  collate_fn=collate_fn, shuffle=False)


# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(train_dataloader):
        features, gaze_direction = batch
        features, gaze_direction = features.to(device), gaze_direction.to(device)
        optimizer.zero_grad()
        # The input and target sequence dimensions need to be swapped to match the Transformer's expected input
        output = model(features.permute(1, 0, 2), gaze_direction[:-1].permute(1, 0, 2))
        loss = criterion(output, gaze_direction[1:].permute(1, 0, 2))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_dataloader)}')
"""
# Training loop
num_epochs = 100
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    total_train_loss = 0.0
    model.train()

    for i, batch in enumerate(train_dataloader):
        features, gaze_direction = batch
        features, gaze_direction = features.to(device), gaze_direction.to(device)
        # Assuming decoder_input is the same as target for simplicity (modify as needed)
        optimizer.zero_grad()
        output = model(features,gaze_direction)
        # output = seq2seq_model(gaze_direction)
        loss = criterion(output, gaze_direction)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        # print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_dataloader)}], Training Loss: {loss.item()}')
    # Validation
    model.eval()

    with torch.no_grad():
        total_val_loss = 0.0
        for batch in val_dataloader:
            features, gaze_direction = batch
            features, gaze_direction = features.to(device), gaze_direction.to(device)
            val_output = model(features,gaze_direction)
            total_val_loss += criterion(val_output, gaze_direction).item()

    average_train_loss = total_train_loss / len(train_dataloader)
    average_val_loss = total_val_loss / len(val_dataloader)
    train_losses.append(average_train_loss)
    val_losses.append(average_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {average_train_loss}, Average Validation Loss: {average_val_loss}')
    scheduler.step(average_val_loss)
"""
# Save the model if needed
torch.save(model.state_dict(), 'transformer_baseline.pth')
epochs_range = range(1, num_epochs + 1)
plt.plot(num_epochs, epoch_loss, label='Training Loss')
#plt.plot(epochs_range, train_losses, label='Training Loss')
#plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('seq2seq + MLP mixing: Sweep')
plt.legend()
plt.show()