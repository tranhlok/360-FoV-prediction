import torch
import torch.nn as nn
import os
import sys
from torch_custom_dataset import GazeDataSet
# from data.torch_custom_dataset import GazeDataSet # for running in the main dir
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils

# def collate_fn(batch):
#     # batch is a list of (sequence, target, sequence_length) tuples
#     sequences, targets, lengths = zip(*batch)

#     # Sort batch by sequence length for pack_padded_sequence
#     sorted_lengths, sorted_indices = torch.sort(torch.tensor(lengths), descending=True)
#     sorted_sequences = [sequences[i] for i in sorted_indices]
#     sorted_targets = [targets[i] for i in sorted_indices]

#     # Pack sequences
#     packed_sequences = rnn_utils.pack_sequence(sorted_sequences)

#     return packed_sequences, torch.stack(sorted_targets), sorted_lengths

# def collate_fn(batch):
#     features, gaze_direction = zip(*batch)

#     # Pad sequences to the length of the longest sequence in the batch
#     padded_features = rnn_utils.pad_sequence(features, batch_first=True, padding_value=0)
#     padded_gaze_direction = rnn_utils.pad_sequence(gaze_direction, batch_first=True, padding_value=0)

#     return padded_features, padded_gaze_direction


# # Define the Seq2Seq model
# class Seq2SeqModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Seq2SeqModel, self).__init__()

#         # Encoder
#         self.encoder_lstm = nn.LSTM(input_size, hidden_size)

#         # Decoder
#         self.decoder_lstm = nn.LSTM(output_size, hidden_size)
#         self.decoder_dense = nn.Linear(hidden_size, output_size)

#     def forward(self, encoder_input, decoder_input,encoder_lengths):
#         # Encoder
#         packed_encoder_input = rnn_utils.pack_padded_sequence(encoder_input,batch_first=True)

#         _, (encoder_h, encoder_c) = self.encoder_lstm(packed_encoder_input)

#         # Decoder
#         decoder_output, _ = self.decoder_lstm(decoder_input, (encoder_h, encoder_c))
#         decoder_output = self.decoder_dense(decoder_output)

#         return decoder_output

# class Seq2SeqModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Seq2SeqModel, self).__init__()
#         self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, encoder_input, decoder_input, encoder_lengths):
#         _, (encoder_hidden, _) = self.encoder(encoder_input)
#         decoder_output, _ = self.decoder(decoder_input, (encoder_hidden, _))
#         output = self.fc(decoder_output)
#         return output

class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqModel, self).__init__()
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

def collate_fn(batch):
    features, gaze_direction = zip(*batch)

    # Pad sequences to the length of the longest sequence in the batch
    padded_features = rnn_utils.pad_sequence(features, batch_first=True, padding_value=0)
    padded_gaze_direction = rnn_utils.pad_sequence(gaze_direction, batch_first=True, padding_value=0)

    return padded_features, padded_gaze_direction

# Example usage:
data_directory = os.path.expanduser("~/360-FoV-prediction/data/processed")
custom_dataset = GazeDataSet(data_directory)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(custom_dataset))
test_size = len(custom_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])

# DataLoader for batching
batch_size = 16
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

# Model and training loop
input_size = 20  # Adjust based on your features dimension
hidden_size = 256
output_size = 6

model = Seq2SeqModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        encoder_input, target = batch

        # Assuming decoder_input is the same as target for simplicity (modify as needed)
        decoder_input = target

        optimizer.zero_grad()
        output = model(encoder_input, decoder_input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    # Validation
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input, target = batch
            decoder_input = target
            output = model(encoder_input, decoder_input)
            val_loss = criterion(output, target)

    print(f'Validation Loss: {val_loss.item()}')

# Save the model if needed
torch.save(model.state_dict(), 'seq2seq_model.pth')


# # Example usage:
# data_directory = os.path.expanduser("~/360-FoV-prediction/data/processed")
# custom_dataset = GazeDataSet(data_directory)

# input_size = 20  # Adjust based on your features dimension
# hidden_size = 256
# output_size = 6

# model = Seq2SeqModel(input_size, hidden_size, output_size)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters())

# # DataLoader for batching
# batch_size = 16  # Adjust based on your needs
# dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         encoder_input, target = batch

#         # Assuming decoder_input is the same as target for simplicity (modify as needed)
#         decoder_input = target

#         optimizer.zero_grad()
#         output = model(encoder_input, decoder_input, None)  # No need for encoder_lengths in this example
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# # Save the model if needed
# torch.save(model.state_dict(), 'seq2seq_model.pth')

