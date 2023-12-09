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

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # Find the longest sequence
        max_len = max(x[0].shape[self.dim] for x in batch)
        # Pad according to max_len
        batch = [(pad_tensor(x, pad=max_len, dim=self.dim), y) for x, y in batch]
        # Stack all
        xs = torch.stack([x[0] for x in batch], dim=0)
        ys = torch.LongTensor([x[1] for x in batch])
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)

# Define the Seq2Seq model
class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqModel, self).__init__()

        # Encoder
        self.encoder_lstm = nn.LSTM(input_size, hidden_size)

        # Decoder
        self.decoder_lstm = nn.LSTM(output_size, hidden_size)
        self.decoder_dense = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_input, decoder_input):
        # Encoder
        packed_encoder_input = rnn_utils.pack_padded_sequence(encoder_input, encoder_lengths, batch_first=True)

        _, (encoder_h, encoder_c) = self.encoder_lstm(packed_encoder_input)

        # Decoder
        decoder_output, _ = self.decoder_lstm(decoder_input, (encoder_h, encoder_c))
        decoder_output = self.decoder_dense(decoder_output)

        return decoder_output

# Example usage:
data_directory = os.path.expanduser("~/360-FoV-prediction/data/processed")
custom_dataset = GazeDataSet(data_directory)

# Assuming you have defined the input_size, hidden_size, and output_size
input_size = 28
hidden_size = 256
output_size = 6

model = Seq2SeqModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# DataLoader for batching
batch_size = 16  # Adjust based on your needs
dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=PadCollate(dim=0), shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        encoder_input, target, encoder_lengths = batch

        # Assuming decoder_input is the same as target for simplicity (modify as needed)
        decoder_input = target

        optimizer.zero_grad()
        output = model(encoder_input, decoder_input, encoder_lengths)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Save the model if needed
torch.save(model.state_dict(), 'seq2seq_model.pth')
