import torch
import torch.nn as nn

class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        super(Seq2SeqModel, self).__init__()

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)

        # Decoder LSTMCell
        self.decoder_lstm = nn.LSTMCell(input_size=output_size, hidden_size=hidden_size)

        # Projection layer for final prediction
        self.projection_layer = nn.Linear(hidden_size, output_size)

        self.seq_length = seq_length

    def forward(self, input_data):
        # Encoder
        encoder_hidden_states, (encoder_last_hidden, encoder_last_memory) = self.encoder_lstm(input_data)

        # Decoder
        decoder_hidden_states = []
        decoder_last_hidden = encoder_last_hidden
        decoder_last_memory = encoder_last_memory

        # Initial input for the decoder (you might replace this with actual initial input)
        decoder_input = torch.zeros_like(decoder_last_hidden)

        for _ in range(self.seq_length):
            # Flatten the input before passing it to LSTMCell
            decoder_input_flat = decoder_input.view(-1, decoder_input.size(-1))
            
            # Decoder LSTMCell
            decoder_last_hidden, decoder_last_memory = self.decoder_lstm(decoder_input_flat, (decoder_last_hidden, decoder_last_memory))

            # Projection layer for prediction
            prediction = self.projection_layer(decoder_last_hidden)

            # Append prediction to the list
            decoder_hidden_states.append(prediction)

            # Update decoder input for the next iteration
            decoder_input = prediction

        # Stack predictions along the sequence length dimension
        predictions = torch.stack(decoder_hidden_states, dim=1)

        return predictions

# Example usage:
input_size = 3  # Input size (assuming (x, y, z) coordinates)
hidden_size = 256  # Hidden size for both encoder and decoder
output_size = 2  # Output size (assuming mean and std for each time step)
seq_length = 10  # Sequence length for future predictions

# Create an instance of the Seq2SeqModel
model = Seq2SeqModel(input_size, hidden_size, output_size, seq_length)

# Dummy input data (adjust batch size as needed)
batch_size = 32
input_data = torch.rand((batch_size, 10, input_size))

# Forward pass
output = model(input_data)
print(output.shape)  # Adjust the shape based on your specific output size
