'''
not currentcly used
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class GLULayer(nn.Module):
    def forward(self, x, gate):
        return x * torch.sigmoid(gate)

class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Seq2SeqModel, self).__init__()

        self.encoder1 = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.encoder2 = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

        self.decoder1 = nn.LSTM(output_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder2 = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

        self.decoder_dense = nn.Linear(hidden_size, output_size)

        self.flatten = Flatten()
        self.expand_dim = lambda x: x.unsqueeze(1)
        self.concat = lambda x: torch.cat(x, dim=1)
        self.glu = GLULayer()

    def forward(self, encoder_input, others_fut_input, decoder_input=None):
        encoder1_outputs, (state_h_1, state_c_1) = self.encoder1(encoder_input)
        encoder2_outputs, (state_h_2, state_c_2) = self.encoder2(encoder1_outputs)

        decoder1_states_inputs = (state_h_1, state_c_1)
        decoder2_states_inputs = (state_h_2, state_c_2)

        decoder_outputs = []

        inputs = self.expand_dim(decoder_input) if decoder_input is not None else encoder_input[:, -1, :].unsqueeze(1)

        for _ in range(max_decoder_seq_length):
            decoder1_outputs, (state_decoder1_h, state_decoder1_c) = self.decoder1(inputs, decoder1_states_inputs)
            decoder2_outputs, (state_decoder2_h, state_decoder2_c) = self.decoder2(decoder1_outputs, decoder2_states_inputs)

            if target_user_only:
                outputs = self.decoder_dense(decoder2_outputs)
            else:
                # Process others_fut_input and concatenate with decoder2_outputs
                # Update this part based on the specific logic you want to implement

                concat_state = self.concat([others_fut_input, decoder2_outputs])
                outputs = self.expand_dim(self.decoder_dense(concat_state))

            decoder_outputs.append(outputs)
            inputs = outputs

        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs

# Define model and optimizer
input_size = 6  # Update based on your input data
output_size = 6  # Update based on your output data
hidden_size = 32
max_decoder_seq_length = 10  # Update based on your sequence length
num_layers = 2
num_user = 34  # Adjust the number of users as needed

model = Seq2SeqModel(input_size, hidden_size, output_size, num_layers=num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
batch_size = 32  # Adjust the batch size as needed

# Dummy input data (replace with your actual data loading logic)
encoder_input_data = torch.rand((batch_size, 10, input_size))
others_fut_input_data = torch.rand((batch_size, max_decoder_seq_length, num_user-1, 6))
decoder_input_data = torch.rand((batch_size, 1, output_size))

# Training loop
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()

    # Replace max_decoder_seq_length with the actual length or define it appropriately
    max_decoder_seq_length = 10  # Update based on your sequence length or logic
    
    decoder_outputs = []
    
    # Loop through the decoder sequence
    for step in range(max_decoder_seq_length):
        # Update decoder_input_data based on your logic
        inputs = decoder_input_data[:, step:step+1, :] if decoder_input_data is not None else encoder_input_data[:, -1, :].unsqueeze(1)

        encoder_input = encoder_input_data if step == 0 else None  # Use encoder_input only in the first step

        # You might need to adjust the following line based on your specific logic
        decoder_outputs_step = model(encoder_input, others_fut_input_data, inputs)
        
        decoder_outputs.append(decoder_outputs_step)

    decoder_outputs = torch.cat(decoder_outputs, dim=1)

    # Replace decoder_target_data with the actual target data for training
    loss = criterion(decoder_outputs, decoder_target_data)
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


# Save or use the trained model as needed
