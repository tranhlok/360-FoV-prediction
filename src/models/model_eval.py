import torch
# import your_model_module  # Replace with the actual module where your model is defined

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Step 1: Define your model architecture (exactly as it was when you saved it)
# model = your_model_module.YourModelClass(input_size, hidden_size, output_size)  # Replace with your model class and architecture

# Step 2: Create an instance of your model
model = model.to(device)  # Send the model to the appropriate device (e.g., 'cuda' or 'cpu')

# Step 3: Load the model's state dictionary from the .pth file
model.load_state_dict(torch.load('your_model.pth', map_location=device))  # Replace 'your_model.pth' with the actual path to your .pth file

# Step 4: Set the model to evaluation mode
model.eval()
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
