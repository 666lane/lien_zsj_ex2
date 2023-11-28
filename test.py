import torch
import matplotlib.pyplot as plt
# from train import CustomDataset, DataLoader, SimpleModel
from torch.utils.data import Dataset, DataLoader
# Assuming you've already defined the SimpleModel class and loaded the trained model weights
import os
import numpy as np

# Specify the folder containing your input and output txt files
input_folder = 'C:/Users/48828/Desktop/my/experiment2/Dataset/In_test'
output_folder = 'C:/Users/48828/Desktop/my/experiment2/Dataset//Out_test'


class CustomDataset(Dataset):
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.input_file_list = sorted(os.listdir(input_folder))
        self.output_file_list = sorted(os.listdir(output_folder))

    def __len__(self):
        return len(self.input_file_list)

    def __getitem__(self, idx):
        input_file_path = os.path.join(self.input_folder, self.input_file_list[idx])
        output_file_path = os.path.join(self.output_folder, self.output_file_list[idx])

        input_data = np.loadtxt(input_file_path)
        output_data = np.loadtxt(output_file_path)

        return torch.Tensor(input_data), torch.Tensor(output_data)


# Create a dataset and dataloader for testing
test_dataset = CustomDataset(input_folder, output_folder)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)  # Set batch_size to 1 for simplicity

# Load the trained model

model = torch.load('simple_model.pth')
# model.eval()

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Test the model on a specific sample from the test set
with torch.no_grad():
    # Choose a specific sample index for testing
    sample_index = 0
    input_data, true_output = test_dataset[sample_index]
    input_data, true_output = input_data.unsqueeze(0).to(device), true_output.unsqueeze(0).to(device)

    # Forward pass
    predicted_output = model(input_data)

    # Convert tensors to numpy arrays
    input_data = input_data.cpu().numpy()
    true_output = true_output.cpu().numpy()
    predicted_output = predicted_output.cpu().numpy()

    # Plot the true and predicted output waveforms
    plt.figure(figsize=(10, 4))
    plt.plot(true_output[0], label='True Output', linewidth=2)
    plt.plot(predicted_output[0], label='Predicted Output', linestyle='dashed', linewidth=2)
    plt.title('True and Predicted Output Waveforms')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
