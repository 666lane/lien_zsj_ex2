import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


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


# Specify the folders containing your input and output txt files
input_folder = 'C:/Users/48828/Desktop/my/experiment2/Dataset/In'
output_folder = 'C:/Users/48828/Desktop/my/experiment2/Dataset/Out'

# Create datasets and dataloaders
dataset = CustomDataset(input_folder, output_folder)
train_size = int(0.75 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
print(len(dataset))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define the neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.model1 = nn.Sequential(
            nn.Linear(1001, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# Initialize model, loss function, and optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Evaluation on the test set
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss / len(test_loader)}')

# Save the trained model
torch.save(model.state_dict(), 'simple_model.pth')
