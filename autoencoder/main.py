from model import ResNetAutoencoder
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Collect all image file paths from subdirectories
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('png', 'jpg', 'jpeg')):  # Include only image files
                    self.image_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Open image as grayscale (1-channel)
        image = Image.open(img_path).convert('L')  # 'L' mode for 1-channel grayscale

        if self.transform:
            image = self.transform(image)

        return image

# Function to visualize with Matplotlib (same window, updates each iteration)
def visualize_in_ion(original, reconstructed, fig, ax1, ax2):
    """
    Visualize the original and reconstructed images using Matplotlib (interactive mode).

    Args:
        original (torch.Tensor): Original image tensor (1-channel).
        reconstructed (torch.Tensor): Reconstructed image tensor (1-channel).
        fig (plt.Figure): Matplotlib figure object.
        ax1 (plt.Axes): Axis for the original image.
        ax2 (plt.Axes): Axis for the reconstructed image.
    """
    original = original.cpu().squeeze().numpy()
    reconstructed = reconstructed.cpu().squeeze().detach().numpy()

    ax1.clear()
    ax1.imshow(original, cmap="magma")
    ax1.set_title("Original")
    ax1.axis("off")

    ax2.clear()
    ax2.imshow(reconstructed, cmap="magma")
    ax2.set_title("Reconstructed")
    ax2.axis("off")

    fig.canvas.draw()
    plt.pause(0.001)  # Pause to allow the figure to update

# Dataset paths
train_data_path = "path/to/your/train/data"
validation_data_path = "/path/to/your/validation/data"

# Model, loss function, optimizer
model = ResNetAutoencoder(1, 512)
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Transformations
data_transforms = transforms.Compose([
    transforms.Resize((16, 720)),
    transforms.ToTensor()
])

# Dataset and DataLoader
train_dataset = CustomDataset(root_dir=train_data_path, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

validation_dataset = CustomDataset(root_dir=validation_data_path, transform=data_transforms)
validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

# Training loop
num_epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter('runs/autoencoder_experiment')

# Initialize Matplotlib for interactive visualization
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))  # Two subplots: original and reconstructed

global_step = 0  # Global step for TensorBoard

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        images = data.to(device)  # Move images to the device (GPU/CPU)

        optimizer.zero_grad()  # Zero out gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, images)  # Compute loss

        # Backpropagation
        loss.backward()
        optimizer.step()  # Update model parameters
        running_loss += loss.item()

        global_step += 1  # Increment global step

        # Visualize the first image in the batch using Matplotlib (interactive mode)
        if i % 100 == 0:  # Adjust visualization frequency as needed
            visualize_in_ion(images[0], outputs[0], fig, ax1, ax2)
            # Validation phase
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():
                for images in validation_loader:
                    images = images.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, images).item()

                val_loss /= len(validation_loader)
                print(f'Validation Loss: {val_loss:.4f}')
                writer.add_scalar('Validation Loss', val_loss, epoch + 1)  # Log validation loss
        # Print training loss every 10 batches
        if (i + 1) % 10 == 0:
            avg_loss = running_loss / 10
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}')
            writer.add_scalar('Training Loss', avg_loss, global_step)  # Log training loss
            running_loss = 0.0

    # save model for each epoch with name including epoch number
    torch.save(model.state_dict(), '/path/to/model/directory/autoencoder_epoch_{}.pth'.format(epoch))
# Close TensorBoard writer
writer.close()

# Close interactive mode after training
plt.ioff()
plt.show()

# Save the trained model
