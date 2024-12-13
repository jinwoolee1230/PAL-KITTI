import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from model import ResNetAutoencoder
from torch.utils.data import Dataset, DataLoader

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(subdir, file)
            for subdir, _, files in os.walk(root_dir)
            for file in files if file.lower().endswith(('png', 'jpg', 'jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, img_path  # 이미지와 경로 반환

# Visualization function
def visualize_images(original, reconstructed, fig, ax1, ax2, img_path):
    original = original.cpu().squeeze().numpy()
    reconstructed = reconstructed.cpu().squeeze().detach().numpy()

    ax1.clear()
    ax1.imshow(original, cmap="magma")
    ax1.set_title(f"Original: {os.path.basename(img_path)}")
    ax1.axis("off")

    ax2.clear()
    ax2.imshow(reconstructed, cmap="magma")
    ax2.set_title("Reconstructed")
    ax2.axis("off")

    fig.canvas.draw()
    plt.pause(0.1)

# Paths
test_data_path = "/path/to/your/test/data"
model_path = "/path/to/your/model.pth"

# Transformations
data_transforms = transforms.Compose([
    transforms.Resize((16, 720)),
    transforms.ToTensor()
])

# Dataset and DataLoader
test_dataset = CustomDataset(root_dir=test_data_path, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNetAutoencoder(1, 512).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Matplotlib setup
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

# Inference and visualization
for images, img_path in test_loader:
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    # Visualize original and reconstructed images
    visualize_images(images[0], outputs[0], fig, ax1, ax2, img_path[0])

plt.ioff()
plt.show()
