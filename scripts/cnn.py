import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Compose, ToTensor, RandomRotation, RandomAffine, ColorJitter, GaussianBlur, Resize
)
from PIL import Image, ImageDraw, ImageFont
import random
import os

# -------------------------
# Configuration
# -------------------------
CHAR_CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ!?:")
NUM_CLASSES = len(CHAR_CLASSES)
IMAGE_SIZE = 28
SAMPLES_PER_CLASS = 1000
EPOCHS = 10
BATCH_SIZE = 64
MODEL_PATH = "char_symbol_cnn.pt"

# -------------------------
# CNN Model
# -------------------------
class SimpleCharCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Dataset with Augmentation
# -------------------------
class CharSymbolDataset(Dataset):
    def __init__(self, characters=CHAR_CLASSES, samples_per_class=SAMPLES_PER_CLASS, img_size=IMAGE_SIZE):
        self.characters = characters
        self.count = samples_per_class * len(characters)
        self.img_size = img_size
        self.font = ImageFont.load_default()
        self.data = []
        self.labels = []

        self.augment = Compose([
            Resize((img_size, img_size)),
            RandomRotation(degrees=30),
            RandomAffine(degrees=0, scale=(0.8, 1.2), translate=(0.1, 0.1)),
            ColorJitter(brightness=0.3, contrast=0.3),
            GaussianBlur(kernel_size=3),
            ToTensor()
        ])

        for idx, char in enumerate(characters):
            for _ in range(samples_per_class):
                img = Image.new("L", (img_size, img_size), color=255)
                draw = ImageDraw.Draw(img)
                draw.text((8, 6), char, fill=0, font=self.font)
                self.data.append(img)
                self.labels.append(idx)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        aug_image = self.augment(image)
        return aug_image, label

# -------------------------
# Training Function
# -------------------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CharSymbolDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleCharCNN(NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "class_names": CHAR_CLASSES
    }, MODEL_PATH)

    print(f"\n✅ Model trained and saved to '{MODEL_PATH}'")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    train_model()
