import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from PIL import Image
from matplotlib import pyplot as plt

label_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(
    root='cifar10_images',
    transform=transform
)


train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

class SimpleCNN(pl.LightningModule):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_embd = nn.Embedding(10, 10)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 3 * 32 * 32),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        label_embd = self.label_embd(labels)
        x = torch.cat([z, label_embd], dim=1)
        x = self.fc(x)
        return x.view(-1, 3, 32, 32)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        z = torch.randn(x.size(0), self.latent_dim).type_as(x)
        generated = self(z, labels)
        loss = F.mse_loss(generated, x)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
model = SimpleCNN()

if input("train model? (y/n): ") == "y":
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='auto',
        devices='auto'
    )

    trainer.fit(model, train_loader)
else:
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

def generate_images(model, labels):
    model.eval()
    device = next(model.parameters()).device
    labels = labels.to(device)
    with torch.no_grad():
        z = torch.randn(labels.size(0), model.latent_dim, device=device)
        generated_images = model(z, labels)
        generated_images = (generated_images + 1) / 2
        return generated_images, labels

torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    
    while input("generate? (y/n): ") == "y":
        labels = input("Enter labels (0-9) separated by spaces: ")
        labels = torch.tensor([int(label) for label in labels.split()])

        generated_images, labels = generate_images(model, labels=labels)
        
        out_path="generated.png"
        save_image(generated_images, out_path, normalize=True)