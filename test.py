import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# -----------------------------
# Device
# -----------------------------
device = torch.device("cpu")


# -----------------------------
# Same architecture as Kaggle
# -----------------------------
class SiameseNetwork(nn.Module):

    def __init__(self, embedding_dim=128):

        super().__init__()

        backbone = models.resnet18(weights=None)

        num_features = backbone.fc.in_features

        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.embedding = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )

    def forward_once(self, x):

        features = self.backbone(x)

        embedding = self.embedding(features)

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding

    def forward(self, image1, image2):

        return (
            self.forward_once(image1),
            self.forward_once(image2)
        )


# -----------------------------
# Load model
# -----------------------------
print("Loading model...")

model = SiameseNetwork(
    embedding_dim=128
)

state_dict = torch.load(
    "model/best_saree_model.pth",
    map_location=device
)

model.load_state_dict(state_dict)

model.to(device)
model.eval()

print("✅ Model loaded successfully!")

# -----------------------------
# Load gallery
# -----------------------------
import numpy as np
import pandas as pd
import json

gallery_embeddings = np.load(
    "model/gallery_embeddings.npy"
)

gallery_paths = pd.read_csv(
    "model/gallery_paths.csv"
)["path"].tolist()

with open(
    "model/config.json",
    "r"
) as f:
    config = json.load(f)


print("Gallery images:", len(gallery_paths))
print(
    "Embedding shape:",
    gallery_embeddings.shape
)

print(
    "Verification threshold:",
    config["verification_threshold"]
)