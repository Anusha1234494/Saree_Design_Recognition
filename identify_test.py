import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms

from PIL import Image

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# MODEL
# ============================================================

class SiameseNetwork(nn.Module):

    def __init__(self, embedding_dim=128):

        super().__init__()

        backbone = models.resnet18(
            weights=None
        )

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


# ============================================================
# LOAD MODEL
# ============================================================

model = SiameseNetwork(
    embedding_dim=128
)

model.load_state_dict(
    torch.load(
        "model/best_saree_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("✅ Model loaded")


# ============================================================
# LOAD GALLERY
# ============================================================

gallery_embeddings = np.load(
    "model/gallery_embeddings.npy"
)

gallery_paths = pd.read_csv(
    "model/gallery_paths.csv"
)["path"].tolist()

print(
    "Gallery:",
    len(gallery_paths),
    "images"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# GET EMBEDDING
# ============================================================

def get_embedding(image):

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    ).to(device)

    with torch.no_grad():

        embedding = model.forward_once(
            tensor
        )

    return embedding.cpu().numpy()[0]


# ============================================================
# LOAD QUERY IMAGE
# ============================================================

query_path = "test_images/test_saree.jpg"

query_image = Image.open(
    query_path
).convert("RGB")

print(
    "Query image:",
    query_path
)


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

query_embedding = get_embedding(
    query_image
)


# ============================================================
# CALCULATE SIMILARITY
# ============================================================

similarities = cosine_similarity(
    query_embedding.reshape(1, -1),
    gallery_embeddings
)[0]


# ============================================================
# TOP 5
# ============================================================

top_indices = np.argsort(
    similarities
)[::-1][:5]


print("\n==============================")
print("TOP 5 SIMILAR DESIGNS")
print("==============================")

for rank, index in enumerate(
    top_indices,
    start=1
):

    print(
        f"\nRank {rank}"
    )

    print(
        f"Similarity: "
        f"{similarities[index]:.4f}"
    )

    print(
        f"Image: "
        f"{gallery_paths[index]}"
    )