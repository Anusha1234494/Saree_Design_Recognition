# DeepLure-Saree-AI

DeepLure-Saree-AI is an AI-based saree design retrieval and verification system that identifies similar saree designs even when their colors differ.

## Technologies

* Python
* PyTorch
* ResNet18
* Siamese-style Architecture
* Contrastive Loss
* Computer Vision
* Cosine Similarity
* Streamlit

## Project Overview

The system uses a Siamese-style ResNet18 model to generate compact 128-dimensional image embeddings focused on saree design patterns rather than exact colors. Cosine similarity is then used to retrieve and verify similar designs.

## Key Features

* Identifies similar saree designs from a gallery
* Handles significant color variations
* Retrieves Top-5 similar designs
* Verifies whether two images represent the same or similar design
* Provides an interactive Streamlit dashboard
* Uses precomputed gallery embeddings for faster retrieval

## Model & Training

* ResNet18 backbone
* 128-dimensional embeddings
* Contrastive Loss
* 10 training epochs
* Image size: 224 × 224
* Augmentation: Random Flip, Color Jitter, and Random Grayscale
* Best model saved as `best_saree_model.pth`

## Results

* Gallery Images: 115
* Query Images: 25
* Top-1 Accuracy: 100%
* Top-5 Accuracy: 100%
* Verification Accuracy: 100%
* Precision: 100%
* Recall: 100%
* F1-Score: 100%
* ROC-AUC: 1.00
* Color-transformed same-design similarity: 98.47%
* Average embedding inference latency: 11.15 ms

## Streamlit Application

The application provides three sections:

1. **Identify Design** – Upload a saree image and retrieve the Top-5 most similar designs with similarity scores.
2. **Verify Design** – Compare two saree images using cosine similarity and a threshold of 0.9437.
3. **Analytics** – View model details, training performance, and evaluation metrics.

## Project Structure

```text
DeepLure_Saree_AI/
├── README.md
├── APPROACH_NOTE.md
├── DeepLure_Training.ipynb
├── app.py
├── identify_test.py
├── test.py
├── requirements.txt
└── model/
    ├── best_saree_model.pth
    ├── config.json
    ├── gallery_embeddings.npy
    └── gallery_images/
```

## Key Outcome

The project demonstrates a computer vision system capable of retrieving and verifying saree designs while maintaining robustness to color variations. The compact embedding approach also enables efficient comparison against precomputed gallery embeddings.
