# DeepLure-Saree-AI


An AI-based saree design retrieval and verification system that identifies similar saree designs even when their colors are different.

The project uses a Siamese-style ResNet18 architecture to learn design-focused image embeddings. Instead of depending mainly on color, the model learns visual patterns and structure that help match the same or similar saree design.

## Problem Statement

Sarees can have the same design pattern in different colors. A traditional image similarity system may consider these images very different because of the color variation.

The goal of this project is to build a system that can:

- Identify similar saree designs
- Handle significant color changes
- Retrieve the most similar designs from a gallery
- Verify whether two saree images represent the same or similar design

## Approach

The system follows a Siamese-style image embedding approach.

Each saree image is passed through a ResNet18 backbone and converted into a compact 128-dimensional embedding.

The embeddings are compared using cosine similarity.

### Pipeline

Input Image
→ Image Preprocessing
→ ResNet18 Feature Extraction
→ 128-D Embedding
→ Cosine Similarity
→ Retrieval / Verification

## Data Preprocessing

Images are:

- Resized to `224 × 224`
- Converted to tensors
- Normalized using ImageNet mean and standard deviation

Normalization:

- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`

## Data Augmentation

To improve robustness to color variations, the training pipeline uses:

- Random Horizontal Flip
- Color Jitter
- Random Grayscale

Color augmentation is particularly important because the main objective is to focus more on saree design and pattern rather than exact color.

## Model

### Backbone

ResNet18

### Embedding Size

128 dimensions

### Similarity Measure

Cosine Similarity

### Loss

Contrastive Loss

The model learns to bring similar designs closer in embedding space and push different designs farther apart.

## Training

The model was trained for 10 epochs using paired images.

The training and validation loss were monitored during training.

The best trained model is saved as:

`model/best_saree_model.pth`

## Identification

For identification, the system compares a query image against the precomputed gallery embeddings.

The gallery contains:

- 115 gallery images
- 25 query images

For each query:

1. Generate the query embedding
2. Compare it with all gallery embeddings
3. Calculate cosine similarity
4. Rank the gallery images
5. Return the most similar designs

### Identification Results

| Metric | Result |
|---|---:|
| Gallery Images | 115 |
| Query Images | 25 |
| Top-1 Accuracy | 100% |
| Top-5 Accuracy | 100% |

All 25 queries retrieved the correct design at Rank 1 in the evaluation.

## Verification

The verification task determines whether two images represent the same or similar saree design.

The evaluation used:

- 25 positive pairs
- 25 negative pairs

Cosine similarity was used as the verification score.

The decision threshold was selected using the ROC curve and Youden's J statistic.

### Selected Threshold

`0.9437`

### Verification Results

| Metric | Result |
|---|---:|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1-Score | 100% |
| ROC-AUC | 1.00 |

-These results are based on the implemented evaluation protocol and should not be interpreted as a guarantee of performance on unseen real-world data.

## Color Invariance

-A separate experiment was performed to check whether the model remains consistent when the same design undergoes color changes.

### Results

| Comparison | Average Similarity |
|---|---:|
| Same Design + Color Transformation | 98.47% |
| Different Images | 78.15% |

The high similarity for the same design after color transformation indicates that the learned embedding is relatively robust to color changes.

## Efficiency

The system uses a compact 128-dimensional embedding, making gallery storage and similarity comparison lightweight.

Measured embedding inference latency:

| Metric | Latency |
|---|---:|
| Average | 11.15 ms |
| Minimum | 10.73 ms |
| Maximum | 11.77 ms |

The gallery embeddings are precomputed, so during inference the system only needs to generate one embedding for the query image and compare it with the stored gallery embeddings.

## Streamlit Application

A Streamlit dashboard was developed to demonstrate the complete system.

The application contains three main sections:

### 1. Identify Design

Upload a saree image and retrieve the most similar designs from the gallery.

The application displays:

- Uploaded image
- Top-5 similar designs
- Similarity scores
- Similarity visualization

### 2. Verify Design

Upload two saree images and check whether they represent the same or similar design.

The application provides:

- Cosine similarity score
- Verification threshold
- Verification result
- Visual similarity indication

### 3. Analytics

The analytics section displays project-related information such as:

- Training performance
- Evaluation metrics
- Similarity results
- Model information

# How to Run in VS Code

1. Clone the repository and open the project folder in VS Code.

2. Open the VS Code Terminal and run:

       git clone <your-github-repository-url>
       cd DeepLure_Saree_AI

3. Create a virtual environment:

       python -m venv venv

4. Activate it:
    
       Windows:
       venv\Scripts\activate

5. Install the required libraries:

       pip install -r requirements.txt

6. Make sure the required model files and local gallery files are available:
   
       model/best_saree_model.pth
       model/config.json
       model/gallery_embeddings.npy
       model/gallery_images/
       model/gallery_paths.csv

7. Run the Streamlit application:

       streamlit run app.py

8. Open the URL shown in the terminal, usually:

       http://localhost:8501

9. Use the dashboard:

- Identify Design → Upload a saree image and retrieve Top-5 similar designs.
- Verify Design → Upload two images and check design similarity.
- Analytics → View model and evaluation results.


## Project Structure

```text
DeepLure_Saree_AI/
│
├── README.md
├── APPROACH_NOTE.md
├── DeepLure_Training.ipynb
├── app.py
├── identify_test.py
├── test.py
├── requirements.txt
├── .gitignore
│
└── model/
    ├── best_saree_model.pth
    ├── config.json
    └── gallery_embeddings.npy

