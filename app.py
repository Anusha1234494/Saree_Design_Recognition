# ============================================================
# DEEPLURE — COLOR-INVARIANT SAREE DESIGN INTELLIGENCE
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DeepLure | Saree AI",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================================================
   GLOBAL DARK BACKGROUND
   ========================================================= */

.stApp {
    background: #080d19 !important;
    color: #f8fafc !important;
}

/* Main application */
[data-testid="stAppViewContainer"] {
    background: #080d19 !important;
}

/* Main content */
[data-testid="stAppViewContainer"] > .main {
    background: #080d19 !important;
}

/* Prevent white header */
[data-testid="stHeader"] {
    background: #080d19 !important;
}

/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: #050914 !important;
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] > div {
    background: #050914 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}


/* =========================================================
   HEADINGS
   ========================================================= */

h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
}


/* =========================================================
   NORMAL TEXT
   ========================================================= */

p {
    color: #cbd5e1 !important;
}

span {
    color: #cbd5e1;
}


/* =========================================================
   MAIN TITLE
   ========================================================= */

.main-title {
    color: #f8fafc !important;
    font-size: 42px;
    font-weight: 800;
}

.subtitle {
    color: #94a3b8 !important;
    font-size: 17px;
}


/* =========================================================
   DARK CARDS
   ========================================================= */

.metric-card {
    background: #111827 !important;
    border: 1px solid #263244;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}

.rank-card {
    background: #111827 !important;
    border: 1px solid #263244;
    border-radius: 16px;
    padding: 15px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}


/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-title {
    color: #f8fafc !important;
    font-size: 27px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 15px;
}


/* =========================================================
   FILE UPLOADER — DARK
   ========================================================= */

[data-testid="stFileUploader"] {
    background: #111827 !important;
    border: 1px solid #334155 !important;
    border-radius: 14px !important;
}

[data-testid="stFileUploader"] section {
    background: #111827 !important;
    border: none !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #111827 !important;
    border: 1px dashed #475569 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #cbd5e1 !important;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    background: #4f46e5 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    min-height: 48px;
}

.stButton > button:hover {
    background: #6366f1 !important;
    color: white !important;
}


/* =========================================================
   METRICS
   ========================================================= */

[data-testid="stMetric"] {
    background: #111827 !important;
    border: 1px solid #263244 !important;
    border-radius: 14px !important;
    padding: 15px !important;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

[data-testid="stMetricDelta"] {
    color: #a5b4fc !important;
}


/* =========================================================
   INFO / SUCCESS / WARNING BOXES
   ========================================================= */

[data-testid="stAlert"] {
    background: #111827 !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}


/* =========================================================
   RADIO BUTTONS
   ========================================================= */

[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
}


/* =========================================================
   DIVIDERS
   ========================================================= */

hr {
    border-color: #263244 !important;
}


/* =========================================================
   DATAFRAME
   ========================================================= */

[data-testid="stDataFrame"] {
    background: #111827 !important;
}


/* =========================================================
   EXPANDERS
   ========================================================= */

[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid #263244 !important;
    border-radius: 12px !important;
}


/* =========================================================
   INPUTS
   ========================================================= */

input {
    background: #111827 !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
}


/* =========================================================
   SELECT BOX
   ========================================================= */

[data-baseweb="select"] > div {
    background: #111827 !important;
    color: #f8fafc !important;
    border-color: #334155 !important;
}


/* =========================================================
   CHECKBOX
   ========================================================= */

[data-testid="stCheckbox"] label {
    color: #cbd5e1 !important;
}


/* =========================================================
   SPINNER
   ========================================================= */

.stSpinner > div {
    border-top-color: #6366f1 !important;
}


/* =========================================================
   FOOTER
   ========================================================= */

footer {
    background: #080d19 !important;
}


/* =========================================================
   SCROLLBAR
   ========================================================= */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #080d19;
}

::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_saree_model.pth"
)

EMBEDDINGS_PATH = os.path.join(
    MODEL_DIR,
    "gallery_embeddings.npy"
)

GALLERY_CSV = os.path.join(
    MODEL_DIR,
    "gallery_paths.csv"
)

GALLERY_IMAGE_DIR = os.path.join(
    MODEL_DIR,
    "gallery_images"
)

THRESHOLD = 0.9437035918235779


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# MODEL ARCHITECTURE
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
            nn.Linear(
                num_features,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                embedding_dim
            )
        )

    def forward_once(self, x):

        features = self.backbone(x)

        embedding = self.embedding(
            features
        )

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

eval_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = SiameseNetwork(
        embedding_dim=128
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)

    model.eval()

    return model


# ============================================================
# LOAD GALLERY
# ============================================================

@st.cache_resource
def load_gallery():

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    gallery_df = pd.read_csv(
        GALLERY_CSV
    )

    return embeddings, gallery_df


# ============================================================
# GET IMAGE EMBEDDING
# ============================================================

def get_embedding(
    image,
    model
):

    tensor = eval_transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(device)

    with torch.no_grad():

        embedding = model.forward_once(
            tensor
        )

    return embedding.cpu().numpy()[0]


# ============================================================
# GET LOCAL GALLERY IMAGE
# ============================================================

def get_gallery_image(
    index,
    gallery_df
):

    path = str(
        gallery_df.iloc[index]["path"]
    )

    # If path is already local
    if os.path.exists(path):

        return path

    # Normal packaged format
    filename = os.path.basename(
        path
    )

    local_path = os.path.join(
        GALLERY_IMAGE_DIR,
        filename
    )

    if os.path.exists(local_path):

        return local_path

    # Try gallery index naming
    indexed_path = os.path.join(
        GALLERY_IMAGE_DIR,
        f"gallery_{index:03d}.jpg"
    )

    if os.path.exists(indexed_path):

        return indexed_path

    return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;">

        <div style="font-size:55px;">
        🧵
        </div>

        <h1>DeepLure</h1>

        <p>
        Color-Invariant Saree<br>
        Design Intelligence
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🔎 Identify Design",
            "🔐 Verify Design",
            "📊 Analytics"
        ]
    )

    st.divider()

    st.markdown(
        """
        **AI Architecture**

        🧠 ResNet18 Backbone

        ↓

        🔢 128-D Embedding

        ↓

        📐 Cosine Similarity

        ↓

        🎯 Top-K Retrieval
        """
    )

    st.divider()

    st.caption(
        "DeepLure AI Assessment Prototype"
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    model = load_model()

    gallery_embeddings, gallery_df = (
        load_gallery()
    )

except Exception as e:

    st.error(
        "Unable to load the model or gallery."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# PAGE 1 — IDENTIFY DESIGN
# ============================================================

if page == "🔎 Identify Design":

    st.markdown(
        '<div class="main-title">🧵 Saree Design Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Upload a saree image and discover the most visually similar
        designs while remaining robust to color variations.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload a saree design",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        help="Upload an image of a saree or textile design."
    )


    if uploaded_file is None:

        st.info(
            "👆 Upload a saree image to start the AI analysis."
        )

        st.markdown(
            """
            ### How it works

            **1. Upload**

            Provide a saree design image.

            **2. Generate embedding**

            The neural network converts the image into a
            128-dimensional visual representation.

            **3. Search**

            The embedding is compared with the gallery.

            **4. Retrieve**

            The five most similar designs are displayed.
            """
        )


    else:

        query_image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ----------------------------------------------------
        # Query image
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">📷 Query Image</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(
            [1, 2]
        )

        with col1:

            st.image(
                query_image,
                caption="Uploaded Design",
                use_container_width=True
            )

        with col2:

            st.markdown(
                """
                ### Ready for analysis

                The AI will compare this image against
                the DeepLure design gallery.
                """
            )

            if st.button(
                "🔍 Analyze Design",
                type="primary",
                use_container_width=True
            ):

                progress = st.progress(
                    0
                )

                status = st.empty()

                status.info(
                    "🔄 Preparing image..."
                )

                progress.progress(
                    25
                )

                time.sleep(
                    0.3
                )

                status.info(
                    "🧠 Generating visual embedding..."
                )

                start_time = time.perf_counter()

                query_embedding = get_embedding(
                    query_image,
                    model
                )

                progress.progress(
                    55
                )

                status.info(
                    "📐 Comparing with gallery..."
                )

                similarities = cosine_similarity(
                    query_embedding.reshape(
                        1,
                        -1
                    ),
                    gallery_embeddings
                )[0]

                top_indices = np.argsort(
                    similarities
                )[::-1][:5]

                inference_time = (
                    time.perf_counter()
                    - start_time
                ) * 1000

                progress.progress(
                    100
                )

                status.success(
                    "✅ Analysis complete!"
                )

                time.sleep(
                    0.4
                )

                progress.empty()


                # Store results in session
                st.session_state[
                    "similarities"
                ] = similarities

                st.session_state[
                    "top_indices"
                ] = top_indices

                st.session_state[
                    "query_embedding"
                ] = query_embedding

                st.session_state[
                    "inference_time"
                ] = inference_time


        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        if (
            "top_indices"
            in st.session_state
        ):

            similarities = (
                st.session_state[
                    "similarities"
                ]
            )

            top_indices = (
                st.session_state[
                    "top_indices"
                ]
            )

            inference_time = (
                st.session_state[
                    "inference_time"
                ]
            )


            st.divider()


            st.markdown(
                '<div class="section-title">🎯 Top 5 Design Matches</div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # Top 5 cards
            # ------------------------------------------------

            cols = st.columns(
                5
            )

            for rank, (
                col,
                index
            ) in enumerate(
                zip(
                    cols,
                    top_indices
                ),
                start=1
            ):

                similarity = (
                    similarities[index]
                    * 100
                )

                image_path = (
                    get_gallery_image(
                        index,
                        gallery_df
                    )
                )

                with col:

                    st.markdown(
                        f"""
                        <div class="rank-card">

                        <div style="font-size:24px;">
                        {"🥇" if rank == 1 else
                         "🥈" if rank == 2 else
                         "🥉" if rank == 3 else
                         "🏅"}
                        </div>

                        <div class="rank-number">
                        Rank {rank}
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if image_path:

                        st.image(
                            image_path,
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "Gallery image unavailable"
                        )

                    st.metric(
                        "Similarity",
                        f"{similarity:.1f}%"
                    )


            # ------------------------------------------------
            # Chart
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">📈 Similarity Analysis</div>',
                unsafe_allow_html=True
            )

            scores = [
                similarities[index] * 100
                for index in top_indices
            ]

            rank_labels = [
                f"Rank {i}"
                for i in range(
                    1,
                    6
                )
            ]

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=rank_labels,
                    y=scores,
                    text=[
                        f"{x:.1f}%"
                        for x in scores
                    ],
                    textposition="auto"
                )
            )

            fig.update_layout(
                title="Top-5 Design Similarity",
                xaxis_title="Retrieved Rank",
                yaxis_title="Similarity (%)",
                yaxis_range=[
                    0,
                    100
                ],
                height=420,
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            c1, c2, c3, c4 = st.columns(
                4
            )

            c1.metric(
                "Best Match",
                f"{scores[0]:.1f}%"
            )

            c2.metric(
                "5th Match",
                f"{scores[4]:.1f}%"
            )

            c3.metric(
                "Average Top-5",
                f"{np.mean(scores):.1f}%"
            )

            c4.metric(
                "Inference",
                f"{inference_time:.1f} ms"
            )


            # ------------------------------------------------
            # Similarity distribution
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">📊 Gallery Similarity Distribution</div>',
                unsafe_allow_html=True
            )

            distribution_fig = go.Figure()

            distribution_fig.add_trace(
                go.Histogram(
                    x=similarities * 100,
                    nbinsx=20
                )
            )

            distribution_fig.update_layout(
                title="Similarity Across Gallery",
                xaxis_title="Cosine Similarity (%)",
                yaxis_title="Number of Images",
                height=400,
                template="plotly_white"
            )

            st.plotly_chart(
                distribution_fig,
                use_container_width=True
            )


            # ------------------------------------------------
            # AI explanation
            # ------------------------------------------------

            st.markdown(
                """
                <div class="info-box">

                <h3>🧠 Why these matches?</h3>

                The model compares visual embeddings rather than
                relying only on raw pixel color.

                <br><br>

                ✓ Pattern structure<br>
                ✓ Motif arrangement<br>
                ✓ Visual texture<br>
                ✓ Spatial composition<br>
                ✓ Color-invariant visual representation

                <br><br>

                <b>Goal:</b> recognize similar saree designs
                even when their colors change.

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# PAGE 2 — VERIFY DESIGN
# ============================================================

elif page == "🔐 Verify Design":

    st.markdown(
        '<div class="main-title">🔐 Design Verification</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Compare two saree images and determine whether they
        represent the same underlying design.
        </div>
        """,
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(
        2
    )


    with col1:

        file1 = st.file_uploader(
            "Upload Image A",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ],
            key="verify_a"
        )


    with col2:

        file2 = st.file_uploader(
            "Upload Image B",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ],
            key="verify_b"
        )


    if file1 and file2:

        image1 = Image.open(
            file1
        ).convert("RGB")

        image2 = Image.open(
            file2
        ).convert("RGB")


        col1, col2 = st.columns(
            2
        )

        with col1:

            st.image(
                image1,
                caption="Image A",
                use_container_width=True
            )

        with col2:

            st.image(
                image2,
                caption="Image B",
                use_container_width=True
            )


        if st.button(
            "🔍 Compare Designs",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "🧠 Comparing visual designs..."
            ):

                emb1 = get_embedding(
                    image1,
                    model
                )

                emb2 = get_embedding(
                    image2,
                    model
                )

                similarity = cosine_similarity(
                    emb1.reshape(
                        1,
                        -1
                    ),
                    emb2.reshape(
                        1,
                        -1
                    )
                )[0][0]


            similarity_percent = (
                similarity * 100
            )


            st.divider()


            if similarity >= THRESHOLD:

                st.success(
                    f"✅ SAME / HIGHLY SIMILAR DESIGN — "
                    f"{similarity_percent:.2f}% similarity"
                )

                result = "SAME DESIGN"

            else:

                st.warning(
                    f"⚠️ DIFFERENT DESIGN — "
                    f"{similarity_percent:.2f}% similarity"
                )

                result = "DIFFERENT DESIGN"


            c1, c2, c3 = st.columns(
                3
            )

            c1.metric(
                "Similarity",
                f"{similarity_percent:.2f}%"
            )

            c2.metric(
                "Threshold",
                f"{THRESHOLD * 100:.2f}%"
            )

            c3.metric(
                "Decision",
                result
            )


            # Gauge-style bar

            verify_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=similarity_percent,
                    title={
                        "text":
                        "Design Similarity"
                    },
                    gauge={
                        "axis": {
                            "range": [
                                0,
                                100
                            ]
                        },
                        "threshold": {
                            "line": {
                                "width": 4
                            },
                            "value":
                            THRESHOLD * 100
                        }
                    }
                )
            )

            verify_fig.update_layout(
                height=350
            )

            st.plotly_chart(
                verify_fig,
                use_container_width=True
            )


# ============================================================
# PAGE 3 — ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="main-title">📊 Model Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Performance and evaluation results from the
        DeepLure saree design retrieval system.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(
        4
    )


    with col1:

        st.markdown(
            """
            <div class="metric-card">

            <div class="metric-title">
            Top-1 Accuracy
            </div>

            <div class="metric-value">
            100%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="metric-card">

            <div class="metric-title">
            Top-5 Accuracy
            </div>

            <div class="metric-value">
            100%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="metric-card">

            <div class="metric-title">
            ROC-AUC
            </div>

            <div class="metric-value">
            1.00
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            """
            <div class="metric-card">

            <div class="metric-title">
            F1 Score
            </div>

            <div class="metric-value">
            1.00
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # SYSTEM INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "🧠 Model Information"
    )

    info1, info2, info3, info4 = st.columns(
        4
    )

    info1.metric(
        "Gallery Images",
        len(gallery_df)
    )

    info2.metric(
        "Embedding Size",
        "128-D"
    )

    info3.metric(
        "Architecture",
        "ResNet18"
    )

    info4.metric(
        "Threshold",
        f"{THRESHOLD:.4f}"
    )


    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📉 Training History</div>',
        unsafe_allow_html=True
    )


    # Values obtained during your 10-epoch training
    train_loss = [
        0.2339,
        0.1904,
        0.1933,
        0.1798,
        0.1949,
        0.1622,
        0.1683,
        0.1369,
        0.1523,
        0.1520
    ]

    val_loss = [
        0.0864,
        0.0524,
        0.0543,
        0.0789,
        0.0405,
        0.0895,
        0.0753,
        0.0646,
        0.0601,
        0.1073
    ]

    epochs = list(
        range(
            1,
            11
        )
    )


    loss_fig = go.Figure()


    loss_fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_loss,
            mode="lines+markers",
            name="Training Loss"
        )
    )


    loss_fig.add_trace(
        go.Scatter(
            x=epochs,
            y=val_loss,
            mode="lines+markers",
            name="Validation Loss"
        )
    )


    loss_fig.update_layout(
        title="Training vs Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Contrastive Loss",
        height=450,
        template="plotly_white"
    )


    st.plotly_chart(
        loss_fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # RETRIEVAL PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🎯 Retrieval Performance</div>',
        unsafe_allow_html=True
    )


    retrieval_df = pd.DataFrame({

        "Metric": [
            "Top-1 Accuracy",
            "Top-5 Accuracy",
            "ROC-AUC",
            "Verification Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Score": [
            100,
            100,
            100,
            100,
            100,
            100,
            100
        ]

    })


    retrieval_fig = go.Figure()


    retrieval_fig.add_trace(
        go.Bar(
            x=retrieval_df["Metric"],
            y=retrieval_df["Score"],
            text=[
                f"{x:.0f}%"
                for x in retrieval_df["Score"]
            ],
            textposition="auto"
        )
    )


    retrieval_fig.update_layout(
        title="Model Evaluation Metrics",
        yaxis_title="Score (%)",
        yaxis_range=[
            0,
            110
        ],
        height=450,
        template="plotly_white"
    )


    st.plotly_chart(
        retrieval_fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CROSS COLOR INSIGHT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🎨 Color-Invariance Evaluation</div>',
        unsafe_allow_html=True
    )


    color_col1, color_col2 = st.columns(
        2
    )


    with color_col1:

        st.metric(
            "Same Design / Color Changed",
            "98.47% similarity"
        )


    with color_col2:

        st.metric(
            "Different Images",
            "78.15% similarity"
        )


    color_fig = go.Figure()


    color_fig.add_trace(
        go.Bar(
            x=[
                "Same Design\nColor Changed",
                "Different Images"
            ],

            y=[
                98.47,
                78.15
            ],

            text=[
                "98.47%",
                "78.15%"
            ],

            textposition="auto"
        )
    )


    color_fig.update_layout(
        title="Color-Invariance Evaluation",
        yaxis_title="Average Similarity (%)",
        yaxis_range=[
            0,
            100
        ],
        height=400,
        template="plotly_white"
    )


    st.plotly_chart(
        color_fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # FINAL INSIGHT
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="info-box">

        <h3>💡 Key Finding</h3>

        The model produces substantially higher similarity
        for the <b>same design under color transformation</b>
        than for unrelated images.

        <br><br>

        This supports the core objective of DeepLure:

        <br><br>

        <b>
        Identify the underlying saree design rather than
        simply matching its color.
        </b>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DeepLure • Color-Invariant Saree Design Intelligence • AI Engineering Assessment"
)