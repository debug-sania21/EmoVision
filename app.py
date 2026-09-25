import streamlit as st
import time
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import os
import pickle
import tempfile

# ============================================================
# VERSION CHECK – st.html requires Streamlit >= 1.33.0
# ============================================================
try:
    # Check if st.html exists and is callable
    if not hasattr(st, "html") or not callable(st.html):
        st.error(
            "Your Streamlit version does not support `st.html`. "
            "Please upgrade to version 1.33.0 or later using:\n"
            "pip install --upgrade streamlit"
        )
        st.stop()
except Exception:
    # Fallback: if the check itself fails, assume old version
    st.error(
        "Your Streamlit version does not support `st.html`. "
        "Please upgrade to version 1.33.0 or later."
    )
    st.stop()

# Optional libraries
try:
    import librosa
except ImportError:
    librosa = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multimodal Emotion AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS - now using st.html()
# ============================================================

st.html("""
<style>
    /* Main background and font */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* Title styling */
    .main-title {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: 2px;
        font-family: 'Segoe UI', sans-serif;
        text-shadow: 0 0 40px rgba(255, 210, 0, 0.2);
    }

    .sub-title {
        text-align: center;
        color: #c0c0ff;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-family: 'Segoe UI', sans-serif;
        letter-spacing: 4px;
        opacity: 0.8;
    }

    /* Card style for input sections */
    .input-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 1.8rem 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.3s ease;
    }

    .input-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5);
        border-color: rgba(255, 210, 0, 0.3);
    }

    /* How-to card */
    .howto-card {
        background: linear-gradient(
            135deg,
            rgba(255, 215, 0, 0.08),
            rgba(255, 215, 0, 0.02)
        );
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 215, 0, 0.2);
        border-radius: 20px;
        padding: 1.8rem 2rem;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 8px 32px rgba(255, 215, 0, 0.08);
        transition: all 0.3s ease;
    }

    .howto-card:hover {
        border-color: rgba(255, 215, 0, 0.4);
        box-shadow: 0 12px 48px rgba(255, 215, 0, 0.12);
    }

    .howto-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #ffd700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .howto-title span {
        font-size: 2rem;
    }

    .howto-steps {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 0.5rem;
    }

    .howto-step {
        background: rgba(255, 255, 255, 0.03);
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        border-left: 3px solid #ffd700;
        color: #d0d0ff;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .howto-step .step-num {
        background: #ffd700;
        color: #1a1a2e;
        font-weight: 700;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        flex-shrink: 0;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffd700;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 10px;
        border-bottom: 1px solid rgba(255, 215, 0, 0.2);
        padding-bottom: 0.5rem;
    }

    .section-header span {
        font-size: 1.8rem;
    }

    /* Custom checkbox styling */
    .stCheckbox label {
        color: #e0e0ff !important;
        font-weight: 500;
        font-size: 1.05rem;
    }

    .stCheckbox label span {
        background-color: rgba(255, 215, 0, 0.1) !important;
        border-radius: 6px;
        padding: 0 8px;
    }

    /* File uploader styling */
    .stFileUploader label {
        color: #d0d0ff !important;
        font-weight: 400;
    }

    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 2px dashed rgba(255, 215, 0, 0.3) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        transition: all 0.3s ease;
    }

    .stFileUploader > div:hover {
        border-color: #ffd700 !important;
        background: rgba(255, 215, 0, 0.05) !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #f7971e, #ffd200) !important;
        color: #1a1a2e !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        padding: 0.6rem 2.8rem !important;
        border: none !important;
        border-radius: 60px !important;
        box-shadow: 0 8px 24px rgba(255, 210, 0, 0.3) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 1px;
        width: 100%;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 36px rgba(255, 210, 0, 0.5) !important;
        background: linear-gradient(135deg, #ffa825, #ffd700) !important;
    }

    .stButton > button:active {
        transform: scale(0.96);
    }

    /* Prediction result box */
    .result-box {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 215, 0, 0.25);
        border-radius: 24px;
        padding: 2rem 1.5rem;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4);
    }

    .result-emotion {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.2rem 0;
    }

    .result-confidence {
        font-size: 1.6rem;
        color: #c0c0ff;
        font-weight: 300;
    }

    .result-detail {
        color: #a0a0d0;
        font-size: 1rem;
        margin-top: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        padding-top: 0.8rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(20, 18, 50, 0.85) !important;
        backdrop-filter: blur(8px);
        border-right: 1px solid rgba(255, 215, 0, 0.08);
    }

    .css-1d391kg .stSelectbox label {
        color: #d0d0ff !important;
    }

    /* Status indicators */
    .status-active {
        color: #4caf50;
        font-weight: 500;
    }

    .status-inactive {
        color: #f44336;
        font-weight: 500;
    }

    /* File info text */
    .file-info {
        color: #b0b0e0;
        font-size: 0.9rem;
        background: rgba(255, 255, 255, 0.04);
        padding: 0.4rem 1rem;
        border-radius: 30px;
        display: inline-block;
        margin-top: 0.3rem;
    }

    /* Modal selection badge */
    .modal-badge {
        display: inline-block;
        background: rgba(255, 215, 0, 0.12);
        padding: 0.2rem 1.2rem;
        border-radius: 40px;
        color: #ffd700;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid rgba(255, 215, 0, 0.15);
        margin-right: 0.5rem;
    }

    /* Chart container */
    .chart-container {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 16px;
        padding: 1rem;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Responsive */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.8rem;
        }

        .result-emotion {
            font-size: 2.8rem;
        }

        .howto-steps {
            grid-template-columns: 1fr;
        }
    }
</style>
""")


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "face",
    "face_model_v5_1.keras"
)

AUDIO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "audio",
    "audio_model_v1.keras"
)

TEXT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "text",
    "text_model_v3_svm.pkl"
)


# ============================================================
# EMOTION DEFINITIONS
# ============================================================

EMOTIONS = [
    "Joy",
    "Sadness",
    "Anger",
    "Fear",
    "Surprise",
    "Neutral",
    "Disgust"
]

EMOTION_EMOJIS = {
    "Joy": "😊",
    "Sadness": "😢",
    "Anger": "😡",
    "Fear": "😨",
    "Surprise": "😲",
    "Neutral": "😐",
    "Disgust": "🤢"
}


# ============================================================
# LABEL CONVERSION
# ============================================================

LABEL_MAP = {
    "happy": "Joy",
    "joy": "Joy",
    "sad": "Sadness",
    "sadness": "Sadness",
    "angry": "Anger",
    "anger": "Anger",
    "fear": "Fear",
    "surprise": "Surprise",
    "neutral": "Neutral",
    "disgust": "Disgust"
}


def normalize_emotion_label(label):
    """
    Converts model labels to the UI's emotion names.
    """
    if isinstance(label, np.ndarray):
        label = label.item()
    label = str(label).strip().lower()
    return LABEL_MAP.get(label, str(label).capitalize())


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_face_model():
    if tf is None:
        raise ImportError(
            "TensorFlow is not installed. Install it using: "
            "pip install tensorflow"
        )
    if not os.path.exists(FACE_MODEL_PATH):
        raise FileNotFoundError(
            f"Face model not found: {FACE_MODEL_PATH}"
        )
    return tf.keras.models.load_model(
        FACE_MODEL_PATH,
        compile=False
    )


@st.cache_resource
def load_audio_model():
    if tf is None:
        raise ImportError(
            "TensorFlow is not installed. Install it using: "
            "pip install tensorflow"
        )
    if not os.path.exists(AUDIO_MODEL_PATH):
        raise FileNotFoundError(
            f"Audio model not found: {AUDIO_MODEL_PATH}"
        )
    return tf.keras.models.load_model(
        AUDIO_MODEL_PATH,
        compile=False
    )


@st.cache_resource
def load_text_model():
    if not os.path.exists(TEXT_MODEL_PATH):
        raise FileNotFoundError(
            f"Text model not found: {TEXT_MODEL_PATH}"
        )
    with open(TEXT_MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ============================================================
# SIDEBAR - Configuration & Status
# ============================================================

# with st.sidebar:

#     st.markdown("## ⚙️ System Status")
#     st.markdown("---")

#     st.markdown("**🧠 Model:** Multimodal Fusion v2.1")
#     st.markdown("**📊 Modalities:** Text · Image · Audio")

#     st.markdown("---")

#     # Actual model status
#     face_exists = os.path.exists(FACE_MODEL_PATH)
#     audio_exists = os.path.exists(AUDIO_MODEL_PATH)
#     text_exists = os.path.exists(TEXT_MODEL_PATH)

#     col1, col2, col3 = st.columns(3)

#     if text_exists:
#         col1.markdown("🟢 **Text**")
#     else:
#         col1.markdown("🔴 **Text**")

#     if face_exists:
#         col2.markdown("🟢 **Image**")
#     else:
#         col2.markdown("🔴 **Image**")

#     if audio_exists:
#         col3.markdown("🟢 **Audio**")
#     else:
#         col3.markdown("🔴 **Audio**")

#     st.markdown("---")

#     st.markdown("### 🎯 Supported Emotions")

#     st.markdown("""
#     😊 Joy · 😢 Sadness · 😡 Anger · 😨 Fear · 😲 Surprise · 😐 Neutral · 🤢 Disgust
#     """)

#     st.markdown("---")

#     st.markdown(
#         "🔬 *Fusion weights: Text 40% · Image 35% · Audio 25%*"
#     )

#     st.markdown("---")

#     st.markdown(
#         "💡 *Tip: Try different combinations of modalities to see how the fusion adapts.*"
#     )


# ============================================================
# MAIN CONTENT
# ============================================================

st.html("""
<div class="main-title">🎭 Multimodal Emotion AI</div>
""")

st.html("""
<div class="sub-title">✨ fuse text · image · audio for deeper emotional insight</div>
""")


# ============================================================
# HOW TO USE CARD
# ============================================================

st.html("""
<div class="howto-card">

    <div class="howto-title">
        <span>📖</span> How to Use
    </div>

    <div class="howto-steps">

        <div class="howto-step">
            <span class="step-num">1</span>
            <span>
                <strong>Select Modalities</strong><br>
                Check the boxes for Text, Image, or Audio
            </span>
        </div>

        <div class="howto-step">
            <span class="step-num">2</span>
            <span>
                <strong>Provide Input</strong><br>
                Type text, upload an image, or upload audio
            </span>
        </div>

        <div class="howto-step">
            <span class="step-num">3</span>
            <span>
                <strong>Predict</strong><br>
                Click the
                <span style="color:#ffd700;">
                    🚀 Predict Emotion
                </span>
                button
            </span>
        </div>

        <div class="howto-step">
            <span class="step-num">4</span>
            <span>
                <strong>View Results</strong><br>
                See the fused emotion prediction with confidence
            </span>
        </div>

    </div>

    <div style="
        margin-top: 0.8rem;
        color: #8888bb;
        font-size: 0.9rem;
        text-align: center;
        border-top: 1px solid rgba(255,215,0,0.08);
        padding-top: 0.8rem;
    ">
        ⚡ You can use 1, 2, or all 3 modalities simultaneously —
        the system will fuse them intelligently
    </div>

</div>
""")


# ============================================================
# MODALITY SELECTION & INPUTS
# ============================================================

col1, col2, col3 = st.columns(3, gap="large")


# ============================================================
# TEXT MODALITY
# ============================================================

with col1:

    with st.container():

        st.html("""
        <div class="input-card">
            <div class="section-header">
                <span>📝</span> Text
            </div>
        """)

        text_enabled = st.checkbox(
            "Enable Text",
            value=True,
            key="text_enabled"
        )

        if text_enabled:

            text_input = st.text_area(
                "Enter text:",
                placeholder="e.g., I feel so happy today!",
                height=120,
                key="text_area"
            )

            if text_input:

                st.html(
                    f'<div class="file-info">📄 '
                    f'{len(text_input)} characters</div>'
                )

        else:

            text_input = ""

            st.info(
                "Text modality disabled.",
                icon="⏸️"
            )

        st.html("</div>")


# ============================================================
# IMAGE MODALITY
# ============================================================

with col2:

    with st.container():

        st.html("""
        <div class="input-card">
            <div class="section-header">
                <span>🖼️</span> Image
            </div>
        """)

        image_enabled = st.checkbox(
            "Enable Image",
            value=True,
            key="image_enabled"
        )

        if image_enabled:
        
            image_source = st.radio(
                "Image input:",
                ["Upload Image", "Take Live Photo"],
                horizontal=True,
                key="image_source"
            )
        
            image_file = None
        
            if image_source == "Upload Image":
        
                image_file = st.file_uploader(
                    "Upload image:",
                    type=[
                        "jpg",
                        "jpeg",
                        "png",
                        "bmp",
                        "webp"
                    ],
                    key="image_upload"
                )
        
                if image_file:
                    st.html(
                        f'<div class="file-info">📸 '
                        f'{image_file.name}</div>'
                    )
        
                    st.image(
                        image_file,
                        width=180,
                        caption="Preview"
                    )
        
            else:
        
                image_file = st.camera_input(
                    "Take a picture using your webcam:",
                    key="camera_input"
                )
        
        else:
            image_file = None
            st.info(
                "Image modality disabled.",
                icon="⏸️"
            )

        st.html("</div>")


# ============================================================
# AUDIO MODALITY
# ============================================================

with col3:

    with st.container():

        st.html("""
        <div class="input-card">
            <div class="section-header">
                <span>🎵</span> Audio
            </div>
        """)

        audio_enabled = st.checkbox(
            "Enable Audio",
            value=True,
            key="audio_enabled"
        )

        if audio_enabled:

            audio_source = st.radio(
                "Audio input:",
                ["Upload Audio", "Record Live Audio"],
                horizontal=True,
                key="audio_source"
            )

            audio_file = None

            # ------------------------------------------------
            # Upload existing audio file
            # ------------------------------------------------

            if audio_source == "Upload Audio":

                audio_file = st.file_uploader(
                    "Upload audio:",
                    type=[
                        "wav",
                        "mp3",
                        "flac",
                        "ogg",
                        "m4a"
                    ],
                    key="audio_upload"
                )

                if audio_file:

                    st.html(
                        f'<div class="file-info">🎧 '
                        f'{audio_file.name}</div>'
                    )

                    st.audio(
                        audio_file
                    )

            # ------------------------------------------------
            # Record audio directly from microphone
            # ------------------------------------------------

            else:

                audio_file = st.audio_input(
                    "Record your voice:",
                    key="audio_recorder"
                )

                if audio_file:

                    st.html(
                        '<div class="file-info">🎙️ '
                        'Live audio recorded</div>'
                    )

                    st.audio(
                        audio_file
                    )

        else:

            audio_file = None

            st.info(
                "Audio modality disabled.",
                icon="⏸️"
            )

        st.html("</div>")

# ============================================================
# MODEL PREPROCESSING
# ============================================================


def preprocess_face(image_file, model):

    """
    Standard FER-style preprocessing.

    Converts image to grayscale and resizes according
    to the model input dimensions.

    If the trained model expects RGB or another shape,
    this function can be adjusted without changing UI.
    """

    if Image is None:
        raise ImportError(
            "Pillow is not installed. "
            "Install it using: pip install pillow"
        )

    image = Image.open(image_file)

    # Convert to grayscale
    image = image.convert("L")

    # Determine expected shape
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1]
    width = input_shape[2]

    if height is None:
        height = 48

    if width is None:
        width = 48

    image = image.resize(
        (int(width), int(height))
    )

    arr = np.asarray(
        image,
        dtype=np.float32
    )

    # FER2013 standard normalization
    arr = arr / 255.0

    # Add channel dimension
    arr = np.expand_dims(
        arr,
        axis=-1
    )

    # Add batch dimension
    arr = np.expand_dims(
        arr,
        axis=0
    )

    return arr


def preprocess_audio(audio_file, model):

    """
    Standard MFCC based preprocessing.

    Uses:
        sample rate = 22050
        MFCC = 40
        fixed sequence length = 174

    This section is isolated so it can easily be
    replaced if the audio model was trained differently.
    """

    if librosa is None:
        raise ImportError(
            "librosa is not installed. "
            "Install it using: pip install librosa"
        )

    audio_bytes = audio_file.getvalue()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(audio_file.name)[1]
    ) as temp:

        temp.write(audio_bytes)
        temp_path = temp.name

    try:

        signal, sample_rate = librosa.load(
            temp_path,
            sr=22050,
            mono=True
        )

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)

    # MFCC extraction
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=40
    )

    # Normalize
    mfcc = (
        mfcc - np.mean(mfcc)
    ) / (
        np.std(mfcc) + 1e-8
    )

    # Determine model input
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    # Common possibilities:
    # (None, time, features)
    # (None, features, time)
    # (None, time, features, 1)

    if len(input_shape) == 3:

        expected_dim_1 = input_shape[1]
        expected_dim_2 = input_shape[2]

        # Common RNN arrangement
        if expected_dim_2 == 40:

            target_time = (
                int(expected_dim_1)
                if expected_dim_1 is not None
                else 174
            )

            if mfcc.shape[1] < target_time:

                pad_width = (
                    target_time - mfcc.shape[1]
                )

                mfcc = np.pad(
                    mfcc,
                    ((0, 0), (0, pad_width)),
                    mode="constant"
                )

            else:

                mfcc = mfcc[:, :target_time]

            features = mfcc.T

        else:

            target_time = (
                int(expected_dim_1)
                if expected_dim_1 is not None
                else 174
            )

            features = mfcc

            if features.shape[1] < target_time:

                pad_width = (
                    target_time - features.shape[1]
                )

                features = np.pad(
                    features,
                    ((0, 0), (0, pad_width)),
                    mode="constant"
                )

            else:

                features = features[:, :target_time]

            features = features.T

        return np.expand_dims(
            features,
            axis=0
        )

    elif len(input_shape) == 4:

        target_height = (
            int(input_shape[1])
            if input_shape[1] is not None
            else 40
        )

        target_width = (
            int(input_shape[2])
            if input_shape[2] is not None
            else 174
        )

        mfcc_resized = np.resize(
            mfcc,
            (target_height, target_width)
        )

        mfcc_resized = np.expand_dims(
            mfcc_resized,
            axis=-1
        )

        return np.expand_dims(
            mfcc_resized,
            axis=0
        )

    else:

        raise ValueError(
            f"Unsupported audio model input shape: {input_shape}"
        )


# ============================================================
# PREDICTION HELPERS
# ============================================================


def convert_prediction_to_probabilities(
    prediction,
    model=None
):

    """
    Converts model output into a probability vector.
    """

    prediction = np.asarray(
        prediction
    )

    prediction = np.squeeze(
        prediction
    )

    # Binary output
    if prediction.ndim == 0:

        value = float(prediction)

        value = max(
            0.0,
            min(1.0, value)
        )

        return np.array(
            [1 - value, value]
        )

    # Single probability
    if prediction.ndim == 1:

        values = prediction.astype(
            np.float32
        )

    else:

        values = prediction.flatten().astype(
            np.float32
        )

    # Already probabilities
    if (
        np.all(values >= 0)
        and np.all(values <= 1)
        and abs(np.sum(values) - 1.0) < 0.05
    ):
        return values

    # Softmax
    values = values - np.max(values)

    exp_values = np.exp(values)

    total = np.sum(exp_values)

    if total == 0:
        return np.ones_like(values) / len(values)

    return exp_values / total


def make_emotion_distribution(probabilities):

    """
    Maps model output positions to UI emotions.

    FER2013 conventional order:
        angry
        disgust
        fear
        happy
        sad
        surprise
        neutral
    """

    distribution = {
        emotion: 0.0
        for emotion in EMOTIONS
    }

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    ).flatten()

    conventional_labels = [
        "Anger",
        "Disgust",
        "Fear",
        "Joy",
        "Sadness",
        "Surprise",
        "Neutral"
    ]

    count = min(
        len(probabilities),
        len(conventional_labels)
    )

    for i in range(count):

        emotion = conventional_labels[i]

        distribution[emotion] = float(
            probabilities[i]
        )

    # Normalize
    total = sum(
        distribution.values()
    )

    if total > 0:

        distribution = {
            k: v / total
            for k, v in distribution.items()
        }

    return distribution


# ============================================================
# TEXT MODEL PREDICTION
# ============================================================


def predict_text(text):
    model_data = load_text_model()

    if not isinstance(model_data, dict):
        raise ValueError(
            "Text model does not contain the expected dictionary format."
        )

    classifier = model_data["model"]
    vectorizer = model_data["vectorizer"]
    classes = model_data["classes"]

    # Convert text using the SAME TF-IDF vectorizer used during training
    text_features = vectorizer.transform([text])

    # LinearSVC uses decision_function instead of predict_proba
    decision_scores = classifier.decision_function(text_features)[0]

    # Convert decision scores into normalized confidence values
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores / np.sum(exp_scores)

    # FER/text model labels → UI labels
    emotion_mapping = {
        "happy": "Joy",
        "sad": "Sadness",
        "angry": "Anger",
        "fear": "Fear",
        "surprise": "Surprise",
        "neutral": "Neutral",
        "disgust": "Disgust"
    }

    # Create UI emotion distribution
    distribution = {}

    for class_name, probability in zip(classes, probabilities):
        ui_emotion = emotion_mapping.get(
            class_name.lower(),
            class_name.capitalize()
        )

        distribution[ui_emotion] = float(probability)

    return distribution    # --------------------------------------------------------
    # Case 2:
    # Pipeline with decision_function
    # --------------------------------------------------------

    if hasattr(
        model,
        "decision_function"
    ):

        try:

            scores = model.decision_function(
                [text]
            )

            scores = np.asarray(
                scores
            ).flatten()

            probabilities = convert_prediction_to_probabilities(
                scores
            )

            if hasattr(
                model,
                "classes_"
            ):

                classes = model.classes_

                distribution = {
                    emotion: 0.0
                    for emotion in EMOTIONS
                }

                for label, probability in zip(
                    classes,
                    probabilities
                ):

                    emotion = normalize_emotion_label(
                        label
                    )

                    if emotion in distribution:

                        distribution[emotion] = float(
                            probability
                        )

                total = sum(
                    distribution.values()
                )

                if total > 0:

                    distribution = {
                        k: v / total
                        for k, v in distribution.items()
                    }

                return distribution

        except Exception:
            pass

    # --------------------------------------------------------
    # Case 3:
    # Raw SVM + separately saved vectorizer
    # --------------------------------------------------------

    if hasattr(model, "predict"):

        raise ValueError(
            "The text SVM requires its original text "
            "vectorizer/TF-IDF preprocessing. "
            "The .pkl appears to contain the classifier "
            "without a compatible vectorizer."
        )

    raise ValueError(
        "Unsupported text model format."
    )


# ============================================================
# FACE MODEL PREDICTION
# ============================================================


def predict_face(image_file):

    model = load_face_model()

    processed_image = preprocess_face(
        image_file,
        model
    )

    raw_prediction = model.predict(
        processed_image,
        verbose=0
    )

    probabilities = convert_prediction_to_probabilities(
        raw_prediction,
        model
    )

    return make_emotion_distribution(
        probabilities
    )


# ============================================================
# AUDIO MODEL PREDICTION
# ============================================================


def predict_audio(audio_file):

    model = load_audio_model()

    processed_audio = preprocess_audio(
        audio_file,
        model
    )

    raw_prediction = model.predict(
        processed_audio,
        verbose=0
    )

    probabilities = convert_prediction_to_probabilities(
        raw_prediction,
        model
    )

    return make_emotion_distribution(
        probabilities
    )


# ============================================================
# CONFIDENCE-BASED MULTIMODAL FUSION
# ============================================================


BASE_WEIGHTS = {
    "text": 0.40,
    "image": 0.35,
    "audio": 0.25
}


def confidence_based_fusion(
    modality_predictions
):

    """
    Adaptive weighted voting.

    Each modality contributes:

        base_weight × model_confidence

    The effective weights are then normalized.

    This means a highly confident model has more
    influence than a low-confidence model.
    """

    if not modality_predictions:

        return (
            None,
            0.0,
            {
                emotion: 0.0
                for emotion in EMOTIONS
            },
            {}
        )

    effective_weights = {}

    for modality, distribution in modality_predictions.items():

        confidence = max(
            distribution.values()
        )

        base_weight = BASE_WEIGHTS.get(
            modality,
            1.0
        )

        effective_weights[modality] = (
            base_weight * confidence
        )

    total_weight = sum(
        effective_weights.values()
    )

    if total_weight <= 0:

        total_weight = len(
            effective_weights
        )

        effective_weights = {
            modality: 1.0
            for modality in effective_weights
        }

    normalized_weights = {
        modality: weight / total_weight
        for modality, weight in effective_weights.items()
    }

    fused = {
        emotion: 0.0
        for emotion in EMOTIONS
    }

    for modality, distribution in modality_predictions.items():

        weight = normalized_weights[
            modality
        ]

        for emotion in EMOTIONS:

            fused[emotion] += (
                weight *
                distribution.get(
                    emotion,
                    0.0
                )
            )

    total = sum(
        fused.values()
    )

    if total > 0:

        fused = {
            emotion: value / total
            for emotion, value in fused.items()
        }

    top_emotion = max(
        fused,
        key=fused.get
    )

    confidence = (
        fused[top_emotion] * 100
    )

    return (
        top_emotion,
        confidence,
        fused,
        normalized_weights
    )


# ============================================================
# FUNCTION TO CREATE PIE CHART
# ============================================================


def create_emotion_pie_chart(
    emotions,
    emotion_emojis
):

    filtered_emotions = {
        emotion: value
        for emotion, value in emotions.items()
        if value > 0.01
    }

    labels = [
        f"{emotion_emojis.get(emotion, '')} {emotion}"
        for emotion in filtered_emotions.keys()
    ]

    values = list(
        filtered_emotions.values()
    )

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.58,
            textinfo="label+percent",
            textfont=dict(size=15),
            hovertemplate=(
                "<b>%{label}</b>"
                "<br>Confidence: %{percent}"
                "<extra></extra>"
            ),
            marker=dict(
                line=dict(width=2)
            ),
            pull=[
                0.04
            ] + [
                0
            ] * (
                len(values) - 1
            )
        )
    )

    fig.update_layout(
        height=560,
        margin=dict(
            l=20,
            r=20,
            t=80,
            b=20
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(size=14)
        ),
        hoverlabel=dict(
            bgcolor="#111111",
            bordercolor="#FFD700",
            font=dict(size=15)
        ),
        annotations=[
            dict(
                text="<b>Emotion<br>Distribution</b>",
                x=0.5,
                y=0.5,
                font=dict(size=17),
                showarrow=False
            )
        ]
    )

    return fig


# ============================================================
# PREDICTION BUTTON
# ============================================================

col_btn1, col_btn2, col_btn3 = st.columns(
    [1, 2, 1]
)

with col_btn2:

    predict_clicked = st.button(
        "🚀 Predict Emotion",
        use_container_width=True
    )


# ============================================================
# REAL MULTIMODAL PREDICTION
# ============================================================


def run_multimodal_prediction(
    text,
    image,
    audio,
    text_on,
    img_on,
    aud_on
):

    active_inputs = []

    if text_on and text and text.strip():

        active_inputs.append(
            "text"
        )

    if img_on and image is not None:

        active_inputs.append(
            "image"
        )

    if aud_on and audio is not None:

        active_inputs.append(
            "audio"
        )

    if not active_inputs:

        return {
            "error":
                "No active modalities with input. "
                "Please enable and provide at least one input."
        }

    modality_predictions = {}

    errors = {}

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    if "text" in active_inputs:

        try:

            modality_predictions[
                "text"
            ] = predict_text(
                text
            )

        except Exception as e:

            errors[
                "Text"
            ] = str(e)

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if "image" in active_inputs:

        try:

            modality_predictions[
                "image"
            ] = predict_face(
                image
            )

        except Exception as e:

            errors[
                "Image"
            ] = str(e)

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    if "audio" in active_inputs:

        try:

            modality_predictions[
                "audio"
            ] = predict_audio(
                audio
            )

        except Exception as e:

            errors[
                "Audio"
            ] = str(e)

    # --------------------------------------------------------
    # If every model failed
    # --------------------------------------------------------

    if not modality_predictions:

        return {
            "error":
                "All selected models failed.",
            "details":
                errors
        }

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    (
        top_emotion,
        confidence,
        fused_distribution,
        fusion_weights
    ) = confidence_based_fusion(
        modality_predictions
    )

    return {
        "top_emotion": top_emotion,
        "confidence": confidence,
        "emotions": fused_distribution,
        "active_mods": list(
            modality_predictions.keys()
        ),
        "individual_predictions":
            modality_predictions,
        "fusion_weights":
            fusion_weights,
        "errors":
            errors
    }


# ============================================================
# DISPLAY PREDICTION RESULT
# ============================================================

if predict_clicked:

    with st.spinner(
        "🧠 Analyzing multimodal inputs..."
    ):

        result = run_multimodal_prediction(
            text_input,
            image_file,
            audio_file,
            text_enabled,
            image_enabled,
            audio_enabled
        )

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    if "error" in result:

        st.warning(
            result["error"],
            icon="⚠️"
        )

        if result.get("details"):

            for modality, error in result[
                "details"
            ].items():

                st.error(
                    f"{modality} model: {error}"
                )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    else:

        top_emotion = result[
            "top_emotion"
        ]

        confidence = result[
            "confidence"
        ]

        all_emotions = result[
            "emotions"
        ]

        active_mods = result[
            "active_mods"
        ]

        fusion_weights = result[
            "fusion_weights"
        ]

        emotion_emojis = EMOTION_EMOJIS

        # Active modality badge
        badge_text = " + ".join(
            [
                m.capitalize()
                for m in active_mods
            ]
        )

        # ----------------------------------------------------
        # RESULT BOX
        # ----------------------------------------------------

        st.html(
            f"""
            <div class="result-box">

                <div style="
                    display: flex;
                    justify-content: center;
                    gap: 1rem;
                    flex-wrap: wrap;
                    margin-bottom: 0.5rem;
                ">

                    <span class="modal-badge">
                        🧩 {badge_text}
                    </span>

                </div>

                <div style="
                    font-size: 1rem;
                    color: #aaaadd;
                    letter-spacing: 2px;
                ">
                    PREDICTED EMOTION
                </div>

                <div class="result-emotion">
                    {emotion_emojis[top_emotion]}
                    {top_emotion}
                </div>

                <div class="result-confidence">
                    Confidence: {confidence:.1f}%
                </div>

                <div class="result-detail">
                    <span style="color: #ffd700;">
                        Fusion
                    </span>
                    · {len(active_mods)}
                    modality(ies) active
                </div>

            </div>
            """
        )

        # ----------------------------------------------------
        # EMOTION DISTRIBUTION
        # ----------------------------------------------------

        st.markdown("---")

        st.markdown(
            "#### 📊 Emotion Distribution"
        )

        fig = create_emotion_pie_chart(
            all_emotions,
            emotion_emojis
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

        # ----------------------------------------------------
        # FUSION INFORMATION
        # ----------------------------------------------------

        st.markdown("---")

        st.markdown(
            "#### 🔬 Adaptive Fusion Weights"
        )

        fusion_text = " · ".join(
            [
                f"{modality.capitalize()} "
                f"{weight * 100:.1f}%"
                for modality, weight
                in fusion_weights.items()
            ]
        )

        st.caption(
            f"Confidence-adjusted weights: "
            f"{fusion_text}"
        )

        # ----------------------------------------------------
        # INDIVIDUAL MODEL RESULTS
        # ----------------------------------------------------

        st.markdown(
            "#### 🤖 Individual Model Predictions"
        )

        for modality, distribution in result[
            "individual_predictions"
        ].items():

            modality_emotion = max(
                distribution,
                key=distribution.get
            )

            modality_confidence = (
                distribution[
                    modality_emotion
                ] * 100
            )

            st.html(
                f"""
                <div class="file-info">
                    {modality.capitalize()}:
                    {emotion_emojis.get(
                        modality_emotion,
                        "🎭"
                    )}
                    {modality_emotion}
                    ·
                    {modality_confidence:.1f}%
                </div>
                """
            )

        # ----------------------------------------------------
        # MODEL WARNINGS
        # ----------------------------------------------------

        if result.get("errors"):

            st.warning(
                "Some selected modalities could not "
                "be processed. The available successful "
                "modalities were fused.",
                icon="⚠️"
            )

            for modality, error in result[
                "errors"
            ].items():

                st.caption(
                    f"{modality}: {error}"
                )

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        st.caption(
            f"🕒 Prediction time: "
            f"{datetime.now().strftime('%H:%M:%S')} "
            f"· Modalities fused: "
            f"{', '.join(active_mods)}"
        )


else:

    # ========================================================
    # PLACEHOLDER
    # ========================================================

    st.html("""
    <div style="
        text-align: center;
        padding: 2rem 0;
        color: #8888bb;
    ">

        <span style="
            font-size: 3rem;
            opacity: 0.3;
        ">
            🎭
        </span>

        <p style="
            font-size: 1.2rem;
        ">
            Select modalities, provide inputs,
            then click
            <strong style="color: #ffd700;">
                Predict Emotion
            </strong>
        </p>

        <p style="
            font-size: 0.9rem;
            opacity: 0.6;
        ">
            The system will fuse predictions
            from active modalities.
        </p>

    </div>
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.html("""
<div style="
    display: flex;
    justify-content: space-between;
    color: #666699;
    font-size: 0.8rem;
    padding: 0.5rem 0;
">

    <span>
        🧠 Multimodal Emotion Detection v1.0
    </span>

    <span>
        ⚡ Fusion · Text · Image · Audio
    </span>

    <span>
        ✨ Streamlit Frontend
    </span>

</div>
""")
