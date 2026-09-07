# Assignment 5 - Crop Recommendation System using Deep Learning by Streamlit and Flask

import os
import threading
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import requests
import streamlit as st

from flask import Flask, request, jsonify

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATASET_PATH = "Crop_recommendation.csv"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "crop_model.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)

FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000

FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

# PAGE CONFIGURATION
st.set_page_config(
    page_title="CropWise AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown(
    """
<style>

.stApp {
    background-color: #ffffff;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header {
    visibility: hidden;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background-color: #f7faf7;
    border-right: 1px solid #e5e9e5;
}

/* ================= BRAND ================= */

.brand {
    font-size: 26px;
    font-weight: 700;
    color: #2f6b3c;
    margin-bottom: 3px;
}

.brand-subtitle {
    color: #7b857d;
    font-size: 13px;
}

/* ================= MAIN TITLE ================= */

.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #202820;
    letter-spacing: -1px;
    margin-bottom: 4px;
}

.main-subtitle {
    font-size: 16px;
    color: #6b756d;
    margin-bottom: 25px;
}

/* ================= INTRO ================= */

.intro-box {
    background-color: #f5faf6;
    border-left: 4px solid #3f7d4a;
    border-radius: 7px;
    padding: 18px 22px;
    margin-bottom: 28px;
}

.intro-title {
    font-size: 18px;
    font-weight: 650;
    color: #26352a;
    margin-bottom: 7px;
}

.intro-text {
    font-size: 14px;
    line-height: 1.65;
    color: #5f6a62;
}

/* ================= SECTION ================= */

.section-title {
    font-size: 22px;
    font-weight: 650;
    color: #29332c;
    margin-top: 18px;
    margin-bottom: 4px;
}

.section-description {
    color: #737d75;
    font-size: 14px;
    margin-bottom: 18px;
}

/* ================= FORM ================= */

[data-testid="stForm"] {
    border: 1px solid #e3e8e3;
    border-radius: 12px;
    padding: 28px;
    background-color: #ffffff;
}

/* ================= INPUTS ================= */

div[data-baseweb="input"] {
    min-height: 48px;
}

div[data-baseweb="input"] input {
    font-size: 16px;
}

label[data-testid="stWidgetLabel"] p {
    font-size: 14px;
    font-weight: 600;
    color: #3e4741;
}

/* ================= PREDICTION BUTTON ================= */

div[data-testid="stFormSubmitButton"] > button {
    min-height: 56px;
    font-size: 16px;
    font-weight: 700;
    border-radius: 8px;
    margin-top: 12px;
    transition: all 0.2s ease;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-1px);
}

/* ================= METRICS ================= */

div[data-testid="stMetric"] {
    background-color: #fafcfa;
    border: 1px solid #e6eae6;
    border-radius: 8px;
    padding: 14px;
}

/* ================= RESULT ================= */

.result-box {
    background-color: #f5faf6;
    border: 1px solid #d6e6d9;
    border-radius: 12px;
    padding: 28px;
    margin-top: 12px;
}

.result-label {
    font-size: 14px;
    color: #6d776f;
    margin-bottom: 5px;
}

.result-crop {
    font-size: 36px;
    font-weight: 700;
    color: #2f6b3c;
    margin-bottom: 6px;
}

.result-description {
    font-size: 14px;
    color: #6c766e;
}

/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #9aa19b;
    font-size: 12px;
    padding-top: 30px;
    margin-top: 45px;
    border-top: 1px solid #eeeeee;
}

</style>
""",
    unsafe_allow_html=True
)

# GLOBAL MODEL VARIABLES
dl_model = None
data_scaler = None
crop_encoder = None

# FLASK APPLICATION
flask_app = Flask(__name__)

# FLASK HOME ROUTE
@flask_app.route("/", methods=["GET"])
def home():
    return jsonify({
        "application": "CropWise AI",
        "status": "running",
        "model": "Deep Neural Network",
        "endpoint": "/predict"
    })

# FLASK PREDICTION ROUTE
@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({
                "success": False,
                "error": "JSON input is required."
            }), 400
        missing_features = [
            feature
            for feature in FEATURES
            if feature not in data
        ]
        if missing_features:
            return jsonify({
                "success": False,
                "error": "Missing required features.",
                "missing": missing_features
            }), 400

        # Prepare input
        input_data = np.array([[
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"])
        ]])

        # Scale input
        input_scaled = data_scaler.transform(
            input_data
        )

        # Deep Learning predictiON
        probabilities = dl_model.predict(
            input_scaled,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_crop = crop_encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = float(
            probabilities[predicted_index] * 100
        )

        # best 5 predictions
        top_indices = np.argsort(
            probabilities
        )[::-1][:5]

        top_predictions = []

        for index in top_indices:
            crop_name = crop_encoder.inverse_transform(
                [int(index)]
            )[0]

            crop_confidence = float(
                probabilities[index] * 100
            )

            top_predictions.append({
                "crop": crop_name,
                "confidence": round(
                    crop_confidence,
                    2
                )
            })

        return jsonify({
            "success": True,
            "recommended_crop":
                predicted_crop,
            "confidence":
                round(
                    confidence,
                    2
                ),
            "top_5":
                top_predictions
        })


    except Exception as error:
        return jsonify({
            "success": False,
            "error":
                str(error)
        }), 500

# TRAIN DEEP LEARNING MODEL
def train_model():
    global dl_model
    global data_scaler
    global crop_encoder

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"""
Dataset not found.
Please place:
{DATASET_PATH}
in the same folder as app.py.
"""
        )

    # Load dataset
    df = pd.read_csv(
        DATASET_PATH
    )

    # Clean dataset
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.reset_index(
        drop=True
    )
    
    # VALIDATE REQUIRED COLUMNS

    required_columns = FEATURES + ["label"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing columns: "
            + str(missing_columns)
        )

    # INPUT FEATURES
    X = df[FEATURES]

    # TARGET LABEL
    y = df["label"]

    # ENCODE TARGET LABELS
    crop_encoder = LabelEncoder()

    y_encoded = crop_encoder.fit_transform(
        y
    )

    number_of_classes = len(
        crop_encoder.classes_
    )

    # TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    # DATA SCALING
    data_scaler = StandardScaler()
    X_train_scaled = data_scaler.fit_transform(
        X_train
    )
    X_test_scaled = data_scaler.transform(
        X_test
    )

    # DEEP LEARNING MODEL ARCHITECTURE 
    dl_model = Sequential([
        Input(
            shape=(7,)
        ),
        Dense(
            128,
            activation="relu"
        ),
        BatchNormalization(),
        Dropout(
            0.30
        ),
        Dense(
            64,
            activation="relu"
        ),
        BatchNormalization(),
        Dropout(
            0.25
        ),
        Dense(
            32,
            activation="relu"
        ),
        Dropout(
            0.20
        ),
        Dense(
            number_of_classes,
            activation="softmax"
        )
    ])

    # COMPILE MODEL
    dl_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # EARLY STOPPING CALLBACK
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True
    )

    # TRAIN MODEL
    dl_model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.20,
        epochs=100,
        batch_size=32,
        callbacks=[
            early_stopping
        ],
        verbose=0
    )

    # evaluate
    loss, accuracy = dl_model.evaluate(
        X_test_scaled,
        y_test,
        verbose=0
    )

    # create model directory if it doesn't exist
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # save model, scaler, and encoder
    dl_model.save(
        MODEL_PATH
    )
    joblib.dump(
        data_scaler,
        SCALER_PATH
    )
    joblib.dump(
        crop_encoder,
        ENCODER_PATH
    )
    return float(
        accuracy
    )

# load existing model
def load_existing_model():
    global dl_model
    global data_scaler
    global crop_encoder
    dl_model = load_model(
        MODEL_PATH
    )
    data_scaler = joblib.load(
        SCALER_PATH
    )
    crop_encoder = joblib.load(
        ENCODER_PATH
    )

# initialize model
@st.cache_resource
def initialize_model():

    if (
        os.path.exists(MODEL_PATH)
        and
        os.path.exists(SCALER_PATH)
        and
        os.path.exists(ENCODER_PATH)
    ):
        load_existing_model()
        return None, "loaded"

    accuracy = train_model()
    return accuracy, "trained"

# initialize the model and handle exceptions
try:
    model_accuracy, model_status = initialize_model()

except Exception as error:
    st.error(
        "Unable to initialize the Deep Learning model."
    )
    st.exception(error)
    st.stop()

# start Flask API in a separate thread
if "flask_started" not in st.session_state:
    st.session_state.flask_started = False

if not st.session_state.flask_started:
    flask_thread = threading.Thread(
        target=lambda: flask_app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            debug=False,
            use_reloader=False
        ),
        daemon=True
    )

    flask_thread.start()
    st.session_state.flask_started = True

# sidebar content
with st.sidebar:
    st.markdown(
        '<div class="brand">🌱 CropWise AI</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="brand-subtitle">'
        'Intelligent Crop Recommendation'
        '</div>',
        unsafe_allow_html=True
    )
    st.divider()
    st.markdown("### About")
    st.write(
        """
CropWise AI analyzes soil and environmental
conditions and recommends the most suitable
crop using a Deep Neural Network.
"""
    )
    st.divider()
    st.markdown("### Model")
    st.write("🧠 Deep Neural Network")
    st.write("⚙️ StandardScaler")
    st.write("🔤 Label Encoding")
    st.write("🔌 Flask REST API")
    st.write("🖥️ Streamlit Interface")
    st.divider()
    st.markdown("### Input Parameters")
    st.write("• Nitrogen")
    st.write("• Phosphorus")
    st.write("• Potassium")
    st.write("• Temperature")
    st.write("• Humidity")
    st.write("• Soil pH")
    st.write("• Rainfall")
    st.divider()
    st.caption(
        "ADS Deep Learning Project"
    )

# main handler
st.markdown(
    '<div class="main-title">'
    '🌱 Crop Recommendation System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Deep Learning based crop recommendation using '
    'soil and environmental conditions.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="intro-box">
<div class="intro-title">
How does the system work?
</div>
<div class="intro-text">
CropWise AI analyzes seven important field parameters:
<b>Nitrogen, Phosphorus, Potassium, Temperature, Humidity,
Soil pH and Rainfall</b>.
The trained Deep Neural Network learns the relationship
between these conditions and different crops. After you
enter the field values, the model predicts the most
suitable crop and also provides the next best alternatives.
</div>
</div>
""",
    unsafe_allow_html=True
)

# field conditions section
st.markdown(
    '<div class="section-title">'
    '🌾 Enter Field Conditions'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Enter the current soil and environmental measurements '
    'for the field.'
    '</div>',
    unsafe_allow_html=True
)

# form
with st.form(
    "crop_prediction_form"
):

    # soil nutrients section
    st.markdown(
        "#### 🧪 Soil Nutrients"
    )

    st.caption(
        "These parameters represent the nutrient composition "
        "of the soil."
    )

    soil_col1, soil_col2, soil_col3 = st.columns(3)

    with soil_col1:
        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=140.0,
            value=50.0,
            step=1.0,
            help=
            "Nitrogen content in the soil."
        )

    with soil_col2:
        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            max_value=145.0,
            value=50.0,
            step=1.0,
            help=
            "Phosphorus content in the soil."
        )

    with soil_col3:
        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            max_value=205.0,
            value=50.0,
            step=1.0,
            help=
            "Potassium content in the soil."
        )

    st.markdown("")
    st.markdown(
        "#### 🌤️ Environmental Conditions"
    )
    st.caption(
        "These parameters describe the climatic and soil "
        "environment of the field."
    )

    env_col1, env_col2 = st.columns(2)

    with env_col1:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=25.0,
            step=0.1,
            help=
            "Average temperature of the field."
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.1,
            help=
            "Relative humidity of the environment."
        )

    with env_col2:
        soil_ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1,
            help=
            "Soil acidity or alkalinity."
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=200.0,
            step=1.0,
            help=
            "Rainfall received or expected."
        )

    st.markdown("")

    predict_button = st.form_submit_button(
        "🌱   RECOMMEND SUITABLE CROP",
        type="primary",
        use_container_width=True
    )

# predict button handler
if predict_button:

    payload = {
        "N":
            nitrogen,
        "P":
            phosphorus,
        "K":
            potassium,
        "temperature":
            temperature,
        "humidity":
            humidity,
        "ph":
            soil_ph,
        "rainfall":
            rainfall
    }

    try:
        # call the Flask API for prediction
        with st.spinner(
            "Analyzing field conditions with the Deep Learning model..."
        ):

            response = requests.post(
                f"http://{FLASK_HOST}:{FLASK_PORT}/predict",
                json=payload,
                timeout=30
            )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                crop = result[
                    "recommended_crop"
                ]
                confidence = result[
                    "confidence"
                ]

                st.markdown(
                    '<div class="section-title">'
                    '🎯 Recommendation'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
<div class="result-box">
<div class="result-label">
Most suitable crop for the selected field conditions
</div>
<div class="result-crop">
🌱 {crop.title()}
</div>
<div class="result-description">
Based on the seven selected soil and environmental
parameters, the Deep Learning model identifies this
as the most suitable crop.
</div>
</div>
""",
                    unsafe_allow_html=True
                )

                st.markdown("")
                metric1, metric2, metric3 = st.columns(3)
                with metric1:
                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )
                with metric2:
                    st.metric(
                        "Features Used",
                        "7"
                    )
                with metric3:
                    st.metric(
                        "Model",
                        "DNN"
                    )

                # top 5 recommendations
                st.markdown(
                    '<div class="section-title">'
                    '🏆 Top Crop Recommendations'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<div class="section-description">'
                    'The model probabilities for the five most '
                    'suitable crops are shown below.'
                    '</div>',
                    unsafe_allow_html=True
                )
                top5 = result[
                    "top_5"
                ]
                for rank, item in enumerate(
                    top5,
                    start=1
                ):
                    crop_name = item[
                        "crop"
                    ]
                    crop_confidence = item[
                        "confidence"
                    ]
                    col_rank, col_crop, col_value = st.columns(
                        [0.5, 4, 1]
                    )
                    with col_rank:
                        st.markdown(
                            f"**{rank}**"
                        )
                    with col_crop:
                        st.markdown(
                            f"**{crop_name.title()}**"
                        )
                        st.progress(
                            min(
                                crop_confidence / 100,
                                1.0
                            )
                        )
                    with col_value:
                        st.markdown(
                            f"**{crop_confidence:.2f}%**"
                        )
                # SUCCESS MESSAGE
                st.success(
                    f"Prediction completed successfully. "
                    f"Recommended crop: {crop.title()}."
                )
            else:
                st.error(
                    result.get(
                        "error",
                        "Prediction failed."
                    )
                )
        else:
            st.error(
                "Flask API returned an error."
            )
    except requests.exceptions.ConnectionError:
        st.error(
            """
Unable to connect to the Flask API.
Please stop Streamlit and run the application again.
"""
        )
    except Exception as error:
        st.error(
            f"Prediction error: {error}"
        )
        
# MODEL & SYSTEM INFORMATION
st.markdown("")
with st.expander(
    "⚙️ Model & System Information"
):
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown(
            "### 🧠 Deep Learning Architecture"
        )
        st.write(
            """
            • Input Layer: 7 features
            • Dense Layer: 128 neurons
            • Batch Normalization
            • Dropout: 30%
            • Dense Layer: 64 neurons
            • Batch Normalization
            • Dropout: 25%
            • Dense Layer: 32 neurons
            • Dropout: 20%
            • Output Layer: Softmax
            • Optimizer: Adam
            """
        )

    with info_col2:
        st.markdown(
            "### 🔧 Preprocessing & Deployment"
        )
        st.write(
            """
            • Missing values handled
            • Duplicate rows removed
            • StandardScaler for numerical features
            • LabelEncoder for crop labels
            • 80/20 train-test split
            • Early stopping during training
            • Flask REST API
            • Streamlit frontend
            """
        )
        
    # MODEL STATUS MESSAGE
    if model_status == "loaded":
        st.info(
            "Pre-trained Deep Learning model loaded successfully."
        )
    elif model_accuracy is not None:
        st.info(
            f"Model trained successfully. "
            f"Test accuracy: "
            f"{model_accuracy * 100:.2f}%"
        )

# Footer
st.markdown(
    """
<div class="footer">

CropWise AI &nbsp; • &nbsp;
Deep Learning &nbsp; • &nbsp;
Flask REST API &nbsp; • &nbsp;
Streamlit

</div>
""",
    unsafe_allow_html=True
)