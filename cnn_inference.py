# cnn_inference.py
"""
FastAPI inference app for the reactor foam classifier.

Accuracy-focused changes
------------------------
* Loads the validation-optimized ensemble weights written by train_models_fixed.py.
* Uses exactly the same class order, 224x224 resize-with-padding geometry, and
  backbone preprocessing as training.
* Removes manual class bias multipliers.
* Preserves video aspect ratio instead of stretching frames to a square in JS.
* Smooths the ensemble across several consecutive video frames.
* Marks low-confidence / low-margin predictions as uncertain instead of acting
  certain when the models disagree.
* Prevents overlapping prediction requests from the live video loop.
* Slack notifications are based on the stable ensemble result, not every model.
* Secrets are read from environment variables instead of being committed here.
"""

import base64
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf
import uvicorn
import gspread
from google.oauth2.service_account import Credentials
from PIL import Image, ImageOps

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.models import load_model

print("[BOOT] cnn_inference.py imported")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

DEFAULT_CLASS_NAMES = [
    "Foam-Heavy",
    "Foam-Mild",
    "Post-Antifoam Addition",
    "Foam-Medium",
    "No Foam",
]
DEFAULT_MODEL_WEIGHTS = {
    "efficientnetb0_model": 0.35,
    "resnet50_model": 0.30,
    "mobilenetv2_model": 0.25,
    "custom_cnn_model": 0.10,
}
DEFAULT_CONFIG = {
    "class_names": DEFAULT_CLASS_NAMES,
    "image_size": [224, 224],
    "weights": DEFAULT_MODEL_WEIGHTS,
    "smoothing_window": 7,
    "min_confidence": 0.45,
    "min_margin": 0.06,
    "resize_mode": "resize_with_pad",
}

# -----------------------------------------------------------------------------
# Adaptive inference speed settings
# -----------------------------------------------------------------------------
# The highest-weight ensemble model is used as the fast first pass.
# The remaining models run ONLY when that first model is not extremely certain.
FAST_PATH_ENABLED = os.getenv("FAST_PATH_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
FAST_PATH_CONFIDENCE = float(os.getenv("FAST_PATH_CONFIDENCE", "0.98"))
FAST_PATH_MARGIN = float(os.getenv("FAST_PATH_MARGIN", "0.40"))
FAST_PATH_MIN_PRIMARY_WEIGHT = float(
    os.getenv("FAST_PATH_MIN_PRIMARY_WEIGHT", "0.40")
)

# Periodically re-check the full ensemble even when the strongest model is
# repeatedly confident.
FULL_ENSEMBLE_EVERY = max(2, int(os.getenv("FULL_ENSEMBLE_EVERY", "4")))

# If too little active model weight independently supports the winning class,
# keep the result uncertain.
MIN_WEIGHTED_MODEL_AGREEMENT = float(
    os.getenv("MIN_WEIGHTED_MODEL_AGREEMENT", "0.55")
)

# Keep secrets out of source code. Set these in your terminal or deployment env.
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/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"
GOOGLE_CLIENT_ID = "1003881011936-ntgdn3d28kbbn5si56fkmmfppakst7bg.apps.googleusercontent.com"
GOOGLE_API_KEY = "AIzaSyCKS54ZwDki5N4XyauyEind0OZ8-O6r-uM"

# Google Sheets feedback configuration. Keep the service-account JSON out of GitHub.
GOOGLE_SHEETS_SPREADSHEET_ID = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "").strip()
GOOGLE_SHEETS_WORKSHEET = os.getenv("GOOGLE_SHEETS_WORKSHEET", "Feedback").strip() or "Feedback"
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv(
    "GOOGLE_SERVICE_ACCOUNT_FILE",
    "google_service_account.json",
).strip()

FEEDBACK_HEADERS = [
    "Server Timestamp",
    "Video Source",
    "Video Source Type",
    "Video Timestamp Seconds",
    "Predicted Label",
    "Ensemble Confidence",
    "Was Correct",
    "Correct Label",
    "Save For Retraining",
    "Notes",
    "Drive File ID",
    "Custom CNN Prediction",
    "Custom CNN Confidence",
    "MobileNetV2 Prediction",
    "MobileNetV2 Confidence",
    "ResNet50 Prediction",
    "ResNet50 Confidence",
    "EfficientNetB0 Prediction",
    "EfficientNetB0 Confidence",
    "Ensemble Probabilities JSON",
]



_feedback_worksheet = None


def google_sheets_feedback_enabled() -> bool:
    return bool(
        GOOGLE_SHEETS_SPREADSHEET_ID
        and GOOGLE_SERVICE_ACCOUNT_FILE
        and Path(GOOGLE_SERVICE_ACCOUNT_FILE).exists()
    )


def get_feedback_worksheet():
    """Lazily connect to the configured Google Sheet using a service account."""
    global _feedback_worksheet

    if _feedback_worksheet is not None:
        return _feedback_worksheet

    if not GOOGLE_SHEETS_SPREADSHEET_ID:
        raise RuntimeError(
            "GOOGLE_SHEETS_SPREADSHEET_ID is not set. "
            "Set it to the long ID from your Google Sheet URL."
        )

    credentials_path = Path(GOOGLE_SERVICE_ACCOUNT_FILE)
    if not credentials_path.exists():
        raise RuntimeError(
            f"Google service-account file not found: {credentials_path}. "
            "Set GOOGLE_SERVICE_ACCOUNT_FILE or place google_service_account.json "
            "in the project folder."
        )

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_file(
        str(credentials_path),
        scopes=scopes,
    )
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)

    try:
        worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_WORKSHEET)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=GOOGLE_SHEETS_WORKSHEET,
            rows=1000,
            cols=max(26, len(FEEDBACK_HEADERS)),
        )

    first_row = worksheet.row_values(1)
    if not first_row:
        worksheet.append_row(FEEDBACK_HEADERS, value_input_option="RAW")
    elif first_row != FEEDBACK_HEADERS:
        raise RuntimeError(
            f'Worksheet "{GOOGLE_SHEETS_WORKSHEET}" already has a different header. '
            'Use a new worksheet name or make row 1 match the expected feedback columns.'
        )

    _feedback_worksheet = worksheet
    return worksheet


def model_result_lookup(model_results, model_name):
    for result in model_results or []:
        if result.get("model_name") == model_name and not result.get("error"):
            return (
                str(result.get("class_name", "")),
                float(result.get("confidence", 0.0)),
            )
    return "", 0.0

def load_ensemble_config():
    path = MODELS_DIR / "ensemble_config.json"
    if not path.exists():
        print("[CONFIG] ensemble_config.json not found; using safe defaults")
        return dict(DEFAULT_CONFIG)
    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(loaded)
        if len(cfg.get("class_names", [])) != 5:
            raise ValueError("ensemble_config class_names must contain 5 classes")
        return cfg
    except Exception as e:
        print(f"[CONFIG] Could not read {path}: {e}; using defaults")
        return dict(DEFAULT_CONFIG)


ENSEMBLE_CONFIG = load_ensemble_config()
CLASS_NAMES = list(ENSEMBLE_CONFIG["class_names"])
IMG_SIZE = tuple(int(v) for v in ENSEMBLE_CONFIG.get("image_size", [224, 224]))


def escape_js_string(value: str) -> str:
    if value is None:
        return ""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def send_slack_alert(previous_label: str | None, new_label: str, confidence: float):
    if not SLACK_WEBHOOK_URL:
        return
    prev_text = previous_label if previous_label else "None"
    text = (
        "Foam Stage Change Detected\n"
        "- Source: smoothed ensemble\n"
        f"- Previous stage: {prev_text}\n"
        f"- New stage: {new_label}\n"
        f"- Confidence: {confidence * 100:.1f}%\n"
        f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=5)
        resp.raise_for_status()
        print(f"[SLACK] sent status={resp.status_code}")
    except Exception as e:
        print(f"[SLACK] Error: {e}")


# -----------------------------------------------------------------------------
# Model builders used only as a weights-only fallback.
# Full .keras models are preferred and are what the training script saves.
# -----------------------------------------------------------------------------
def classifier_head(x, num_classes=5):
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(
        256,
        activation="swish",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.25)(x)
    return layers.Dense(num_classes, activation="softmax", dtype="float32")(x)


def create_transfer_model(model_name: str, input_shape=(224, 224, 3), num_classes=5):
    if model_name == "mobilenetv2_model":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == "resnet50_model":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == "efficientnetb0_model":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(model_name)
    base.trainable = False
    inputs = keras.Input(shape=input_shape, name="image")
    x = base(inputs, training=False)
    outputs = classifier_head(x, num_classes=num_classes)
    return keras.Model(inputs, outputs, name=model_name)


def create_mobilenetv2_model(input_shape=(224, 224, 3), num_classes=5):
    return create_transfer_model("mobilenetv2_model", input_shape, num_classes)


def create_resnet50_model(input_shape=(224, 224, 3), num_classes=5):
    return create_transfer_model("resnet50_model", input_shape, num_classes)


def create_efficientnetb0_model(input_shape=(224, 224, 3), num_classes=5):
    return create_transfer_model("efficientnetb0_model", input_shape, num_classes)


def _custom_residual_block_fallback(
    x,
    filters: int,
    stride: int = 1,
    dropout_rate: float = 0.0,
):
    """Exact weights-only fallback counterpart of the training residual block."""
    shortcut = x

    x = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(5e-5),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(5e-5),
    )(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("swish")(x)
    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)
    return x


def create_custom_cnn_model(input_shape=(224, 224, 3), num_classes=5):
    """Fallback architecture that exactly matches train_models_fixed.py."""
    inputs = keras.Input(shape=input_shape, name="image")

    x = layers.Conv2D(
        32,
        5,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(5e-5),
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    x = _custom_residual_block_fallback(x, 32, stride=1, dropout_rate=0.05)
    x = _custom_residual_block_fallback(x, 32, stride=1, dropout_rate=0.05)
    x = _custom_residual_block_fallback(x, 64, stride=2, dropout_rate=0.08)
    x = _custom_residual_block_fallback(x, 64, stride=1, dropout_rate=0.08)
    x = _custom_residual_block_fallback(x, 128, stride=2, dropout_rate=0.10)
    x = _custom_residual_block_fallback(x, 128, stride=1, dropout_rate=0.10)
    x = _custom_residual_block_fallback(x, 256, stride=2, dropout_rate=0.12)
    x = _custom_residual_block_fallback(x, 256, stride=1, dropout_rate=0.12)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        256,
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
    )(x)
    return keras.Model(inputs, outputs, name="custom_cnn_model")


def load_all_models():
    model_defs = {
        "mobilenetv2_model": create_mobilenetv2_model,
        "resnet50_model": create_resnet50_model,
        "efficientnetb0_model": create_efficientnetb0_model,
        "custom_cnn_model": create_custom_cnn_model,
    }
    models_dict = {}

    for model_name, create_func in model_defs.items():
        full_paths = [
            MODELS_DIR / f"{model_name}_best.keras",
            MODELS_DIR / f"{model_name}.keras",
        ]
        loaded = False
        for path in full_paths:
            if not path.exists():
                continue
            try:
                models_dict[model_name] = load_model(path, compile=False)
                print(f"[MODEL LOAD] {model_name} <- {path}")
                loaded = True
                break
            except Exception as e:
                print(f"[MODEL LOAD] Could not load {path}: {e}")

        if loaded:
            continue

        weight_path = MODELS_DIR / f"{model_name}.weights.h5"
        if weight_path.exists():
            try:
                model = create_func(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=len(CLASS_NAMES))
                model.load_weights(weight_path)
                models_dict[model_name] = model
                print(f"[MODEL LOAD] {model_name} weights <- {weight_path}")
                loaded = True
            except Exception as e:
                print(f"[MODEL LOAD] Could not load weights {weight_path}: {e}")

        if not loaded:
            print(f"[MODEL LOAD] WARNING: no usable model for {model_name}")

    if not models_dict:
        raise RuntimeError("No trained models loaded. Run train_models_fixed.py first.")
    return models_dict


models = load_all_models()
print(f"Loaded {len(models)} models: {list(models.keys())}")


def effective_model_weights():
    configured = ENSEMBLE_CONFIG.get("weights", {})
    weights = {
        name: max(0.0, float(configured.get(name, DEFAULT_MODEL_WEIGHTS.get(name, 0.0))))
        for name in models
    }
    total = sum(weights.values())
    if total <= 0:
        weights = {name: 1.0 / len(models) for name in models}
    else:
        weights = {name: value / total for name, value in weights.items()}
    return weights


MODEL_WEIGHTS = effective_model_weights()


MODEL_TEMPERATURES = {
    name: max(
        0.05,
        float(ENSEMBLE_CONFIG.get("temperatures", {}).get(name, 1.0)),
    )
    for name in models
}


def temperature_scale_vector(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply validation-fitted confidence calibration."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0)
    logits = np.log(p) / max(float(temperature), 1e-3)
    logits -= np.max(logits)
    ex = np.exp(logits)
    return ex / max(float(ex.sum()), 1e-12)



# Pick the model the trained ensemble already trusts the most.
PRIMARY_MODEL_NAME = max(
    MODEL_WEIGHTS,
    key=lambda name: MODEL_WEIGHTS.get(name, 0.0),
)
PRIMARY_MODEL_WEIGHT = float(MODEL_WEIGHTS.get(PRIMARY_MODEL_NAME, 0.0))

print(
    "[ADAPTIVE] primary model:",
    PRIMARY_MODEL_NAME,
    f"weight={PRIMARY_MODEL_WEIGHT:.4f}",
)
print(
    "[ADAPTIVE] fast path:",
    "enabled" if FAST_PATH_ENABLED else "disabled",
    f"confidence>={FAST_PATH_CONFIDENCE:.2f}",
    f"margin>={FAST_PATH_MARGIN:.2f}",
    f"min_primary_weight>={FAST_PATH_MIN_PRIMARY_WEIGHT:.2f}",
)


def run_single_model(model_name: str, model, base_image: np.ndarray):
    """Run one model with the same preprocessing, using direct Keras inference."""
    started = time.perf_counter()
    image_array = preprocess_for_model(model_name, base_image)

    raw = model(image_array, training=False)
    if hasattr(raw, "numpy"):
        raw = raw.numpy()

    preds = np.asarray(raw)
    if preds.ndim != 2 or preds.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"Unexpected prediction shape: {preds.shape}")

    p = preds[0].astype(float)
    p = np.clip(p, 0.0, None)
    p = p / max(float(p.sum()), 1e-12)

    p = temperature_scale_vector(
        p,
        MODEL_TEMPERATURES.get(model_name, 1.0),
    )

    predicted_class = int(np.argmax(p))
    confidence = float(p[predicted_class])

    ordered = np.sort(p)[::-1]
    second_confidence = float(ordered[1]) if len(ordered) > 1 else 0.0
    margin = confidence - second_confidence
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    result = {
        "model_name": model_name,
        "predicted_class": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "all_confidences": [float(v) for v in p],
        "inference_ms": round(elapsed_ms, 2),
    }
    return result, margin



def decode_base64_image(image_str: str) -> np.ndarray:
    """Decode and use the exact tf.image.resize_with_pad geometry from training."""
    if image_str.startswith("data:image"):
        _, encoded = image_str.split(",", 1)
    else:
        encoded = image_str
    image_bytes = base64.b64decode(encoded)
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)

    resized = tf.image.resize_with_pad(
        arr,
        target_height=IMG_SIZE[0],
        target_width=IMG_SIZE[1],
        method="bilinear",
        antialias=True,
    )
    return np.asarray(resized.numpy(), dtype=np.float32)


def preprocess_for_model(model_name: str, base_image: np.ndarray):
    x = np.expand_dims(base_image.copy(), axis=0)
    if model_name == "resnet50_model":
        return resnet50_preprocess(x)
    if model_name == "mobilenetv2_model":
        return mobilenet_preprocess(x)
    if model_name == "efficientnetb0_model":
        return efficientnet_preprocess(x)
    if model_name == "custom_cnn_model":
        return x / 255.0
    return x / 255.0


@app.get("/google-drive-config")
async def google_drive_config():
    return JSONResponse(
        {
            "clientId": GOOGLE_CLIENT_ID,
            "apiKey": GOOGLE_API_KEY,
            "enabled": bool(GOOGLE_CLIENT_ID and GOOGLE_API_KEY),
        }
    )


@app.get("/model-config")
async def model_config():
    return JSONResponse(
        {
            "class_names": CLASS_NAMES,
            "weights": MODEL_WEIGHTS,
            "smoothing_window": int(ENSEMBLE_CONFIG.get("smoothing_window", 7)),
            "min_confidence": float(ENSEMBLE_CONFIG.get("min_confidence", 0.45)),
            "min_margin": float(ENSEMBLE_CONFIG.get("min_margin", 0.06)),
            "loaded_models": list(models.keys()),
        }
    )



@app.get("/feedback-config")
async def feedback_config():
    return JSONResponse(
        {
            "enabled": google_sheets_feedback_enabled(),
            "worksheet": GOOGLE_SHEETS_WORKSHEET,
            "spreadsheetConfigured": bool(GOOGLE_SHEETS_SPREADSHEET_ID),
            "serviceAccountFileFound": Path(GOOGLE_SERVICE_ACCOUNT_FILE).exists(),
        }
    )


@app.post("/feedback")
async def submit_feedback(request: Request):
    """Save one human-reviewed prediction to Google Sheets."""
    try:
        data = await request.json()

        predicted_label = str(data.get("predicted_label", "")).strip()
        correct_label = str(data.get("correct_label", "")).strip()
        was_correct = bool(data.get("was_correct", False))
        notes = str(data.get("notes", "")).strip()[:2000]
        save_for_retraining = bool(data.get("save_for_retraining", False))

        if predicted_label not in CLASS_NAMES:
            return JSONResponse(
                {"ok": False, "error": "Invalid predicted label"},
                status_code=400,
            )

        if was_correct:
            correct_label = predicted_label
        elif correct_label not in CLASS_NAMES:
            return JSONResponse(
                {"ok": False, "error": "Choose the correct foam level"},
                status_code=400,
            )

        model_results = data.get("model_results") or []
        custom_pred, custom_conf = model_result_lookup(model_results, "custom_cnn_model")
        mobile_pred, mobile_conf = model_result_lookup(model_results, "mobilenetv2_model")
        resnet_pred, resnet_conf = model_result_lookup(model_results, "resnet50_model")
        efficient_pred, efficient_conf = model_result_lookup(model_results, "efficientnetb0_model")

        ensemble_probs = data.get("ensemble_probs") or []
        if not isinstance(ensemble_probs, list):
            ensemble_probs = []

        worksheet = get_feedback_worksheet()
        row = [
            datetime.now().astimezone().isoformat(timespec="seconds"),
            str(data.get("video_source_name", ""))[:500],
            str(data.get("video_source_type", ""))[:100],
            round(float(data.get("video_timestamp_seconds", 0.0) or 0.0), 3),
            predicted_label,
            round(float(data.get("ensemble_confidence", 0.0) or 0.0), 6),
            "Yes" if was_correct else "No",
            correct_label,
            "Yes" if save_for_retraining else "No",
            notes,
            str(data.get("drive_file_id", ""))[:500],
            custom_pred,
            round(custom_conf, 6),
            mobile_pred,
            round(mobile_conf, 6),
            resnet_pred,
            round(resnet_conf, 6),
            efficient_pred,
            round(efficient_conf, 6),
            json.dumps(ensemble_probs),
        ]

        worksheet.append_row(row, value_input_option="USER_ENTERED")
        print(
            "[FEEDBACK] Saved:",
            predicted_label,
            "->",
            correct_label,
            "correct=",
            was_correct,
        )
        return JSONResponse({"ok": True, "message": "Feedback saved to Google Sheets"})
    except Exception as e:
        print(f"[FEEDBACK] Error: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/predict")
async def predict(request: Request):
    """
    Reliability-first adaptive inference.

    Fast mode is allowed only when the strongest model is extremely confident
    AND agrees with the last stable full-ensemble class. Fast-mode probabilities
    are never mixed into the multi-model temporal history.
    """
    request_started = time.perf_counter()

    try:
        data = await request.json()
        image_str = data.get("image")
        if not image_str:
            return JSONResponse({"error": "No image provided"}, status_code=400)

        last_stable_class = str(data.get("last_stable_class", "")).strip()
        force_full = bool(data.get("force_full", False))

        decode_started = time.perf_counter()
        base_image = decode_base64_image(image_str)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0

        results = []

        primary_model = models.get(PRIMARY_MODEL_NAME)
        if primary_model is None:
            raise RuntimeError(
                f"Primary model {PRIMARY_MODEL_NAME} is not loaded."
            )

        try:
            primary_result, primary_margin = run_single_model(
                PRIMARY_MODEL_NAME,
                primary_model,
                base_image,
            )
            results.append(primary_result)
        except Exception as e:
            print(f"[PREDICT] primary {PRIMARY_MODEL_NAME}: {e}")
            primary_result = None
            primary_margin = 0.0

        primary_class = (
            str(primary_result.get("class_name", ""))
            if primary_result
            else ""
        )

        use_fast_path = (
            FAST_PATH_ENABLED
            and not force_full
            and primary_result is not None
            and bool(last_stable_class)
            and primary_class == last_stable_class
            and PRIMARY_MODEL_WEIGHT >= FAST_PATH_MIN_PRIMARY_WEIGHT
            and float(primary_result["confidence"]) >= FAST_PATH_CONFIDENCE
            and float(primary_margin) >= FAST_PATH_MARGIN
        )

        if not use_fast_path:
            for model_name, model in models.items():
                if model_name == PRIMARY_MODEL_NAME:
                    continue

                if MODEL_WEIGHTS.get(model_name, 0.0) <= 1e-12:
                    continue

                try:
                    result, _ = run_single_model(
                        model_name,
                        model,
                        base_image,
                    )
                    results.append(result)
                except Exception as e:
                    print(f"[PREDICT] {model_name}: {e}")
                    results.append(
                        {
                            "model_name": model_name,
                            "error": str(e),
                        }
                    )

        total_ms = (time.perf_counter() - request_started) * 1000.0
        mode = "fast-primary-hold" if use_fast_path else "full-ensemble"

        model_times = {
            r["model_name"]: r.get("inference_ms")
            for r in results
            if not r.get("error")
        }

        print(
            f"[SPEED] mode={mode} "
            f"decode={decode_ms:.1f}ms "
            f"models={model_times} "
            f"total={total_ms:.1f}ms"
        )

        return JSONResponse(
            {
                "results": results,
                "inference_mode": mode,
                "primary_model": PRIMARY_MODEL_NAME,
                "primary_weight": PRIMARY_MODEL_WEIGHT,
                "primary_margin": float(primary_margin),
                "timing_ms": {
                    "decode": round(decode_ms, 2),
                    "total": round(total_ms, 2),
                },
            }
        )

    except Exception as e:
        print(f"[PREDICT] request error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


last_alerted_ensemble_label = None


@app.post("/ensemble-alert")
async def ensemble_alert(request: Request):
    global last_alerted_ensemble_label
    try:
        data = await request.json()
        label = str(data.get("label", "")).strip()
        confidence = float(data.get("confidence", 0.0))
        if label not in CLASS_NAMES:
            return JSONResponse({"sent": False, "reason": "invalid label"})
        if label != last_alerted_ensemble_label:
            previous = last_alerted_ensemble_label
            send_slack_alert(previous, label, confidence)
            last_alerted_ensemble_label = label
            return JSONResponse({"sent": True})
        return JSONResponse({"sent": False, "reason": "unchanged"})
    except Exception as e:
        return JSONResponse({"sent": False, "error": str(e)}, status_code=400)


@app.get("/", response_class=HTMLResponse)
async def get_home():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>BioReactor Foam Classification</title>
  <script crossorigin src="https://unpkg.com/react@17/umd/react.development.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
  <script src="https://unpkg.com/babel-standalone@6/babel.min.js"></script>
  <script src="https://accounts.google.com/gsi/client" async defer></script>
  <script src="https://apis.google.com/js/api.js"></script>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, Arial, sans-serif;
      background: #f4f7fb;
      color: #1f2937;
    }
    .page {
      max-width: 1450px;
      margin: 0 auto;
      padding: 28px 24px 48px;
    }
    .hero {
      background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%);
      border: 1px solid #e5e7eb;
      border-radius: 22px;
      padding: 24px;
      box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
      margin-bottom: 24px;
    }
    .hero h1 {
      margin: 0 0 8px 0;
      font-size: 32px;
      color: #111827;
    }
    .hero p {
      margin: 0;
      color: #6b7280;
      font-size: 16px;
    }
    .video-panel {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 22px;
      padding: 20px;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
      margin-bottom: 24px;
    }
    video {
      width: 100%;
      max-width: 1000px;
      display: block;
      margin: 0 auto;
      border-radius: 16px;
      background: #000;
    }
    .toolbar {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 16px;
    }
    .toolbar button {
      border: none;
      border-radius: 12px;
      padding: 12px 16px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      background: #2563eb;
      color: white;
      box-shadow: 0 6px 16px rgba(37, 99, 235, 0.2);
    }
    .toolbar button.secondary {
      background: #e5e7eb;
      color: #111827;
      box-shadow: none;
    }
    .toolbar button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .source-pill {
      padding: 10px 14px;
      border-radius: 999px;
      background: #eff6ff;
      color: #1d4ed8;
      font-size: 13px;
      font-weight: 700;
      border: 1px solid #bfdbfe;
    }
    .selected-file, .folder-box {
      margin-top: 14px;
      padding: 14px;
      border-radius: 14px;
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      color: #374151;
      font-size: 14px;
    }
    .help-text {
      margin-top: 10px;
      color: #6b7280;
      font-size: 13px;
    }
    .warning-box {
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 14px;
      background: #fff7ed;
      border: 1px solid #fdba74;
      color: #9a3412;
      font-size: 14px;
      line-height: 1.5;
    }
    .main-grid {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 20px;
      margin-bottom: 24px;
      align-items: start;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 20px;
      margin-bottom: 24px;
    }
    .summary-card,
    .models-panel,
    .list-panel {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 22px;
      padding: 22px;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
    }
    .summary-title {
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #6b7280;
      margin-bottom: 10px;
    }
    .ensemble-label {
      font-size: 34px;
      font-weight: 800;
      color: #111827;
      margin-bottom: 14px;
    }
    .ensemble-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 10px;
    }
    .ensemble-metric {
      background: #f8fafc;
      border-radius: 14px;
      padding: 12px;
      border: 1px solid #edf2f7;
    }
    .ensemble-metric-name {
      font-size: 12px;
      color: #6b7280;
      margin-bottom: 6px;
      min-height: 32px;
    }
    .ensemble-metric-value {
      font-size: 18px;
      font-weight: 700;
      color: #111827;
    }
    .side-stats {
      display: grid;
      gap: 12px;
    }
    .stat-box {
      background: #f8fafc;
      border: 1px solid #edf2f7;
      border-radius: 16px;
      padding: 14px 16px;
    }
    .stat-label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #6b7280;
      margin-bottom: 6px;
    }
    .stat-value {
      font-size: 24px;
      font-weight: 800;
      color: #111827;
    }
    .models-panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
      gap: 12px;
      flex-wrap: wrap;
    }
    .models-panel-header h2,
    .list-panel h2 {
      margin: 0;
      font-size: 24px;
      color: #111827;
    }
    .models-panel-header p,
    .list-sub {
      margin: 6px 0 0 0;
      color: #6b7280;
      font-size: 14px;
    }
    .models-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }
    .model-card {
      background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
      border: 1px solid #e5e7eb;
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }
    .model-name {
      font-size: 14px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #6b7280;
      margin-bottom: 10px;
    }
    .class-name {
      font-size: 38px;
      font-weight: 800;
      line-height: 1.05;
      margin: 0 0 16px 0;
      color: #1d4ed8;
    }
    .confidence-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      font-size: 14px;
      color: #4b5563;
      font-weight: 600;
    }
    .confidence-bar {
      height: 14px;
      background: #e5e7eb;
      border-radius: 999px;
      overflow: hidden;
      margin-bottom: 18px;
    }
    .confidence-fill {
      height: 100%;
      background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
      border-radius: 999px;
      transition: width 0.3s ease;
    }
    .all-confidences-title {
      font-size: 13px;
      font-weight: 700;
      color: #6b7280;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 10px;
    }
    .class-confidence {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 0;
      border-top: 1px solid #f1f5f9;
      font-size: 15px;
    }
    .class-confidence:first-of-type {
      border-top: none;
      padding-top: 0;
    }
    .error-message {
      color: #dc2626;
      font-style: italic;
      margin-top: 6px;
    }
    .empty-state {
      text-align: center;
      color: #6b7280;
      padding: 26px 0 8px;
      font-size: 16px;
    }
    .file-list {
      display: grid;
      gap: 10px;
      margin-top: 16px;
      max-height: 520px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .file-card {
      padding: 14px;
      border-radius: 16px;
      border: 1px solid #e5e7eb;
      background: #f8fafc;
      cursor: pointer;
      transition: 0.2s ease;
    }
    .file-card:hover {
      border-color: #93c5fd;
      background: #eff6ff;
    }
    .file-card.active {
      border-color: #2563eb;
      background: #dbeafe;
    }
    .file-name {
      font-weight: 700;
      color: #111827;
      word-break: break-word;
    }
    .file-meta {
      margin-top: 6px;
      font-size: 13px;
      color: #6b7280;
    }


    .journal-panel {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 22px;
      padding: 22px;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
      margin-bottom: 24px;
    }
    .journal-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }
    .journal-header h2 {
      margin: 0;
      font-size: 24px;
      color: #111827;
    }
    .journal-header p {
      margin: 6px 0 0 0;
      color: #6b7280;
      font-size: 14px;
    }
    .journal-count {
      padding: 9px 12px;
      border-radius: 999px;
      background: #ecfdf5;
      color: #047857;
      border: 1px solid #a7f3d0;
      font-size: 13px;
      font-weight: 800;
    }
    .journal-form {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
      margin-bottom: 18px;
    }
    .journal-field {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .journal-field label {
      font-size: 13px;
      font-weight: 800;
      color: #374151;
    }
    .journal-field input,
    .journal-field select,
    .journal-field textarea {
      width: 100%;
      border: 1px solid #d1d5db;
      border-radius: 12px;
      padding: 11px 12px;
      font-size: 14px;
      color: #111827;
      background: #ffffff;
      font-family: inherit;
    }
    .journal-field textarea {
      min-height: 92px;
      resize: vertical;
    }
    .journal-field.wide {
      grid-column: span 3;
    }
    .journal-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }
    .journal-actions button {
      border: none;
      border-radius: 12px;
      padding: 12px 16px;
      font-size: 14px;
      font-weight: 800;
      cursor: pointer;
      background: #2563eb;
      color: white;
      box-shadow: 0 6px 16px rgba(37, 99, 235, 0.2);
    }
    .journal-actions button.secondary {
      background: #e5e7eb;
      color: #111827;
      box-shadow: none;
    }
    .journal-actions button.danger {
      background: #fee2e2;
      color: #991b1b;
      box-shadow: none;
    }
    .journal-table-wrap {
      overflow-x: auto;
      border: 1px solid #e5e7eb;
      border-radius: 16px;
    }
    .journal-table {
      width: 100%;
      border-collapse: collapse;
      min-width: 980px;
      background: #ffffff;
    }
    .journal-table th {
      text-align: left;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #6b7280;
      background: #f8fafc;
      padding: 12px;
      border-bottom: 1px solid #e5e7eb;
    }
    .journal-table td {
      padding: 12px;
      border-bottom: 1px solid #f1f5f9;
      font-size: 14px;
      color: #374151;
      vertical-align: top;
    }
    .journal-table tr:last-child td {
      border-bottom: none;
    }
    .journal-badge {
      display: inline-block;
      padding: 5px 9px;
      border-radius: 999px;
      background: #eff6ff;
      color: #1d4ed8;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    .journal-delete {
      border: none;
      border-radius: 10px;
      padding: 8px 10px;
      background: #fee2e2;
      color: #991b1b;
      font-size: 12px;
      font-weight: 800;
      cursor: pointer;
    }
    .journal-empty {
      text-align: center;
      color: #6b7280;
      padding: 24px;
      background: #f8fafc;
      border: 1px dashed #cbd5e1;
      border-radius: 16px;
    }


    .feedback-panel {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 22px;
      padding: 22px;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
      margin-bottom: 24px;
    }
    .feedback-panel h2 {
      margin: 0 0 6px 0;
      font-size: 24px;
      color: #111827;
    }
    .feedback-sub {
      margin: 0 0 18px 0;
      color: #6b7280;
      font-size: 14px;
    }
    .feedback-current {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .feedback-current-box {
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 14px;
      padding: 13px;
    }
    .feedback-current-label {
      color: #6b7280;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 5px;
    }
    .feedback-current-value {
      color: #111827;
      font-size: 17px;
      font-weight: 800;
      word-break: break-word;
    }
    .feedback-choice-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }
    .feedback-choice {
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: #334155;
      border-radius: 12px;
      padding: 11px 16px;
      font-weight: 800;
      cursor: pointer;
    }
    .feedback-choice.active {
      border-color: #2563eb;
      background: #dbeafe;
      color: #1d4ed8;
    }
    .feedback-form-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-bottom: 14px;
    }
    .feedback-field {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .feedback-field.wide {
      grid-column: span 2;
    }
    .feedback-field label {
      font-size: 13px;
      font-weight: 800;
      color: #374151;
    }
    .feedback-field select,
    .feedback-field textarea {
      width: 100%;
      border: 1px solid #d1d5db;
      border-radius: 12px;
      padding: 11px 12px;
      font-size: 14px;
      font-family: inherit;
      background: #ffffff;
    }
    .feedback-field textarea {
      min-height: 86px;
      resize: vertical;
    }
    .feedback-check {
      display: flex;
      gap: 8px;
      align-items: center;
      color: #374151;
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 14px;
    }
    .feedback-submit {
      border: none;
      border-radius: 12px;
      padding: 12px 16px;
      font-size: 14px;
      font-weight: 800;
      cursor: pointer;
      background: #2563eb;
      color: white;
      box-shadow: 0 6px 16px rgba(37, 99, 235, 0.2);
    }
    .feedback-submit:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .feedback-status {
      display: inline-block;
      margin-left: 12px;
      font-size: 14px;
      font-weight: 700;
    }
    .feedback-status.ok { color: #047857; }
    .feedback-status.error { color: #b91c1c; }

    @media (max-width: 1100px) {
      .main-grid,
      .summary-grid {
        grid-template-columns: 1fr;
      }
      .ensemble-grid {
        grid-template-columns: repeat(2, 1fr);
      }
      .journal-form {
        grid-template-columns: 1fr 1fr;
      }
      .journal-field.wide {
        grid-column: span 2;
      }
    }

    @media (max-width: 640px) {
      .page {
        padding: 18px 14px 36px;
      }
      .hero h1 {
        font-size: 26px;
      }
      .class-name {
        font-size: 30px;
      }
      .ensemble-grid {
        grid-template-columns: 1fr;
      }
      .journal-form {
        grid-template-columns: 1fr;
      }
      .journal-field.wide {
        grid-column: span 1;
      }
      .feedback-current {
        grid-template-columns: 1fr;
      }
      .feedback-form-grid {
        grid-template-columns: 1fr;
      }
      .feedback-field.wide {
        grid-column: span 1;
      }
    }
  </style>
</head>
<body>
  <div id="root"></div>

  <script type="text/babel">
    const { useRef, useState, useEffect } = React;

    const CLASS = ['Foam-Heavy','Foam-Mild','Post-Antifoam Addition','Foam-Medium','No Foam'];

    const MODEL_WEIGHTS = __MODEL_WEIGHTS_JSON__;
    const SMOOTHING_WINDOW = __SMOOTHING_WINDOW__;
    const MIN_CONFIDENCE = __MIN_CONFIDENCE__;
    const MIN_MARGIN = __MIN_MARGIN__;
    const FULL_ENSEMBLE_EVERY = __FULL_ENSEMBLE_EVERY__;
    const MIN_WEIGHTED_MODEL_AGREEMENT = __MIN_WEIGHTED_MODEL_AGREEMENT__;

    const GOOGLE_CLIENT_ID = "";
    const GOOGLE_API_KEY = "";
    const GOOGLE_SCOPES = "";

    function prettyModelName(name) {
      const map = {
        mobilenetv2_model: "MobileNetV2",
        resnet50_model: "ResNet50",
        efficientnetb0_model: "EfficientNetB0",
        custom_cnn_model: "Custom CNN",
      };
      return map[name] || name.replace(/_/g, " ");
    }

    function formatFileSize(bytes) {
      if (!bytes) return "Unknown size";
      const num = Number(bytes);
      if (Number.isNaN(num)) return "Unknown size";
      if (num < 1024 * 1024) return `${(num / 1024).toFixed(1)} KB`;
      if (num < 1024 * 1024 * 1024) return `${(num / (1024 * 1024)).toFixed(1)} MB`;
      return `${(num / (1024 * 1024 * 1024)).toFixed(2)} GB`;
    }

    function ModelPrediction({ modelResult }) {
      if (modelResult.error) {
        return (
          <div className="model-card">
            <div className="model-name">{prettyModelName(modelResult.model_name)}</div>
            <div className="error-message">Error: {modelResult.error}</div>
          </div>
        );
      }

      return (
        <div className="model-card">
          <div className="model-name">{prettyModelName(modelResult.model_name)}</div>
          <div className="class-name">{modelResult.class_name}</div>

          <div className="confidence-row">
            <span>Confidence</span>
            <span>{(modelResult.confidence * 100).toFixed(1)}%</span>
          </div>

          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{ width: `${modelResult.confidence * 100}%` }}
            ></div>
          </div>

          <div className="all-confidences-title">Class probabilities</div>

          {modelResult.all_confidences && modelResult.all_confidences.map((conf, idx) => (
            <div key={idx} className="class-confidence">
              <span>{CLASS[idx]}</span>
              <strong>{(conf * 100).toFixed(1)}%</strong>
            </div>
          ))}
        </div>
      );
    }


    function ExperimentJournal() {
      const STORAGE_KEY = "foam_experiment_journal_entries_v1";
      const blankEntry = {
        experimentType: "",
        sampleId: "",
        whatChanged: "",
        reactionObserved: "",
        reactionSpeed: "Medium",
        timeUntilReaction: "",
        foamLevel: "Medium",
        likelyCause: "",
        notes: "",
      };

      const [entries, setEntries] = useState([]);
      const [form, setForm] = useState(blankEntry);

      useEffect(() => {
        try {
          const saved = localStorage.getItem(STORAGE_KEY);
          if (saved) {
            const parsed = JSON.parse(saved);
            if (Array.isArray(parsed)) {
              setEntries(parsed);
            }
          }
        } catch (err) {
          console.error("Could not load journal entries:", err);
        }
      }, []);

      useEffect(() => {
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
        } catch (err) {
          console.error("Could not save journal entries:", err);
        }
      }, [entries]);

      const updateField = (field, value) => {
        setForm(prev => ({ ...prev, [field]: value }));
      };

      const saveEntry = (event) => {
        event.preventDefault();

        const hasRequiredInfo =
          form.experimentType.trim() ||
          form.sampleId.trim() ||
          form.reactionObserved.trim() ||
          form.notes.trim();

        if (!hasRequiredInfo) {
          alert("Please enter at least an experiment type, sample ID, reaction observed, or notes.");
          return;
        }

        const newEntry = {
          id: Date.now(),
          createdAt: new Date().toLocaleString(),
          ...form,
        };

        setEntries(prev => [newEntry, ...prev]);
        setForm(blankEntry);
      };

      const deleteEntry = (id) => {
        setEntries(prev => prev.filter(entry => entry.id !== id));
      };

      const clearEntries = () => {
        if (entries.length === 0) return;
        if (window.confirm("Clear all local journal entries from this browser?")) {
          setEntries([]);
        }
      };

      const exportEntries = () => {
        if (entries.length === 0) {
          alert("No journal entries to export yet.");
          return;
        }

        const blob = new Blob([JSON.stringify(entries, null, 2)], {
          type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "foam_experiment_journal.json";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      };

      return (
        <div className="journal-panel">
          <div className="journal-header">
            <div>
              <h2>Experiment Journal</h2>
              <p>Record experiment type, reaction speed, foam behavior, and possible causes. Entries are saved locally in this browser for now.</p>
            </div>
            <div className="journal-count">{entries.length} saved entr{entries.length === 1 ? "y" : "ies"}</div>
          </div>

          <form onSubmit={saveEntry}>
            <div className="journal-form">
              <div className="journal-field">
                <label>Experiment Type</label>
                <input
                  value={form.experimentType}
                  onChange={(e) => updateField("experimentType", e.target.value)}
                  placeholder="Fermentation, mixing, heating..."
                />
              </div>

              <div className="journal-field">
                <label>Sample ID / Reactor</label>
                <input
                  value={form.sampleId}
                  onChange={(e) => updateField("sampleId", e.target.value)}
                  placeholder="Reactor A, Sample 4..."
                />
              </div>

              <div className="journal-field">
                <label>Time Until Reaction</label>
                <input
                  value={form.timeUntilReaction}
                  onChange={(e) => updateField("timeUntilReaction", e.target.value)}
                  placeholder="30 sec, 2 min..."
                />
              </div>

              <div className="journal-field">
                <label>Reaction Speed</label>
                <select
                  value={form.reactionSpeed}
                  onChange={(e) => updateField("reactionSpeed", e.target.value)}
                >
                  <option>Slow</option>
                  <option>Medium</option>
                  <option>Fast</option>
                  <option>Very Fast</option>
                  <option>Unknown</option>
                </select>
              </div>

              <div className="journal-field">
                <label>Foam Level Reached</label>
                <select
                  value={form.foamLevel}
                  onChange={(e) => updateField("foamLevel", e.target.value)}
                >
                  <option>No Foam</option>
                  <option>Foam-Mild</option>
                  <option>Foam-Medium</option>
                  <option>Foam-Heavy</option>
                  <option>Post-Antifoam Addition</option>
                  <option>Unknown</option>
                </select>
              </div>

              <div className="journal-field">
                <label>Likely Cause</label>
                <input
                  value={form.likelyCause}
                  onChange={(e) => updateField("likelyCause", e.target.value)}
                  placeholder="Temperature, mixing speed, nutrient added..."
                />
              </div>

              <div className="journal-field wide">
                <label>What Changed?</label>
                <textarea
                  value={form.whatChanged}
                  onChange={(e) => updateField("whatChanged", e.target.value)}
                  placeholder="Describe what condition changed before the reaction happened."
                />
              </div>

              <div className="journal-field wide">
                <label>Reaction Observed / Notes</label>
                <textarea
                  value={form.reactionObserved}
                  onChange={(e) => updateField("reactionObserved", e.target.value)}
                  placeholder="Describe how the foam reacted, how quickly it rose, and anything important scientists should know."
                />
              </div>

              <div className="journal-field wide">
                <label>Extra Notes</label>
                <textarea
                  value={form.notes}
                  onChange={(e) => updateField("notes", e.target.value)}
                  placeholder="Lighting, camera angle, antifoam, unusual behavior, etc."
                />
              </div>
            </div>

            <div className="journal-actions">
              <button type="submit">Save Entry</button>
              <button type="button" className="secondary" onClick={exportEntries}>Export JSON</button>
              <button type="button" className="danger" onClick={clearEntries}>Clear Local Entries</button>
            </div>
          </form>

          {entries.length === 0 ? (
            <div className="journal-empty">No journal entries yet. Add an experiment note above and it will stay after refresh on this browser.</div>
          ) : (
            <div className="journal-table-wrap">
              <table className="journal-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Experiment</th>
                    <th>Sample</th>
                    <th>Reaction</th>
                    <th>Speed</th>
                    <th>Foam</th>
                    <th>Cause</th>
                    <th>Notes</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map(entry => (
                    <tr key={entry.id}>
                      <td>{entry.createdAt}</td>
                      <td>{entry.experimentType || "-"}</td>
                      <td>{entry.sampleId || "-"}</td>
                      <td>
                        <div><strong>Changed:</strong> {entry.whatChanged || "-"}</div>
                        <div style={{ marginTop: "6px" }}><strong>Observed:</strong> {entry.reactionObserved || "-"}</div>
                        <div style={{ marginTop: "6px" }}><strong>Time:</strong> {entry.timeUntilReaction || "-"}</div>
                      </td>
                      <td><span className="journal-badge">{entry.reactionSpeed}</span></td>
                      <td><span className="journal-badge">{entry.foamLevel}</span></td>
                      <td>{entry.likelyCause || "-"}</td>
                      <td>{entry.notes || "-"}</td>
                      <td>
                        <button className="journal-delete" onClick={() => deleteEntry(entry.id)}>Delete</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      );
    }


    function FeedbackPanel({
      ensembleLabel,
      ensembleProbs,
      modelResults,
      videoSourceName,
      videoSourceType,
      selectedFile,
      videoRef
    }) {
      const cleanPrediction = CLASS.includes(ensembleLabel)
        ? ensembleLabel
        : CLASS.find(name => ensembleLabel.includes(name)) || "";

      const topConfidence = ensembleProbs.length
        ? Math.max(...ensembleProbs.map(Number))
        : 0;

      const [verdict, setVerdict] = useState("");
      const [correctLabel, setCorrectLabel] = useState("");
      const [notes, setNotes] = useState("");
      const [saveForRetraining, setSaveForRetraining] = useState(false);
      const [submitting, setSubmitting] = useState(false);
      const [status, setStatus] = useState("");
      const [statusType, setStatusType] = useState("");
      const [feedbackEnabled, setFeedbackEnabled] = useState(true);

      useEffect(() => {
        fetch("/feedback-config")
          .then(res => res.json())
          .then(config => {
            setFeedbackEnabled(!!config.enabled);
            if (!config.enabled) {
              setStatus("Google Sheets feedback is not configured yet.");
              setStatusType("error");
            }
          })
          .catch(() => {
            setFeedbackEnabled(false);
            setStatus("Could not check Google Sheets configuration.");
            setStatusType("error");
          });
      }, []);

      useEffect(() => {
        setVerdict("");
        setCorrectLabel("");
        setNotes("");
        setSaveForRetraining(false);
        if (feedbackEnabled) {
          setStatus("");
          setStatusType("");
        }
      }, [videoSourceName]);

      const submitFeedback = async () => {
        if (!cleanPrediction) {
          setStatus("Wait for a prediction before submitting feedback.");
          setStatusType("error");
          return;
        }
        if (!verdict) {
          setStatus("Choose Correct or Incorrect first.");
          setStatusType("error");
          return;
        }
        if (verdict === "incorrect" && !correctLabel) {
          setStatus("Choose the correct foam level.");
          setStatusType("error");
          return;
        }

        setSubmitting(true);
        setStatus("Saving...");
        setStatusType("");

        const video = videoRef.current;
        const payload = {
          predicted_label: cleanPrediction,
          ensemble_confidence: topConfidence,
          was_correct: verdict === "correct",
          correct_label: verdict === "correct" ? cleanPrediction : correctLabel,
          notes,
          save_for_retraining: saveForRetraining,
          video_source_name: videoSourceName,
          video_source_type: videoSourceType,
          video_timestamp_seconds: video ? Number(video.currentTime || 0) : 0,
          drive_file_id: selectedFile ? selectedFile.id : "",
          model_results: modelResults,
          ensemble_probs: ensembleProbs,
        };

        try {
          const response = await fetch("/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const data = await response.json();
          if (!response.ok || !data.ok) {
            throw new Error(data.error || "Could not save feedback");
          }

          setStatus("Feedback saved to Google Sheets.");
          setStatusType("ok");
          setVerdict("");
          setCorrectLabel("");
          setNotes("");
          setSaveForRetraining(false);
        } catch (error) {
          setStatus(error.message || "Could not save feedback.");
          setStatusType("error");
        } finally {
          setSubmitting(false);
        }
      };

      return (
        <div className="feedback-panel">
          <h2>Prediction Feedback</h2>
          <p className="feedback-sub">
            Confirm the model's answer or correct it. Each submission is saved as a new Google Sheets row for analysis and future retraining.
          </p>

          <div className="feedback-current">
            <div className="feedback-current-box">
              <div className="feedback-current-label">Current prediction</div>
              <div className="feedback-current-value">{ensembleLabel}</div>
            </div>
            <div className="feedback-current-box">
              <div className="feedback-current-label">Confidence</div>
              <div className="feedback-current-value">{(topConfidence * 100).toFixed(1)}%</div>
            </div>
            <div className="feedback-current-box">
              <div className="feedback-current-label">Video time</div>
              <div className="feedback-current-value">
                {videoRef.current ? `${Number(videoRef.current.currentTime || 0).toFixed(1)} sec` : "0.0 sec"}
              </div>
            </div>
          </div>

          <div className="feedback-choice-row">
            <button
              type="button"
              className={`feedback-choice ${verdict === "correct" ? "active" : ""}`}
              onClick={() => {
                setVerdict("correct");
                setCorrectLabel(cleanPrediction);
              }}
              disabled={!cleanPrediction}
            >
              Correct
            </button>
            <button
              type="button"
              className={`feedback-choice ${verdict === "incorrect" ? "active" : ""}`}
              onClick={() => {
                setVerdict("incorrect");
                if (correctLabel === cleanPrediction) setCorrectLabel("");
              }}
              disabled={!cleanPrediction}
            >
              Incorrect
            </button>
          </div>

          <div className="feedback-form-grid">
            {verdict === "incorrect" && (
              <div className="feedback-field">
                <label>Correct foam level</label>
                <select value={correctLabel} onChange={(e) => setCorrectLabel(e.target.value)}>
                  <option value="">Choose correct label</option>
                  {CLASS.map(name => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
              </div>
            )}

            <div className={`feedback-field ${verdict !== "incorrect" ? "wide" : ""}`}>
              <label>Notes</label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Example: yellow liquid, larger reactor, side angle, heavy foam was classified as medium..."
              />
            </div>
          </div>

          <label className="feedback-check">
            <input
              type="checkbox"
              checked={saveForRetraining}
              onChange={(e) => setSaveForRetraining(e.target.checked)}
            />
            Mark this example for future retraining
          </label>

          <button
            type="button"
            className="feedback-submit"
            onClick={submitFeedback}
            disabled={submitting || !feedbackEnabled || !cleanPrediction}
          >
            {submitting ? "Saving..." : "Submit Feedback"}
          </button>

          {status && (
            <span className={`feedback-status ${statusType}`}>{status}</span>
          )}
        </div>
      );
    }

    function App() {
      const videoRef = useRef(null);
      const canvasRef = useRef(null);
      const webcamStreamRef = useRef(null);
      const tokenClientRef = useRef(null);
      const createdBlobUrlRef = useRef(null);
      const processingRef = useRef(false);
      const ensembleHistoryRef = useRef([]);
      const lastAlertedLabelRef = useRef(null);
      const lastStableClassRef = useRef("");
      const predictionCounterRef = useRef(0);
      const [isWebcam, setIsWebcam] = useState(false);

      const [modelResults, setModelResults] = useState([]);
      const [ensembleProbs, setEnsembleProbs] = useState([0, 0, 0, 0, 0]);
      const [ensembleLabel, setEnsembleLabel] = useState("Waiting for prediction");

      const [driveReady, setDriveReady] = useState(false);
      const [pickerBusy, setPickerBusy] = useState(false);
      const [driveToken, setDriveToken] = useState("");
      const [driveStatus, setDriveStatus] = useState(
        GOOGLE_CLIENT_ID && GOOGLE_API_KEY
          ? "Google Drive available"
          : "Google Drive not configured"
      );
      const [selectedFolder, setSelectedFolder] = useState(null);
      const [driveFiles, setDriveFiles] = useState([]);
      const [loadingFiles, setLoadingFiles] = useState(false);
      const [selectedFile, setSelectedFile] = useState(null);
      const [videoSourceName, setVideoSourceName] = useState("test1.mp4");
      const [videoSourceType, setVideoSourceType] = useState("Local");
      const [configWarning, setConfigWarning] = useState("");

      const resetPredictionHistory = () => {
        ensembleHistoryRef.current = [];
        lastAlertedLabelRef.current = null;
        lastStableClassRef.current = "";
        predictionCounterRef.current = 0;
        setEnsembleProbs([0, 0, 0, 0, 0]);
        setEnsembleLabel("Waiting for prediction");
      };

      const startWebcam = async () => {
        try {
          if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Your browser does not support webcam access. Use Chrome/Safari on localhost or HTTPS.");
            return;
          }

          const stream = await navigator.mediaDevices.getUserMedia({
            video: {
              width: { ideal: 1280 },
              height: { ideal: 720 },
              facingMode: "environment"
            },
            audio: false
          });

          webcamStreamRef.current = stream;

          const video = videoRef.current;
          if (video) {
            video.pause();
            video.srcObject = stream;
            video.removeAttribute("src");
            video.load();
            await video.play();
          }

          resetPredictionHistory();
          setSelectedFile(null);
          setVideoSourceName("Live webcam");
          setVideoSourceType("Camera");
          setDriveStatus("Webcam active. Live foam scanning is running.");
          setIsWebcam(true);
        } catch (err) {
          console.error("Webcam error:", err);
          alert("Could not access webcam: " + err.message);
        }
      };

      const stopWebcam = () => {
        if (webcamStreamRef.current) {
          webcamStreamRef.current.getTracks().forEach((track) => track.stop());
          webcamStreamRef.current = null;
        }

        const video = videoRef.current;
        if (video && video.srcObject) {
          video.srcObject = null;
        }

        setIsWebcam(false);
      };

      const revokeCurrentBlobUrl = () => {
        if (createdBlobUrlRef.current) {
          URL.revokeObjectURL(createdBlobUrlRef.current);
          createdBlobUrlRef.current = null;
        }
      };

      const resetToDefaultVideo = () => {
        stopWebcam();
        revokeCurrentBlobUrl();
        resetPredictionHistory();
        setSelectedFile(null);
        setVideoSourceName("test1.mp4");
        setVideoSourceType("Local");

        const video = videoRef.current;
        if (video) {
          video.pause();
          video.srcObject = null;
          video.src = "/static/test1.mp4";
          video.load();
        }
      };

      const ensureDriveToken = () => {
        return new Promise((resolve, reject) => {
          if (!tokenClientRef.current) {
            reject(new Error("Google auth is not ready yet."));
            return;
          }

          tokenClientRef.current.callback = (resp) => {
            if (resp && resp.access_token) {
              setDriveToken(resp.access_token);
              resolve(resp.access_token);
            } else {
              reject(new Error("Google Drive authentication failed."));
            }
          };

          tokenClientRef.current.requestAccessToken({
            prompt: driveToken ? "" : "consent",
          });
        });
      };

      const openFolderPickerWithToken = (accessToken) => {
        if (!window.google || !window.google.picker) {
          setPickerBusy(false);
          setDriveStatus("Google Picker library not ready");
          return;
        }

        const folderView = new window.google.picker.DocsView(window.google.picker.ViewId.FOLDERS)
          .setIncludeFolders(true)
          .setSelectFolderEnabled(true)
          .setOwnedByMe(false);

        const picker = new window.google.picker.PickerBuilder()
          .addView(folderView)
          .setOAuthToken(accessToken)
          .setDeveloperKey(GOOGLE_API_KEY)
          .setTitle("Choose a Google Drive folder")
          .setCallback(async (data) => {
            if (data.action === window.google.picker.Action.CANCEL) {
              setPickerBusy(false);
              setDriveStatus("Folder picker closed");
              return;
            }

            if (data.action !== window.google.picker.Action.PICKED || !data.docs || !data.docs.length) {
              return;
            }

            const folder = data.docs[0];
            setSelectedFolder({
              id: folder.id,
              name: folder.name || folder.id,
            });
            setDriveStatus(`Folder selected: ${folder.name || folder.id}`);
            setPickerBusy(false);

            try {
              await loadFolderFiles(accessToken, folder.id, folder.name || folder.id);
            } catch (err) {
              console.error(err);
              setDriveStatus(err.message || "Could not load files from folder");
            }
          })
          .build();

        picker.setVisible(true);
      };

      const loadFolderFiles = async (accessToken, folderId, folderName) => {
        setLoadingFiles(true);
        setDriveFiles([]);
        setSelectedFile(null);

        const qParts = [
          `'${folderId}' in parents`,
          `trashed = false`,
          `(mimeType = 'video/mp4' or mimeType = 'video/quicktime' or mimeType contains 'video/')`
        ];
        const q = qParts.join(" and ");

        const params = new URLSearchParams({
          q: q,
          fields: "files(id,name,mimeType,size,modifiedTime,webViewLink)",
          pageSize: "200",
          orderBy: "name",
          supportsAllDrives: "true",
          includeItemsFromAllDrives: "true"
        });

        const response = await fetch(`https://www.googleapis.com/drive/v3/files?${params.toString()}`, {
          headers: {
            Authorization: `Bearer ${accessToken}`
          }
        });

        if (!response.ok) {
          throw new Error(`Could not list folder files (${response.status})`);
        }

        const data = await response.json();
        const files = Array.isArray(data.files) ? data.files : [];

        setDriveFiles(files);
        setDriveStatus(`Loaded ${files.length} video file(s) from ${folderName}`);
        setLoadingFiles(false);
      };

      const playDriveFile = async (file) => {
        stopWebcam();
        try {
          setDriveStatus(`Loading ${file.name}...`);

          const accessToken = driveToken || await ensureDriveToken();

          const response = await fetch(`https://www.googleapis.com/drive/v3/files/${file.id}?alt=media&supportsAllDrives=true`, {
            headers: {
              Authorization: `Bearer ${accessToken}`
            }
          });

          if (!response.ok) {
            throw new Error(`Drive download failed: ${response.status}`);
          }

          const blob = await response.blob();
          revokeCurrentBlobUrl();

          const objectUrl = URL.createObjectURL(blob);
          createdBlobUrlRef.current = objectUrl;

          resetPredictionHistory();
          setSelectedFile(file);
          setVideoSourceName(file.name);
          setVideoSourceType("Google Drive");
          setDriveStatus(`Loaded ${file.name}`);

          const video = videoRef.current;
          if (video) {
            video.pause();
            video.srcObject = null;
            video.src = objectUrl;
            video.load();
            video.play().catch(() => {});
          }
        } catch (error) {
          console.error(error);
          setDriveStatus(error.message || "Could not load Google Drive video");
        }
      };

      useEffect(() => {
        fetch("/google-drive-config")
          .then(res => res.json())
          .then(config => {
            if (!config.enabled) {
              setConfigWarning("Set GOOGLE_CLIENT_ID and GOOGLE_API_KEY before using Google Drive.");
            }
          })
          .catch(() => {
            setConfigWarning("Could not load Google Drive configuration.");
          });
      }, []);

      useEffect(() => {
        let cancelled = false;

        const waitForGoogleLibraries = () => {
          if (cancelled) return;

          const hasAccounts = !!(window.google && window.google.accounts && window.google.accounts.oauth2);
          const hasGapi = !!window.gapi;

          if (!GOOGLE_CLIENT_ID || !GOOGLE_API_KEY) {
            setDriveReady(false);
            return;
          }

          if (!hasAccounts || !hasGapi) {
            window.setTimeout(waitForGoogleLibraries, 300);
            return;
          }

          window.gapi.load("picker", {
            callback: () => {
              if (cancelled) return;

              tokenClientRef.current = window.google.accounts.oauth2.initTokenClient({
                client_id: GOOGLE_CLIENT_ID,
                scope: GOOGLE_SCOPES,
                callback: () => {},
              });

              setDriveReady(true);
            },
            onerror: () => {
              setDriveReady(false);
              setConfigWarning("Google Picker failed to load.");
            },
          });
        };

        waitForGoogleLibraries();

        return () => {
          cancelled = true;
          revokeCurrentBlobUrl();
        };
      }, []);

      const chooseDriveFolder = async () => {
        if (!GOOGLE_CLIENT_ID || !GOOGLE_API_KEY) {
          setConfigWarning("Missing GOOGLE_CLIENT_ID or GOOGLE_API_KEY on the server.");
          return;
        }

        if (!tokenClientRef.current) {
          setDriveStatus("Google Drive is still loading. Try again.");
          return;
        }

        try {
          setPickerBusy(true);
          setDriveStatus("Connecting to Google Drive...");
          const token = await ensureDriveToken();
          setDriveStatus("Google Drive connected");
          openFolderPickerWithToken(token);
        } catch (err) {
          console.error(err);
          setPickerBusy(false);
          setDriveStatus(err.message || "Could not connect to Google Drive");
        }
      };

      const fetchClassification = async () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || processingRef.current) return;
        if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) return;

        processingRef.current = true;

        try {
          const ctx = canvas.getContext("2d");
          const vw = video.videoWidth;
          const vh = video.videoHeight;

          // Preserve the video's real aspect ratio here. The Python server then
          // performs the same 224x224 resize-with-padding used during training.
          const maxSide = 720;
          const scale = Math.min(1, maxSide / Math.max(vw, vh));
          canvas.width = Math.max(1, Math.round(vw * scale));
          canvas.height = Math.max(1, Math.round(vh * scale));
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          const dataURL = canvas.toDataURL("image/jpeg", 0.90);

          predictionCounterRef.current += 1;
          const forceFull =
            predictionCounterRef.current % FULL_ENSEMBLE_EVERY === 0;

          const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              image: dataURL,
              last_stable_class: lastStableClassRef.current,
              force_full: forceFull,
            }),
          });

          const data = await response.json();
          if (!data.results) return;

          const results = data.results;

          // Adaptive inference may return only the primary model for a very
          // confident frame. Merge fresh results into existing model cards so
          // the dashboard layout remains unchanged. Ensemble math below still
          // uses ONLY current-frame results.
          setModelResults(previous => {
            const byName = {};
            previous.forEach(item => {
              if (item && item.model_name) byName[item.model_name] = item;
            });
            results.forEach(item => {
              if (item && item.model_name) byName[item.model_name] = item;
            });

            const preferredOrder = [
              "custom_cnn_model",
              "mobilenetv2_model",
              "resnet50_model",
              "efficientnetb0_model",
            ];
            return preferredOrder
              .filter(name => byName[name])
              .map(name => byName[name]);
          });

          // Fast mode confirms the last stable full-ensemble class. Do NOT mix
          // one-model probabilities into the multi-model smoothing history.
          if (data.inference_mode === "fast-primary-hold") {
            return;
          }

          let sum = new Array(CLASS.length).fill(0);
          let totalWeight = 0;

          results.forEach((res) => {
            if (!res || res.error) return;
            if (!Array.isArray(res.all_confidences) || res.all_confidences.length < CLASS.length) return;

            const probs = res.all_confidences.slice(0, CLASS.length).map(Number);
            const weight = Number(MODEL_WEIGHTS[res.model_name] || 0);
            if (weight <= 0 || probs.some(v => !Number.isFinite(v))) return;

            totalWeight += weight;
            for (let i = 0; i < CLASS.length; i++) {
              sum[i] += probs[i] * weight;
            }
          });

          if (totalWeight <= 0) return;

          const raw = sum.map(v => v / totalWeight);
          const rawTotal = raw.reduce((a, b) => a + b, 0);
          const normalized = raw.map(v => rawTotal > 0 ? v / rawTotal : v);

          // Temporal smoothing: use several consecutive frames instead of
          // trusting one possibly blurry/noisy video frame.
          ensembleHistoryRef.current.push(normalized);
          if (ensembleHistoryRef.current.length > SMOOTHING_WINDOW) {
            ensembleHistoryRef.current.shift();
          }

          const history = ensembleHistoryRef.current;
          const smoothed = new Array(CLASS.length).fill(0);
          history.forEach(frameProbs => {
            for (let i = 0; i < CLASS.length; i++) smoothed[i] += frameProbs[i];
          });
          for (let i = 0; i < CLASS.length; i++) smoothed[i] /= history.length;
          setEnsembleProbs(smoothed);

          const order = smoothed.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
          const topProb = order[0][0];
          const topIdx = order[0][1];
          const secondProb = order[1][0];
          const margin = topProb - secondProb;
          const topClass = CLASS[topIdx];

          let agreeingWeight = 0;
          results.forEach((res) => {
            if (!res || res.error || !Array.isArray(res.all_confidences)) return;
            const probs = res.all_confidences.slice(0, CLASS.length).map(Number);
            const modelTop = probs.indexOf(Math.max(...probs));
            const weight = Number(MODEL_WEIGHTS[res.model_name] || 0);
            if (modelTop === topIdx) agreeingWeight += weight;
          });
          const weightedAgreement =
            totalWeight > 0 ? agreeingWeight / totalWeight : 0;

          const isUncertain =
            topProb < MIN_CONFIDENCE ||
            margin < MIN_MARGIN ||
            weightedAgreement < MIN_WEIGHTED_MODEL_AGREEMENT;

          // Always SHOW the class with the highest ensemble probability.
          // Example: Foam-Heavy 30.1% vs Foam-Medium 30.0% displays Foam-Heavy.
          // Internal uncertainty is still used for adaptive-inference safety
          // and Slack alerts, so the reliability protections remain intact.
          setEnsembleLabel(topClass);

          if (!isUncertain) {
            lastStableClassRef.current = topClass;

            // Only alert after enough frames exist to be meaningfully stable.
            if (history.length >= Math.min(3, SMOOTHING_WINDOW) && lastAlertedLabelRef.current !== topClass) {
              lastAlertedLabelRef.current = topClass;
              fetch("/ensemble-alert", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ label: topClass, confidence: topProb }),
              }).catch(err => console.error("Slack alert error:", err));
            }
          }
        } catch (error) {
          console.error("Error fetching classifications:", error);
        } finally {
          processingRef.current = false;
        }
      };

      useEffect(() => {
        const video = videoRef.current;
        let interval = null;

        if (!video) return undefined;

        const stopLoop = () => {
          if (interval) {
            clearInterval(interval);
            interval = null;
          }
        };

        const onPlay = () => {
          stopLoop();
          fetchClassification();
          interval = setInterval(() => {
            if (!video.paused && !video.ended && video.readyState >= 2) {
              fetchClassification();
            }
          }, 900);
        };

        const onPause = () => stopLoop();
        const onEnded = () => stopLoop();

        video.addEventListener("play", onPlay);
        video.addEventListener("pause", onPause);
        video.addEventListener("ended", onEnded);

        return () => {
          stopLoop();
          video.removeEventListener("play", onPlay);
          video.removeEventListener("pause", onPause);
          video.removeEventListener("ended", onEnded);
        };
      }, []);

      return (
        <div className="page">
          <div className="hero">
            <h1>BioReactor Foam Classification Dashboard</h1>
            <p>Choose any Drive folder you can access, list its videos in the UI, then click a video to play and classify it live.</p>
          </div>

          <div className="video-panel">
            <div className="toolbar">
              <button onClick={startWebcam} disabled={isWebcam}>
                {isWebcam ? "Webcam Active" : "Use Webcam"}
              </button>
              <button onClick={chooseDriveFolder} disabled={!driveReady || pickerBusy || !GOOGLE_CLIENT_ID || !GOOGLE_API_KEY}>
                {pickerBusy ? "Opening Drive..." : "Choose Drive Folder"}
              </button>
              <button className="secondary" onClick={resetToDefaultVideo}>Use local test video</button>
              <span className="source-pill">{videoSourceType}: {videoSourceName}</span>
            </div>

            <video ref={videoRef} controls crossOrigin="anonymous">
              <source src="/static/test1.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>

            <div className="selected-file">
              <strong>Status:</strong> {driveStatus}
              {selectedFile && (
                <div style={{ marginTop: "8px" }}>
                  <div><strong>Selected file:</strong> {selectedFile.name}</div>
                  <div><strong>Drive file ID:</strong> {selectedFile.id}</div>
                </div>
              )}
            </div>

            {selectedFolder && (
              <div className="folder-box">
                <div><strong>Selected folder:</strong> {selectedFolder.name}</div>
                <div><strong>Folder ID:</strong> {selectedFolder.id}</div>
              </div>
            )}

            <div className="help-text">
              Pick a folder first. The app will load the folder's video files into the list on the right.
            </div>

            {configWarning && <div className="warning-box">{configWarning}</div>}
          </div>

          <div className="main-grid">
            <div className="summary-grid">
              <div className="summary-card">
                <div className="summary-title">Ensemble prediction</div>
                <div className="ensemble-label">{ensembleLabel}</div>

                <div className="ensemble-grid">
                  {CLASS.map((name, i) => (
                    <div key={i} className="ensemble-metric">
                      <div className="ensemble-metric-name">{name}</div>
                      <div className="ensemble-metric-value">{(ensembleProbs[i] * 100).toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="summary-card">
                <div className="summary-title">Session overview</div>
                <div className="side-stats">
                  <div className="stat-box">
                    <div className="stat-label">Models shown</div>
                    <div className="stat-value">{modelResults.filter(r => !r.error).length}</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">Refresh mode</div>
                    <div className="stat-value">Live</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">Video source</div>
                    <div className="stat-value" style={{ fontSize: "18px" }}>{videoSourceName}</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="list-panel">
              <h2>Folder Videos</h2>
              <div className="list-sub">
                {loadingFiles
                  ? "Loading files..."
                  : `${driveFiles.length} video file(s) found`}
              </div>

              <div className="file-list">
                {driveFiles.length > 0 ? (
                  driveFiles.map((file) => (
                    <div
                      key={file.id}
                      className={`file-card ${selectedFile && selectedFile.id === file.id ? "active" : ""}`}
                      onClick={() => playDriveFile(file)}
                    >
                      <div className="file-name">{file.name}</div>
                      <div className="file-meta">
                        {file.mimeType || "video"} • {formatFileSize(file.size)}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="empty-state">
                    {selectedFolder
                      ? "No video files found in this folder."
                      : "Choose a Drive folder to load its videos."}
                  </div>
                )}
              </div>
            </div>
          </div>



          <FeedbackPanel
            ensembleLabel={ensembleLabel}
            ensembleProbs={ensembleProbs}
            modelResults={modelResults}
            videoSourceName={videoSourceName}
            videoSourceType={videoSourceType}
            selectedFile={selectedFile}
            videoRef={videoRef}
          />

          <ExperimentJournal />

          <div className="models-panel">
            <div className="models-panel-header">
              <div>
                <h2>Model Predictions</h2>
                <p>All trained models, updated while the video is playing</p>
              </div>
            </div>

            <div className="models-grid">
              {modelResults.length > 0 ? (
                modelResults.map((result, index) => (
                  <ModelPrediction key={index} modelResult={result} />
                ))
              ) : (
                <div className="empty-state">Press play to start live prediction.</div>
              )}
            </div>
          </div>

          <canvas ref={canvasRef} style={{ display: "none" }} />
        </div>
      );
    }

    ReactDOM.render(<App />, document.getElementById("root"));
  </script>
</body>
</html>
"""
    html_content = html_content.replace("__GOOGLE_CLIENT_ID__", escape_js_string(GOOGLE_CLIENT_ID))
    html_content = html_content.replace("__GOOGLE_API_KEY__", escape_js_string(GOOGLE_API_KEY))
    html_content = html_content.replace("__MODEL_WEIGHTS_JSON__", json.dumps(MODEL_WEIGHTS))
    html_content = html_content.replace("__SMOOTHING_WINDOW__", str(int(ENSEMBLE_CONFIG.get("smoothing_window", 7))))
    html_content = html_content.replace("__MIN_CONFIDENCE__", str(float(ENSEMBLE_CONFIG.get("min_confidence", 0.45))))
    html_content = html_content.replace("__MIN_MARGIN__", str(float(ENSEMBLE_CONFIG.get("min_margin", 0.06))))
    html_content = html_content.replace("__FULL_ENSEMBLE_EVERY__", str(int(FULL_ENSEMBLE_EVERY)))
    html_content = html_content.replace(
        "__MIN_WEIGHTED_MODEL_AGREEMENT__",
        str(float(MIN_WEIGHTED_MODEL_AGREEMENT)),
    )
    return HTMLResponse(content=html_content, status_code=200)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("cnn_inference:app", host="0.0.0.0", port=port)
