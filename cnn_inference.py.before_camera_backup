# cnn_inference.py
import io
import base64
import os
from datetime import datetime

import numpy as np
import requests
import tensorflow as tf
import uvicorn
from PIL import Image

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input

from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

print("[BOOT] cnn_inference.py imported")

os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("static", exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

CLASS_NAMES = ['Foam-Heavy', 'Foam-mild', 'Post-Antifoam Addition', 'Foam-Medium', 'No Foam']

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/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")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "1003881011936-ntgdn3d28kbbn5si56fkmmfppakst7bg.apps.googleusercontent.com")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCKS54ZwDki5N4XyauyEind0OZ8-O6r-uM")

last_label_by_model = {}
IMG_SIZE = (224, 224)


def escape_js_string(value: str) -> str:
    if value is None:
        return ""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def send_slack_alert(model_name: str, previous_label: str, new_label: str, confidence: float):
    if not SLACK_WEBHOOK_URL:
        return

    prev_text = previous_label if previous_label is not None else "None"

    text = (
        "Foam Stage Change Detected\n"
        f"- Model: {model_name}\n"
        f"- Previous stage: {prev_text}\n"
        f"- New stage: {new_label}\n"
        f"- Confidence: {confidence * 100:.1f}%\n"
        f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=5)
        print(f"[SLACK] status={resp.status_code}")
        resp.raise_for_status()
    except Exception as e:
        print(f"[SLACK] Error sending Slack alert: {e}")


def create_mobilenetv2_model(input_shape=(224, 224, 3), num_classes=5):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = Sequential([
        Input(shape=input_shape),
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def create_resnet50_model(input_shape=(224, 224, 3), num_classes=5):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


def create_efficientnetb0_model(input_shape=(224, 224, 3), num_classes=5):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = Sequential([
        Input(shape=input_shape),
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def create_custom_cnn_model(input_shape=(224, 224, 3), num_classes=5):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def create_simulated_models():
    models_dict = {}
    for name in ["mobilenetv2_model", "resnet50_model", "efficientnetb0_model"]:
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        models_dict[name] = model
        print(f"Created simulated model: {name}")
    return models_dict


def load_all_models():
    models_dict = {}
    models_dir = "models"

    model_defs = {
        "mobilenetv2_model": create_mobilenetv2_model,
        "resnet50_model": create_resnet50_model,
        "efficientnetb0_model": create_efficientnetb0_model,
        "custom_cnn_model": create_custom_cnn_model,
    }

    for model_name, create_func in model_defs.items():
        keras_paths = [
            os.path.join(models_dir, f"{model_name}.keras"),
            os.path.join(models_dir, f"{model_name}_best.keras"),
        ]
        h5_paths = [
            os.path.join(models_dir, f"{model_name}.h5"),
        ]

        loaded = False

        for kp in keras_paths:
            if os.path.exists(kp):
                try:
                    models_dict[model_name] = load_model(kp, compile=False)
                    print(f"Loaded full model (.keras): {model_name} from {os.path.basename(kp)}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"Could not load .keras for {model_name} from {os.path.basename(kp)}: {e}")

        if loaded:
            continue

        for hp in h5_paths:
            if os.path.exists(hp):
                try:
                    models_dict[model_name] = load_model(hp, compile=False)
                    print(f"Loaded full model (.h5): {model_name} from {os.path.basename(hp)}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"Could not load .h5 for {model_name} from {os.path.basename(hp)}: {e}")

        if loaded:
            continue

        for weights_path in (keras_paths + h5_paths):
            if os.path.exists(weights_path):
                try:
                    model = create_func()
                    model.load_weights(weights_path)
                    models_dict[model_name] = model
                    print(f"Loaded weights into fresh model: {model_name} from {os.path.basename(weights_path)}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"Could not load weights for {model_name} from {os.path.basename(weights_path)}: {e}")

        if not loaded:
            print(f"WARNING: No usable file found for {model_name} in {models_dir}/")

    if len(models_dict) == 0:
        print("No models could be loaded. Creating simulated models for testing...")
        models_dict = create_simulated_models()

    return models_dict


models = load_all_models()
print(f"Loaded {len(models)} models: {list(models.keys())}")


def preprocess_for_model(model_name: str, image_str: str):
    if image_str.startswith("data:image"):
        _, encoded = image_str.split(",", 1)
    else:
        encoded = image_str

    image_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)

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
    return JSONResponse({
        "clientId": GOOGLE_CLIENT_ID,
        "apiKey": GOOGLE_API_KEY,
        "enabled": bool(GOOGLE_CLIENT_ID and GOOGLE_API_KEY),
    })


@app.post("/predict")
async def predict(request: Request):
    print("[PREDICT] /predict called")
    try:
        data = await request.json()
        image_str = data.get("image")
        if not image_str:
            return JSONResponse({"error": "No image provided"}, status_code=400)

        results = []

        for model_name, model in models.items():
            try:
                image_array = preprocess_for_model(model_name, image_str)
                preds = model.predict(image_array, verbose=0)

                if isinstance(preds, list):
                    preds = preds[0]

                predicted_class = int(np.argmax(preds, axis=1)[0])
                confidence = float(preds[0][predicted_class])
                all_confidences = [float(conf) for conf in preds[0]]

                class_label = CLASS_NAMES[predicted_class]
                previous_label = last_label_by_model.get(model_name)

                if previous_label is None or class_label != previous_label:
                    send_slack_alert(
                        model_name=model_name,
                        previous_label=previous_label,
                        new_label=class_label,
                        confidence=confidence,
                    )

                last_label_by_model[model_name] = class_label

                results.append({
                    "model_name": model_name,
                    "predicted_class": predicted_class,
                    "class_name": class_label,
                    "confidence": confidence,
                    "all_confidences": all_confidences
                })

                print(
                    f"{model_name} prediction: {class_label} "
                    f"(class {predicted_class}) with confidence {confidence:.4f}"
                )

            except Exception as e:
                print(f"Error with model {model_name}: {str(e)}")
                results.append({"model_name": model_name, "error": str(e)})

        return JSONResponse({"results": results})

    except Exception as e:
        print(f"[PREDICT] Prediction error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


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

    @media (max-width: 1100px) {
      .main-grid,
      .summary-grid {
        grid-template-columns: 1fr;
      }
      .ensemble-grid {
        grid-template-columns: repeat(2, 1fr);
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
    }
  </style>
</head>
<body>
  <div id="root"></div>

  <script type="text/babel">
    const { useRef, useState, useEffect } = React;

    const CLASS = ['Foam-Heavy','Foam-mild','Post-Antifoam Addition','Foam-Medium','No Foam'];

    const MODEL_WEIGHTS = {
      efficientnetb0_model: 0.40,
      resnet50_model: 0.30,
      mobilenetv2_model: 0.20,
      custom_cnn_model: 0.10,
    };

    const GOOGLE_CLIENT_ID = "__GOOGLE_CLIENT_ID__";
    const GOOGLE_API_KEY = "__GOOGLE_API_KEY__";
    const GOOGLE_SCOPES = "https://www.googleapis.com/auth/drive.readonly";

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

    function App() {
      const videoRef = useRef(null);
      const canvasRef = useRef(null);
      const tokenClientRef = useRef(null);
      const createdBlobUrlRef = useRef(null);

      const [modelResults, setModelResults] = useState([]);
      const [ensembleProbs, setEnsembleProbs] = useState([0, 0, 0, 0, 0]);
      const [ensembleLabel, setEnsembleLabel] = useState("Waiting for prediction");
      const [isProcessing, setIsProcessing] = useState(false);

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

      const revokeCurrentBlobUrl = () => {
        if (createdBlobUrlRef.current) {
          URL.revokeObjectURL(createdBlobUrlRef.current);
          createdBlobUrlRef.current = null;
        }
      };

      const resetToDefaultVideo = () => {
        revokeCurrentBlobUrl();
        setSelectedFile(null);
        setVideoSourceName("test1.mp4");
        setVideoSourceType("Local");

        const video = videoRef.current;
        if (video) {
          video.pause();
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

          setSelectedFile(file);
          setVideoSourceName(file.name);
          setVideoSourceType("Google Drive");
          setDriveStatus(`Loaded ${file.name}`);

          const video = videoRef.current;
          if (video) {
            video.pause();
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
        if (!video || !canvas || isProcessing) return;
        if (video.readyState < 2) return;

        setIsProcessing(true);

        const ctx = canvas.getContext("2d");
        canvas.width = 224;
        canvas.height = 224;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL("image/png");

        try {
          const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataURL }),
          });

          const data = await response.json();
          if (!data.results) return;

          const results = data.results;
          setModelResults(results);

          let sum = new Array(CLASS.length).fill(0);
          let totalWeight = 0;

          results.forEach((res) => {
            if (!res || res.error) return;

            let probs = null;
            if (Array.isArray(res.all_confidences) && res.all_confidences.length) {
              probs = res.all_confidences.slice(0, CLASS.length).map(Number);
            }

            if (!probs || probs.every(v => Number.isNaN(v) || v === 0)) {
              const arr = new Array(CLASS.length).fill(0);
              if (typeof res.predicted_class === "number" && res.predicted_class >= 0 && res.predicted_class < CLASS.length) {
                const c = typeof res.confidence === "number" ? res.confidence : 1;
                arr[res.predicted_class] = c;
              }
              probs = arr;
            }

            const weight = MODEL_WEIGHTS[res.model_name] || 0;
            if (weight <= 0) return;

            totalWeight += weight;
            for (let i = 0; i < CLASS.length; i++) {
              sum[i] += probs[i] * weight;
            }
          });

          if (totalWeight > 0) {
            const avg = sum.map(v => v / totalWeight);
            setEnsembleProbs(avg);

            let topIdx = 0;
            for (let i = 1; i < CLASS.length; i++) {
              if (avg[i] > avg[topIdx]) topIdx = i;
            }
            setEnsembleLabel(CLASS[topIdx]);
          }
        } catch (error) {
          console.error("Error fetching classifications:", error);
        } finally {
          setIsProcessing(false);
        }
      };

      useEffect(() => {
        const video = videoRef.current;
        let interval;

        if (video) {
          const onPlay = () => {
            interval = setInterval(() => {
              if (!video.paused && !video.ended) {
                fetchClassification();
              }
            }, 700);
          };

          const onPause = () => clearInterval(interval);
          const onEnded = () => clearInterval(interval);

          video.addEventListener("play", onPlay);
          video.addEventListener("pause", onPause);
          video.addEventListener("ended", onEnded);

          return () => {
            clearInterval(interval);
            video.removeEventListener("play", onPlay);
            video.removeEventListener("pause", onPause);
            video.removeEventListener("ended", onEnded);
          };
        }
      }, [isProcessing]);

      return (
        <div className="page">
          <div className="hero">
            <h1>BioReactor Foam Classification Dashboard</h1>
            <p>Choose any Drive folder you can access, list its videos in the UI, then click a video to play and classify it live.</p>
          </div>

          <div className="video-panel">
            <div className="toolbar">
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
    return HTMLResponse(content=html_content, status_code=200)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("cnn_inference:app", host="0.0.0.0", port=port)