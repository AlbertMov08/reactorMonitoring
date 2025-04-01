import io
import base64
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
import tensorflow as tf
import uvicorn

app = FastAPI()

# Allow CORS from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount a static directory for serving files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define class names based on the model training
CLASS_NAMES = ['Foam-Heavy', 'Foam-mild', 'Post-Antifoam Addition', 'Foam-Medium', 'No Foam']

# Load available models
def load_all_models():
    models_dict = {}
    models_dir = "models"
    
    # Get list of all model files (both .h5 and .keras extensions)
    model_files = [f for f in os.listdir(models_dir) if f.endswith(('.h5', '.keras'))]
    
    # Track which models we've already loaded (to avoid duplicates)
    loaded_models = set()
    
    for file in model_files:
        # Get base name without extension
        if file.endswith('.keras'):
            model_name = file[:-6]  # Remove .keras
        else:
            model_name = file[:-3]  # Remove .h5
            
        # Skip if we already loaded this model
        if model_name in loaded_models:
            continue
            
        model_path = os.path.join(models_dir, file)
        
        try:
            # Load model directly as it was saved during training
            model = load_model(model_path, compile=False)
            models_dict[model_name] = model
            loaded_models.add(model_name)
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            # Try other formats or approaches if needed
    
    # Create simulated models only if no real models loaded
    if len(models_dict) == 0:
        print("No models could be loaded. Creating simulated models for testing...")
        models_dict = create_simulated_models()
    
    return models_dict

def create_simulated_models():
    """Create simulated models for UI testing when no real models can be loaded"""
    models = {}
    
    for name in ["custom_cnn_model", "mobilenetv2_model", "vgg16_model", "resnet50_model"]:
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        models[name] = model
        print(f"Created simulated model: {name}")
    
    return models

# Load all available models
models = load_all_models()
print(f"Loaded {len(models)} models: {list(models.keys())}")

IMG_SIZE = (224, 224)

def preprocess_image(image_str: str):
    if image_str.startswith("data:image"):
        header, encoded = image_str.split(",", 1)
    else:
        encoded = image_str
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0  # Normalize to [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        image_str = data.get("image")
        if not image_str:
            return JSONResponse({"error": "No image provided"}, status_code=400)
        
        image_array = preprocess_image(image_str)
        
        # Run prediction with all models
        results = []
        for model_name, model in models.items():
            try:
                preds = model.predict(image_array)
                
                # Get the predicted class and confidence
                predicted_class = int(np.argmax(preds, axis=1)[0])
                confidence = float(preds[0][predicted_class])
                
                # Get all prediction confidences for visualization
                all_confidences = [float(conf) for conf in preds[0]]
                
                result = {
                    "model_name": model_name,
                    "predicted_class": predicted_class,
                    "class_name": CLASS_NAMES[predicted_class],
                    "confidence": confidence,
                    "all_confidences": all_confidences
                }
                results.append(result)
                
                print(f"{model_name} prediction: {CLASS_NAMES[predicted_class]} (class {predicted_class}) with confidence {confidence:.4f}")
            except Exception as e:
                print(f"Error with model {model_name}: {str(e)}")
                results.append({
                    "model_name": model_name,
                    "error": str(e)
                })
        
        return JSONResponse({"results": results})
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def get_home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <title>Multi-Model BioReactor Classification</title>
        <script crossorigin src="https://unpkg.com/react@17/umd/react.development.js"></script>
        <script crossorigin src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
        <script src="https://unpkg.com/babel-standalone@6/babel.min.js"></script>
        <style>
          body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 20px;
          }
          .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
          }
          .models-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
            width: 100%;
          }
          .model-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            width: 300px;
            margin-bottom: 20px.
          }
          .model-name {
            font-weight: bold;
            font-size: 1.2em;
            margin-bottom: 10px;
            color: #333;
            text-transform: capitalize.
          }
          .class-name {
            font-size: 1.4em;
            margin: 10px 0;
            font-weight: bold;
            color: #2196F3.
          }
          .confidence-bar {
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            margin: 8px 0;
            overflow: hidden;
            position: relative.
          }
          .confidence-fill {
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.3s ease.
          }
          .confidence-label {
            position: absolute;
            right: 5px;
            top: 0;
            color: #000;
            font-size: 12px;
            line-height: 20px;
            z-index: 1.
          }
          .video-container {
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1).
          }
          video {
            display: block;
            max-width: 100%.
          }
          .header {
            text-align: center;
            margin-bottom: 20px.
          }
          .label {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 3px.
          }
          .all-confidences {
            margin-top: 15px.
          }
          .class-confidence {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px.
          }
          .error-message {
            color: #f44336;
            font-style: italic.
          }
        </style>
      </head>
      <body>
        <div id="root"></div>
        <script type="text/babel">
          const { useRef, useState, useEffect } = React;
          
          function ModelPrediction({ modelResult }) {
            if (modelResult.error) {
              return (
                <div className="model-card">
                  <div className="model-name">{modelResult.model_name.replace(/_/g, " ")}</div>
                  <div className="error-message">Error: {modelResult.error}</div>
                </div>
              );
            }
            
            return (
              <div className="model-card">
                <div className="model-name">{modelResult.model_name.replace(/_/g, " ")}</div>
                <div className="class-name">{modelResult.class_name}</div>
                
                <div className="label">Confidence:</div>
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill" 
                    style={{width: `${modelResult.confidence * 100}%`}}
                  ></div>
                  <div className="confidence-label">{(modelResult.confidence * 100).toFixed(1)}%</div>
                </div>
                
                <div className="all-confidences">
                  <div className="label">All Classes:</div>
                  {modelResult.all_confidences && modelResult.all_confidences.map((conf, idx) => (
                    <div key={idx} className="class-confidence">
                      <span>{['Foam-Heavy', 'Foam-mild', 'Post-Antifoam Addition', 'Foam-Medium', 'No Foam'][idx]}</span>
                      <span>{(conf * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            );
          }
          
          function App() {
            const videoRef = useRef(null);
            const canvasRef = useRef(null);
            const [modelResults, setModelResults] = useState([]);
            const [isProcessing, setIsProcessing] = useState(false);
            
            const fetchClassification = async () => {
              const video = videoRef.current;
              const canvas = canvasRef.current;
              
              if (video && canvas && !isProcessing) {
                setIsProcessing(true);
                
                const context = canvas.getContext("2d");
                canvas.width = 224;
                canvas.height = 224;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const dataURL = canvas.toDataURL("image/png");
                
                try {
                  const response = await fetch("http://localhost:8001/predict", {
                    method: "POST",
                    headers: {
                      "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ image: dataURL }),
                  });
                  
                  const data = await response.json();
                  
                  if (data.results) {
                    setModelResults(data.results);
                  } else {
                    console.error("Invalid response format:", data);
                  }
                } catch (error) {
                  console.error("Error fetching classifications:", error);
                } finally {
                  setIsProcessing(false);
                }
              }
            };
            
            useEffect(() => {
              const video = videoRef.current;
              let interval;
              
              if (video) {
                // Listen for when video is playing
                video.addEventListener('play', () => {
                  // Set an interval to fetch classifications every 500ms while the video is playing
                  interval = setInterval(() => {
                    if (!video.paused && !video.ended) {
                      fetchClassification();
                    }
                  }, 500);
                });
                
                // Clean up when video pauses or ends
                video.addEventListener('pause', () => {
                  clearInterval(interval);
                });
                
                video.addEventListener('ended', () => {
                  clearInterval(interval);
                });
              }
              
              return () => {
                clearInterval(interval);
                if (video) {
                  video.removeEventListener('play', () => {});
                  video.removeEventListener('pause', () => {});
                  video.removeEventListener('ended', () => {});
                }
              };
            }, []);
            
            return (
              <div className="container">
                <div className="header">
                  <h1>Multi-Model BioReactor Classification</h1>
                  <p>Comparing predictions from all available models</p>
                </div>
                
                <div className="video-container">
                  <video
                    ref={videoRef}
                    width="640"
                    height="480"
                    controls
                    crossOrigin="anonymous"
                  >
                    <source src="/static/test.mp4" type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
                </div>
                
                <div className="models-container">
                  {modelResults.length > 0 ? (
                    modelResults.map((result, index) => (
                      <ModelPrediction key={index} modelResult={result} />
                    ))
                  ) : (
                    <p>Press play to see model predictions</p>
                  )}
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
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    uvicorn.run("cnn_inference:app", host="0.0.0.0", port=8001, reload=True)