import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import MobileNetV2
import uvicorn

app = FastAPI()

# Allow CORS from any origin (adjust if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount a static directory for serving files (like test.mp4)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define class names based on the model training
CLASS_NAMES = ['Foam-Heavy', 'Foam-mild', 'Post-Antifoam Addition', 'Foam-Medium', 'No Foam']

# Custom load function for MobileNetV2 model
def load_mobilenetv2_model(num_classes=5):
    # Recreate the model architecture
    input_shape = (224, 224, 3)
    input_tensor = Input(shape=input_shape)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=predictions)
    
    # Load weights from the saved file
    try:
        model.load_weights("models/mobilenetv2_postdataaug_model.h5")
        print("Successfully loaded model weights")
    except:
        print("Failed to load model weights directly")
        # If direct weight loading fails, we could try more complex weight loading here
    
    return model

# Load the model using our custom function
model = load_mobilenetv2_model()

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

@app.get("/", response_class=HTMLResponse)
async def get_home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <title>Interactive for BioReactor Classification</title>
        <script crossorigin src="https://unpkg.com/react@17/umd/react.development.js"></script>
        <script crossorigin src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
        <script src="https://unpkg.com/babel-standalone@6/babel.min.js"></script>
        <style>
          body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
          }
          .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
          }
          .result-container {
            margin: 20px 0;
            padding: 15px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            width: 100%;
            text-align: center.
          }
          .confidence-bar {
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            margin: 10px 0;
            overflow: hidden;
          }
          .confidence-fill {
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.3s ease;
          }
          video {
            border: 1px solid #ccc;
            border-radius: 8px;
            max-width: 100%;
          }
          h2 {
            color: #333;
          }
          .classification-name {
            font-size: 1.5em;
            font-weight: bold;
            color: #2196F3;
          }
        </style>
      </head>
      <body>
        <div id="root"></div>
        <script type="text/babel">
          const { useRef, useState, useEffect } = React;
          function App() {
            const videoRef = useRef(null);
            const canvasRef = useRef(null);
            const [predictionResult, setPredictionResult] = useState({
              className: "Waiting for prediction...",
              confidence: 0,
              topPredictions: []
            });
            
            const fetchClassification = async () => {
              const video = videoRef.current;
              const canvas = canvasRef.current;
              if (video && canvas) {
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
                  const result = await response.json();
                  if (result.class_name) {
                    setPredictionResult({
                      className: result.class_name,
                      confidence: result.confidence,
                      topPredictions: result.top_predictions || []
                    });
                  } else {
                    setPredictionResult({
                      className: "Error in prediction",
                      confidence: 0,
                      topPredictions: []
                    });
                  }
                } catch (error) {
                  console.error("Error fetching classification:", error);
                  setPredictionResult({
                    className: "Error fetching classification",
                    confidence: 0,
                    topPredictions: []
                  });
                }
              }
            };

            useEffect(() => {
              const video = videoRef.current;
              if (video) {
                video.addEventListener("timeupdate", fetchClassification);
              }
              return () => {
                if (video) {
                  video.removeEventListener("timeupdate", fetchClassification);
                }
              };
            }, []);

            return (
              <div className="container">
                <h1>BioReactor Foam Monitoring</h1>
                
                <div className="result-container">
                  <h2>Current Classification:</h2>
                  <div className="classification-name">{predictionResult.className}</div>
                  <div>Confidence: {(predictionResult.confidence * 100).toFixed(1)}%</div>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill" 
                      style={{width: `${predictionResult.confidence * 100}%`}}
                    ></div>
                  </div>
                  
                  {predictionResult.topPredictions.length > 0 && (
                    <div>
                      <h3>Top Predictions:</h3>
                      {predictionResult.topPredictions.map((pred, index) => (
                        <div key={index}>
                          {pred.name}: {(pred.confidence * 100).toFixed(1)}%
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                
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

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        image_str = data.get("image")
        if not image_str:
            return JSONResponse({"error": "No image provided"}, status_code=400)
        
        image_array = preprocess_image(image_str)
        preds = model.predict(image_array)
        
        # Get the predicted class and confidence
        predicted_class = int(np.argmax(preds, axis=1)[0])
        confidence = float(preds[0][predicted_class])
        
        # Get the top 3 predictions for debugging
        top_indices = np.argsort(preds[0])[-3:][::-1]
        top_predictions = [
            {"class": int(idx), 
             "name": CLASS_NAMES[idx],
             "confidence": float(preds[0][idx])}
            for idx in top_indices
        ]
        
        print(f"Prediction: {CLASS_NAMES[predicted_class]} (class {predicted_class}) with confidence {confidence:.4f}")
        
        return JSONResponse({
            "classification": predicted_class,
            "confidence": confidence,
            "class_name": CLASS_NAMES[predicted_class],
            "top_predictions": top_predictions
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("cnn_inference:app", host="0.0.0.0", port=8001, reload=True)