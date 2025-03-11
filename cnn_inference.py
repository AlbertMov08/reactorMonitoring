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
          }
        .container {
        display: flex;
        flex-direction: column;
        align-items: center; /* optional, centers the items horizontally */
        padding: 20px;
        }

          video {
            border: 1px solid #ccc;
          }
          .classification {
            margin-left: 20px;
            font-size: 1.5em;
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
            const [classification, setClassification] = useState("Waiting for prediction...");
            const classMap = {
              0: "Foam-Heavy",
              1: "Foam-mild",
              2: "Post-Antifoam Addition",
              3: "Foam-Medium",
              4: "No Foam"
            };
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
                  if (result.classification !== undefined) {
                    setClassification(result.classification);
                  } else {
                    setClassification("Error in prediction");
                  }
                } catch (error) {
                  console.error("Error fetching classification:", error);
                  setClassification("Error fetching classification");
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
                                <h2>Current Classification:</h2>
                  <div>{classMap[classification]}</div>
                <video
                  ref={videoRef}
                  width="640"
                  height="480"
                  controls
                  crossOrigin="anonymous"
                >
                  {/*
                    Note the updated source URL to serve the video from /static
                  */}
                  <source src="/static/test.mp4" type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
                <div className="classification">
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

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        image_str = data.get("image")
        if not image_str:
            return JSONResponse({"error": "No image provided"}, status_code=400)
        image_array = preprocess_image(image_str)
        preds = model.predict(image_array)
        predicted_class = int(np.argmax(preds, axis=1)[0])
        return JSONResponse({"classification": predicted_class})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("cnn_inference:app", host="0.0.0.0", port=8001, reload=True)