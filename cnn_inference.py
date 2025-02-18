# main.py
import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # You can also use ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load your CNN model (make sure the model path is correct)
model = load_model("models/custom_cnn_model.h5",compile=False)

# Example: assume the model expects 224x224 RGB images normalized to [0, 1]
IMG_SIZE = (224, 224)

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        image_str = data.get("image")
        if image_str is None:
            return JSONResponse({"error": "No image provided"}, status_code=400)

        # Remove the data URL header if present (e.g. "data:image/png;base64,")
        if image_str.startswith("data:image"):
            header, encoded = image_str.split(",", 1)
        else:
            encoded = image_str

        # Decode the image
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess the image for your CNN (resize, normalize, etc.)
        image = image.resize(IMG_SIZE)
        image_array = np.array(image) / 255.0  # Normalize to [0,1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Run the model prediction
        preds = model.predict(image_array)
        # For example, if using softmax output:
        predicted_class = int(np.argmax(preds, axis=1)[0])

        return JSONResponse({"classification": predicted_class})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)