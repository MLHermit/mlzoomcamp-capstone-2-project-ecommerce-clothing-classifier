from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import onnx
import numpy as np
import uvicorn
from PIL import Image

model_onnx = onnx.load('clothing_classifier.onnx')
ort_session = ort.InferenceSession('clothing_classifier.onnx')

app = FastAPI()

# Preprocess the image to match model input requirements: convert to grayscale and resize to 28x28, normalize pixel values
def process_image(image: Image.Image):
    image = image.convert('L').resize((28, 28))
    image_norm = np.array(image).astype(np.float32)/ 255.0
    image_shaped = image_norm.reshape(1, 1, 28, 28)
    return image_shaped

@app.get('/greet')
def greetings():
    return "hello welcome, let's do some classification"

@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    input_data = process_image(image)
    
    inputs = {ort_session.get_inputs()[0].name: input_data}
    outputs = ort_session.run(None, inputs)
    
    predicted_class = np.argmax(outputs[0], axis=1)[0]
    
    return {'predicted_class': int(predicted_class)}