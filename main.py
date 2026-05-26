from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tflite_runtime.interpreter as tflite
import librosa
import numpy as np
import io
app = FastAPI()
# This is critical—it allows your Base44 app to talk to Render securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# 1. Load your lightweight TFLite model
interpreter = tflite.Interpreter(model_path="shot_timer_model.tflite")
interpreter.allocate_tensors()
# Get the exact input and output structure the model expects
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
labels = ["Gunshot", "Reload", "Slide Battery"]
@app.post("/detect")
async def detect_sound(file: UploadFile = File(...)):
    # 2. Process incoming audio data from Base44
    content = await file.read()
    audio, sr = librosa.load(io.BytesIO(content), sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0).astype(np.float32).reshape(1, -1)
    
    # 3. Pass the audio data into the TFLite brain
    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    
    # 4. Grab the final prediction result
