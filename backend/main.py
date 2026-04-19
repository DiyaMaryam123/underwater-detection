from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import base64
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")

CLASS_NAMES = [
    "animal_fish", "animal_starfish", "animal_eel",
    "trash_bag", "trash_container", "trash_can", "trash_branch"
]

@app.get("/")
def home():
    return {"status": "Underwater Detection API running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img_array = np.array(img)
    results = model.predict(img_array, conf=0.25, verbose=False)[0]
    annotated = results.plot()
    _, buffer = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        detections.append({"class": CLASS_NAMES[cls_id], "confidence": round(conf, 3)})
    return {"detections": detections, "count": len(detections), "image": img_base64}
