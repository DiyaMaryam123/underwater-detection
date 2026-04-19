cat > main.py << 'EOF'
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import base64
from PIL import Image
import io

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
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
    # Read uploaded image
    contents = await file.read()
    img       = Image.open(io.BytesIO(contents))
    img_array = np.array(img)
    
    # Run detection
    results = model.predict(img_array, conf=0.25, verbose=False)[0]
    
    # Draw boxes on image
    annotated = results.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # Convert to base64 to send back to frontend
    _, buffer = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Get detection details
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls)
        conf   = float(box.conf)
        detections.append({
            "class": CLASS_NAMES[cls_id],
            "confidence": round(conf, 3)
        })
    
    return {
        "detections": detections,
        "count": len(detections),
        "image": img_base64
    }
EOF
