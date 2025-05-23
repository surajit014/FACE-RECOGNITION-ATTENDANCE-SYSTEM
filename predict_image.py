from PIL import Image
import torch
import joblib
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# --- CONFIG ---
image_path = "test image\image.jpg"           # 🔁 Change this to your test image path
model_path = "face_rf_model.pkl"
encoder_path = "label_encoder.pkl"

# --- LOAD MODELS ---
print("[INFO] Loading models...")
mtcnn = MTCNN(image_size=160, margin=20)
facenet = InceptionResnetV1(pretrained='vggface2').eval()
rf_model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# --- LOAD IMAGE ---
print("[INFO] Reading image...")
img = Image.open(image_path).convert('RGB')

# --- DETECT FACE ---
print("[INFO] Detecting face...")
face = mtcnn(img)

if face is None:
    print("❌ No face detected.")
else:
    # --- EMBEDDING ---
    with torch.no_grad():
        embedding = facenet(face.unsqueeze(0)).numpy()

    # --- PREDICT CLASS ---
    prediction = rf_model.predict(embedding)
    class_name = label_encoder.inverse_transform(prediction)[0]

    print(f"✅ Predicted class: {class_name}")
