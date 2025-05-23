import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import torch

# 1. Initialize Models
print("[INFO] Loading MTCNN and FaceNet...")
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# 2. Prepare Training Data
X = []
y = []

dataset_path = r"D:\Attendance\Datasets"
classes = os.listdir(dataset_path)

print("[INFO] Processing dataset...")
for label in tqdm(classes):
    person_folder = os.path.join(dataset_path, label)
    if not os.path.isdir(person_folder):
        continue

    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        try:
            img = Image.open(img_path).convert('RGB')
            face = mtcnn(img)

            if face is not None:
                # Get embedding
                with torch.no_grad():
                    embedding = facenet(face.unsqueeze(0)).squeeze().numpy()
                X.append(embedding)
                y.append(label)
            else:
                print(f"[WARNING] No face detected in {img_file}")

        except Exception as e:
            print(f"[ERROR] Failed on {img_file}: {e}")

# 3. Convert to arrays
X = np.array(X)
y = np.array(y)

# 4. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5. Train RandomForest
print("[INFO] Training RandomForest Classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y_encoded)

# 6. Save Model and Encoder
print("[INFO] Saving model and label encoder...")
joblib.dump(rf, "face_rf_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Training complete. Model saved as 'face_rf_model.pkl'")
