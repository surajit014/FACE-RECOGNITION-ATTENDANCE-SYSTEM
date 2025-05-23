from flask import Flask, render_template, request, redirect, url_for, session
from PIL import Image
import torch
import joblib
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import base64
from io import BytesIO
from datetime import datetime
import mysql.connector

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session

# Load models
mtcnn = MTCNN(image_size=160, margin=20)
facenet = InceptionResnetV1(pretrained='vggface2').eval()
rf_model = joblib.load("face_rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ID → Name mapping
name_mapping = {
    "001": "Parson 1",
    "002": "parson 2",
    "003": "parson 3",
    "004": "parson 4",
    "005": "parson 5",
    "006": "parson 6",
    "007": "parson 7",
    "008": "parson 8"
}

# ---------- MySQL Utility Functions ----------
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="user_name",
        password="database_password",
        database="database_name"
    )

def insert_entry(name):
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now()
    cur.execute("INSERT INTO attendance (name, entry_time) VALUES (%s, %s)", (name, now))
    conn.commit()
    conn.close()

def check_exit(name):
    conn = get_connection()
    cur = conn.cursor()
    today = datetime.now().date()
    cur.execute("SELECT id FROM attendance WHERE name=%s AND DATE(entry_time)=%s AND exit_time IS NULL", (name, today))
    result = cur.fetchone()
    conn.close()
    return result[0] if result else None

def update_exit(att_id):
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now()
    cur.execute("UPDATE attendance SET exit_time=%s WHERE id=%s", (now, att_id))
    conn.commit()
    conn.close()

# ---------- Flask Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data_url = request.form['image']
    header, encoded = data_url.split(",", 1)
    img_data = base64.b64decode(encoded)

    image = Image.open(BytesIO(img_data)).convert("RGB")
    image_path = os.path.join("static", "captured.jpg")
    image.save(image_path)

    face = mtcnn(image)
    if face is None:
        return render_template("result.html", result="❌ No face detected.", image_path="captured.jpg")

    with torch.no_grad():
        embedding = facenet(face.unsqueeze(0)).numpy()

    pred = rf_model.predict(embedding)
    label_code = label_encoder.inverse_transform(pred)[0]
    actual_name = name_mapping.get(label_code, "Unknown")
    session['name'] = actual_name

    # Check if they have entered today and not exited
    att_id = check_exit(actual_name)

    # Check if any records today (entry already happened)
    conn = get_connection()
    cur = conn.cursor()
    today = datetime.now().date()
    cur.execute("SELECT * FROM attendance WHERE name=%s AND DATE(entry_time)=%s", (actual_name, today))
    records = cur.fetchall()
    conn.close()

    show_entry = False
    show_exit = False

    if not records:
        show_entry = True  # No entry today
    elif att_id:
        show_exit = True   # Entered but not exited
    else:
        show_entry = True  # Entered & exited — allow re-entry

    result_text = f"✅ Predicted ID: {label_code} — Name: {actual_name}"
    return render_template("result.html", result=result_text, image_path="captured.jpg",
                           show_entry=show_entry, show_exit=show_exit, att_id=att_id)

@app.route('/entry', methods=['POST'])
def entry():
    name = session.get('name')
    if name:
        insert_entry(name)
    return redirect(url_for('index'))

@app.route('/exit', methods=['POST'])
def exit():
    att_id = request.form.get("att_id")
    if att_id:
        update_exit(att_id)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
