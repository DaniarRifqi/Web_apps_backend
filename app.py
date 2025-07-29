from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io
import mysql.connector
from datetime import datetime
import os
from mysql.connector import Error

MODEL_PATH = "efficientnetv2b1_model.h5"
class_names = ["basah", "kering", "sedang"]
THRESHOLD = 0.47

# Load the model and ensure it's loaded correctly
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

app = Flask(__name__)
CORS(app, origins=["https://web-apps-frontend-bay.vercel.app/"])
@app.route('/')
def index():
    return jsonify({"status": "success", 'message': 'Welcome to the Dry Apple Detection API'}), 200

# --- MySQL Config ---
DB_CONFIG = {
    'host': os.getenv('MYSQLHOST'),
    'user': os.getenv('MYSQLUSER'),
    'password': os.getenv('MYSQLPASSWORD'),
    'database': os.getenv('MYSQLDATABASE'),
    'port': int(os.getenv('MYSQLPORT'))
}

def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as err:
        print(f"Database connection failed: {err}")
        raise

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE NOT NULL,
            image VARCHAR(255) NOT NULL,
            type VARCHAR(20) NOT NULL,
            confidence FLOAT NOT NULL
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

init_db()

@app.route('/api/predict/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        # Open and preprocess image
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))  # Resize to match EfficientNetV2 input size
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)  # EfficientNetV2B1 preprocessing

        # Make prediction
        pred = model.predict(img)
        confidence = float(np.max(pred))
        class_idx = int(np.argmax(pred, axis=1)[0])

        # Handle prediction based on confidence threshold
        if confidence < THRESHOLD:
            label = "Tidak terdeteksi"
        else:
            label = class_names[class_idx]

        # --- Simpan file gambar ke folder uploads ---
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        filename = datetime.now().strftime('%Y%m%d%H%M%S_') + file.filename
        filepath = os.path.join(uploads_dir, filename)
        file.stream.seek(0)
        file.save(filepath)

        # --- Simpan history ke database ---
        if label != "Tidak terdeteksi":
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO history (date, image, type, confidence) VALUES (%s, %s, %s, %s)",
                (datetime.now().date(), filename, label.capitalize(), confidence * 100)
            )
            conn.commit()
            cursor.close()
            conn.close()

        return jsonify({
            'label': label,
            'confidence': confidence * 100,
            'image': '/uploads/' + filename
        })

    except Exception as e:
        return jsonify({'error': f"Error processing image: {str(e)}"}), 500

@app.route('/api/history/', methods=['GET'])
def get_history():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM history ORDER BY id DESC LIMIT 100")
        rows = cursor.fetchall()
        for row in rows:
            row['image'] = '/uploads/' + row['image']
            row['confidence'] = float(row['confidence'])
        cursor.close()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<int:id>', methods=['DELETE'])
def delete_history(id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        # Ambil nama file gambar sebelum hapus
        cursor.execute('SELECT image FROM history WHERE id = %s', (id,))
        row = cursor.fetchone()
        if not row:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Data not found'}), 404
        filename = row['image']
        # Hapus data dari database
        cursor.execute('DELETE FROM history WHERE id = %s', (id,))
        conn.commit()
        cursor.close()
        conn.close()
        # Hapus file gambar dari uploads jika ada
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        file_path = os.path.join(uploads_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    return send_from_directory(uploads_dir, filename)

if __name__ == '__main__':
    app.run()
