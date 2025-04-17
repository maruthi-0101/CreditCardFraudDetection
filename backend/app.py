from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import os
import logging
import traceback
import PyPDF2
import re

# Set project base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")

# Initialize Flask app
app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, "templates"),
    static_folder=os.path.join(FRONTEND_DIR, "static"),
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load trained fraud detection model
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model_8.keras")
model = None

def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found!")
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info("✅ Model loaded successfully.")
    except Exception as e:
        logging.error(f"❌ Error loading model: {traceback.format_exc()}")
        model = None

load_model()

@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        logging.error(f"❌ Template Error: {traceback.format_exc()}")
        return jsonify({"error": "Template not found"}), 500

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        file_extension = file.filename.split(".")[-1].lower()
        extractors = {
            "csv": extract_data_from_csv,
            "txt": extract_data_from_txt,
            "json": extract_data_from_json,
            "xlsx": extract_data_from_excel,
            "xls": extract_data_from_excel,
            "pdf": extract_data_from_pdf
        }

        if file_extension not in extractors:
            return jsonify({"error": f"Unsupported file format: {file_extension}"}), 400

        extracted_data = extractors[file_extension](file)
        extracted_data = validate_and_fix_features(extracted_data)

        logging.info(f"Extracted Features: {extracted_data}")

        input_data = np.array(extracted_data).reshape(1, -1)
        input_data = normalize_data(input_data)

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        prediction = model.predict(input_data)[0][0]
        fraud_threshold = 0.4  # Adjusted for better fraud detection
        response = {
            "message": "⚠️ Fraud Detected!" if prediction > fraud_threshold else "✅ Transaction is Safe",
            "fraud_probability": round(float(prediction) * 100, 2),
        }

        logging.info(f"📂 File processed. Result: {response}")
        return jsonify(response)
    except Exception as e:
        logging.error(f"❌ File Upload Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

# File Extraction Functions

def extract_data_from_pdf(file):
    try:
        extracted_text = ""
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + " "
        numbers = re.findall(r"\b\d+\.?\d*\b", extracted_text)
        numbers = [float(num) for num in numbers]
        return numbers[:30] + [0] * max(0, 30 - len(numbers))
    except Exception as e:
        raise ValueError(f"❌ Failed to extract data from PDF: {str(e)}")

def extract_data_from_csv(file):
    try:
        df = pd.read_csv(file, dtype=str, header=None)
        df = df.applymap(lambda x: x.replace('"', '').strip() if isinstance(x, str) else x)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df.values.flatten().tolist()[:30]
    except Exception as e:
        raise ValueError(f"❌ Failed to extract data from CSV: {str(e)}")

def extract_data_from_txt(file):
    try:
        text = file.read().decode("utf-8")
        return extract_numbers(text)
    except Exception as e:
        raise ValueError(f"❌ Failed to extract data from TXT: {str(e)}")

def extract_data_from_json(file):
    try:
        data = json.load(file)
        values = list(data.values()) if isinstance(data, dict) else data
        numbers = [float(v) for v in values if isinstance(v, (int, float))]
        return numbers[:30] + [0] * max(0, 30 - len(numbers))
    except Exception as e:
        raise ValueError(f"❌ Failed to extract data from JSON: {str(e)}")

def extract_data_from_excel(file):
    try:
        df = pd.read_excel(file, engine="openpyxl", header=None)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df.values.flatten().tolist()[:30]
    except Exception as e:
        raise ValueError(f"❌ Failed to extract data from Excel: {str(e)}")

def extract_numbers(text):
    numbers = [float(n) for n in text.split() if n.replace(".", "", 1).isdigit()]
    return numbers[:30] + [0] * max(0, 30 - len(numbers))

def validate_and_fix_features(data):
    return data[:30] + [0] * max(0, 30 - len(data))

def normalize_data(data):
    mean, std = np.mean(data), np.std(data)
    return np.zeros_like(data) if std == 0 else (data - mean) / std

if __name__ == "__main__":
    logging.info("🚀 Server Started")
    app.run(debug=True)