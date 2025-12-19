import base64
import sqlite3
from io import BytesIO
from datetime import datetime

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, g, render_template
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
# preallocate memory for model
interpreter.allocate_tensors()

# get input/output details from model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print model info, since model is obscure
print(f"Input shape: {input_details[0]['shape']}")
print(f"Input dtype: {input_details[0]['dtype']}")
print(f"Output shape: {output_details[0]['shape']}")
print(f"Output dtype: {output_details[0]['dtype']}")

CLASS_LABELS = ["cat", "dog"]


# --- Database ---

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect("results.db")
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect("results.db")
    db.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            score REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    db.commit()
    db.close()


def save_result(label, score):
    db = get_db()
    db.execute(
        "INSERT INTO classifications (label, score, timestamp) VALUES (?, ?, ?)",
        (label, score, datetime.now().isoformat())
    )
    db.commit()


# --- Image Processing ---

def preprocess_image(image_data):
    """Convert base64 image to model input format."""
    # Decode base64 to image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Resize to model's expected input size
    input_shape = input_details[0]["shape"]
    height, width = input_shape[1], input_shape[2]
    image = image.resize((width, height))

    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def classify_image(image_array):
    """Run inference on preprocessed image."""
    interpreter.set_tensor(input_details[0]["index"], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output[0]  # Remove batch dimension


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()

    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # Remove data URL prefix if present (e.g., "data:image/png;base64,")
    image_data = data["image"]
    if "," in image_data:
        image_data = image_data.split(",")[1]

    # Process and classify
    image_array = preprocess_image(image_data)
    scores = classify_image(image_array)

    # Get prediction
    predicted_index = int(np.argmax(scores))
    label = CLASS_LABELS[predicted_index]
    score = float(scores[predicted_index])

    # Save to database
    save_result(label, score)

    return jsonify({
        "label": label,
        "score": score,
        "all_scores": {CLASS_LABELS[i]: float(scores[i]) for i in range(len(CLASS_LABELS))}
    })


@app.route("/history", methods=["GET"])
def history():
    """Get classification history."""
    db = get_db()
    results = db.execute(
        "SELECT label, score, timestamp FROM classifications ORDER BY id DESC"
    ).fetchall()
    return jsonify([dict(row) for row in results])


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
