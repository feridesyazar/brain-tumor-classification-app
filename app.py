import json
import os
from pathlib import Path
from threading import Lock

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "brain_tumor_mri_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

model = None
class_names = None
model_lock = Lock()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def load_model_and_classes():
    global model, class_names

    if model is not None and class_names is not None:
        return model, class_names

    with model_lock:
        if model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    "brain_tumor_mri_model.keras was not found in the model folder."
                )

            model = keras.models.load_model(
                MODEL_PATH,
                compile=False
            )

        if class_names is None:
            if not CLASS_NAMES_PATH.exists():
                raise FileNotFoundError(
                    "class_names.json was not found in the model folder."
                )

            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
                class_names = json.load(file)

    return model, class_names


def prepare_image(image, loaded_model):
    input_shape = loaded_model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = int(input_shape[1] or 224)
    width = int(input_shape[2] or 224)

    image = image.convert("RGB")
    image = image.resize((width, height))

    image_array = keras.utils.img_to_array(image)
    image_array = image_array.astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(image):
    loaded_model, labels = load_model_and_classes()

    image_array = prepare_image(image, loaded_model)

    predictions = loaded_model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))
    predicted_label = labels[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    probabilities = []

    for index, probability in enumerate(predictions):
        probabilities.append({
            "label": labels[index],
            "probability": round(float(probability) * 100, 2)
        })

    probabilities = sorted(
        probabilities,
        key=lambda item: item["probability"],
        reverse=True
    )

    return {
        "label": predicted_label,
        "confidence": round(confidence, 2),
        "probabilities": probabilities
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("image")

    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({
            "error": "Please select an MRI image."
        }), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({
            "error": "Only PNG, JPG and JPEG files are supported."
        }), 400

    try:
        image = Image.open(uploaded_file.stream)
        image.load()

        result = predict_image(image)

        return jsonify(result)

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "error": "The uploaded file is not a valid image."
        }), 400

    except FileNotFoundError as error:
        app.logger.exception("Model file error")

        return jsonify({
            "error": str(error)
        }), 503

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction could not be completed."
        }), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok"
    })


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "The image is too large. Maximum file size is 10 MB."
    }), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )