
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os

app = Flask(__name__)

model_path = os.path.join(
    os.path.dirname(__file__),
    "brain_tumor_model.keras"
)

model = load_model(model_path)

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    # Save uploaded image
    upload_folder = os.path.join(
        os.path.dirname(__file__),
        "static"
    )

    os.makedirs(upload_folder, exist_ok=True)

    image_path = os.path.join(
        upload_folder,
        "uploaded_image.jpg"
    )

    file.save(image_path)

    # Image preprocessing
    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = float(
        prediction[0][predicted_index] * 100
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result</title>

        <style>
            body {{
                font-family: Arial, sans-serif;
                text-align: center;
                background: #f2f2f2;
                padding: 40px;
            }}

            .result {{
                background: white;
                max-width: 650px;
                margin: auto;
                padding: 40px;
                border-radius: 15px;
            }}

            .mri-image {{
                width: 350px;
                height: 350px;
                object-fit: contain;
                margin: 20px;
                border-radius: 10px;
            }}

            .prediction {{
                font-size: 25px;
                font-weight: bold;
            }}

            .confidence {{
                font-size: 20px;
            }}

            a {{
                display: inline-block;
                margin-top: 20px;
                padding: 12px 25px;
                background: black;
                color: white;
                text-decoration: none;
                border-radius: 8px;
            }}
        </style>
    </head>

    <body>

        <div class="result">

            <h1>🧠 Brain Tumor Detection Result</h1>

            <img
                class="mri-image"
                src="/static/uploaded_image.jpg"
            >

            <div class="prediction">
                Prediction: {predicted_class}
            </div>

            <div class="confidence">
                Confidence: {confidence:.2f}%
            </div>

            <a href="/">Upload Another Image</a>

        </div>

    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
