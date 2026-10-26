from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ======================================
# Flask App
# ======================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ======================================
# Load Model
# ======================================

model = tf.keras.models.load_model("model/brain_tumor_efficientnetb0_finetuned.keras")

print("\nMODEL LOADED SUCCESSFULLY")


# ======================================
# Classes
# ======================================

classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ======================================
# Grad-CAM
# ======================================

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name
):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array,
            training=False
        )

        predicted_index = tf.argmax(
            predictions[0]
        )

        class_channel = predictions[
            :, predicted_index
        ]

    # Calculate gradients
    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    # Average gradients
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # Multiply feature maps by importance
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # Keep only positive importance
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize heatmap
    max_heat = tf.reduce_max(
        heatmap
    )

    if max_heat != 0:
        heatmap = heatmap / max_heat

    return heatmap.numpy()


# ======================================
# Home Page
# ======================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ======================================
# Prediction
# ======================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # Check uploaded file
    if "file" not in request.files:

        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":

        return "No file selected"

    # ==================================
    # Save uploaded image
    # ==================================

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)


    # ==================================
    # Prepare image
    # ==================================

    img = image.load_img(
        filepath,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(
        img
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )
   
    

    # ==================================
    # Prediction
    # ==================================

    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    predicted_class = classes[
        predicted_index
    ]

    confidence = round(
        float(
            prediction[0][predicted_index]
        ) * 100,
        2
    )

    print("\nPrediction:", predicted_class)
    print("Confidence:", confidence, "%")


    # ==================================
    # Grad-CAM
    # ==================================

    # Using the final activated convolutional
    # layer of EfficientNetB0

    last_conv_layer_name = "block4c_add"

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name
    )


    # ==================================
    # Read original image
    # ==================================

    original = cv2.imread(
        filepath
    )

    original = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )


    # ==================================
    # Resize heatmap
    # ==================================

    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )


    # ==================================
    # Convert heatmap
    # ==================================

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )


    # ==================================
    # Overlay
    # ==================================

    overlay = cv2.addWeighted(
        original,
        0.70,
        colored_heatmap,
        0.30,
        0
    )


    # ==================================
    # Save results
    # ==================================

    heatmap_path = os.path.join(
        app.config["RESULT_FOLDER"],
        "heatmap.jpg"
    )

    overlay_path = os.path.join(
        app.config["RESULT_FOLDER"],
        "overlay.jpg"
    )

    cv2.imwrite(
        heatmap_path,
        cv2.cvtColor(
            colored_heatmap,
            cv2.COLOR_RGB2BGR
        )
    )

    cv2.imwrite(
        overlay_path,
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )


    # ==================================
    # Add cache-busting parameter
    # ==================================

    import time

    version = int(
        time.time()
    )

    heatmap_url = (
        heatmap_path.replace("\\", "/")
        + "?v="
        + str(version)
    )

    overlay_url = (
        overlay_path.replace("\\", "/")
        + "?v="
        + str(version)
    )


    # ==================================
    # Display results
    # ==================================

    return render_template(
        "index.html",

        prediction=predicted_class,

        confidence=confidence,

        image_path=filepath.replace(
            "\\",
            "/"
        ),

        heatmap_path=heatmap_url,

        overlay_path=overlay_url
    )


# ======================================
# Run Application
# ======================================

if __name__ == "__main__":

    app.run(
        debug=True
    )