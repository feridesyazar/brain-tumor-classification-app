import json
from pathlib import Path

import gradio as gr
import numpy as np
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "brain_tumor_mri_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"


model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)


@spaces.GPU(duration=30)
def predict_mri(image):
    if image is None:
        return {}

    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = int(input_shape[1] or 224)
    width = int(input_shape[2] or 224)

    image = image.convert("RGB")
    image = image.resize((width, height))

    image_array = keras.utils.img_to_array(image)
    image_array = image_array.astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    results = {
        class_names[index]: float(probability)
        for index, probability in enumerate(predictions)
    }

    return results


def clear_app():
    return None, {}


CUSTOM_CSS = """
:root {
    --medical-blue: #1d4ed8;
    --medical-blue-dark: #163f8c;
    --medical-blue-soft: #eaf3ff;
    --medical-border: #d9e6f5;
    --medical-text: #172033;
    --medical-muted: #5f6f85;
}

/* Page background */
html,
body {
    min-height: 100%;
    margin: 0;
    padding: 0;
    background:
        radial-gradient(circle at top left, rgba(184, 214, 245, 0.65) 0%, transparent 26%),
        radial-gradient(circle at top right, rgba(198, 222, 248, 0.50) 0%, transparent 24%),
        linear-gradient(135deg, #dceaf7 0%, #edf5fc 45%, #d7e7f5 100%) !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    min-height: 100vh;
    background:
        radial-gradient(circle at top left, rgba(238, 246, 255, 0.95) 0%, transparent 34%),
        linear-gradient(180deg, #eef6fd 0%, #f8fbff 100%) !important;
}

#medical-shell {
    padding: 28px 20px 34px 20px;
}

#medical-header {
    background: linear-gradient(135deg, #0f4c81, #2563eb);
    border-radius: 24px;
    padding: 28px 30px;
    box-shadow: 0 18px 45px rgba(37, 99, 235, 0.18);
    margin-bottom: 22px;
}

#medical-header h1 {
    color: white;
    margin: 0 0 8px 0;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

#medical-header p {
    color: rgba(255,255,255,0.90);
    margin: 0;
    font-size: 1rem;
    line-height: 1.65;
}

#medical-header strong {
    color: white !important;
    font-weight: 700;
}

.medical-badge {
    display: inline-block;
    margin-bottom: 14px;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.30);
    color: white;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}

#upload-card,
#result-card {
    background: white;
    border: 1px solid var(--medical-border);
    border-radius: 22px;
    padding: 20px;
    box-shadow: 0 12px 34px rgba(31, 65, 114, 0.08);
}

#upload-card h3,
#result-card h3 {
    color: var(--medical-text);
    margin-top: 0;
    margin-bottom: 6px;
}

.section-note {
    color: var(--medical-muted);
    font-size: 0.92rem;
    margin-bottom: 14px;
}

#brain-image {
    border-radius: 18px !important;
    overflow: hidden;
    border: 1px dashed #9fc3eb !important;
    background: #f8fbff !important;
}

#analyze-btn {
    background: linear-gradient(90deg, #1d4ed8, #2563eb) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 13px !important;
    min-height: 46px !important;
}

#analyze-btn:hover {
    background: linear-gradient(90deg, #163f8c, #1d4ed8) !important;
}

#clear-btn {
    border-radius: 13px !important;
    min-height: 46px !important;
}

#prediction-output {
    border-radius: 16px !important;
}

#safety-note {
    margin-top: 20px;
    background: #f8fafc;
    border-left: 4px solid #60a5fa;
    border-radius: 12px;
    padding: 14px 16px;
    color: #475569;
    font-size: 0.88rem;
    line-height: 1.6;
}

#footer-note {
    text-align: center;
    color: #718096;
    font-size: 0.78rem;
    margin-top: 18px;
}
"""


with gr.Blocks(
    css=CUSTOM_CSS,
    title="Brain MRI Tumor Classification"
) as demo:

    with gr.Column(elem_id="medical-shell"):

        gr.HTML(
            """
            <div id="medical-header">
                <div class="medical-badge">MEDICAL AI • MRI ANALYSIS</div>
                <h1>Brain MRI Tumor Classification</h1>
                <p>
                    Upload a brain MRI image to classify it as
                    <strong>glioma</strong>, <strong>meningioma</strong>,
                    <strong>pituitary tumor</strong> or <strong>no tumor</strong>.
                </p>
            </div>
            """
        )

        with gr.Row(equal_height=True):

            with gr.Column(scale=1, elem_id="upload-card"):
                gr.HTML(
                    """
                    <h3>Upload MRI Image</h3>
                    <div class="section-note">
                        Select a clear brain MRI image for model analysis.
                    </div>
                    """
                )

                image_input = gr.Image(
                    type="pil",
                    label="Brain MRI Image",
                    elem_id="brain-image"
                )

                with gr.Row():
                    clear_button = gr.Button(
                        "Clear",
                        elem_id="clear-btn"
                    )

                    analyze_button = gr.Button(
                        "Analyze MRI",
                        variant="primary",
                        elem_id="analyze-btn"
                    )

            with gr.Column(scale=1, elem_id="result-card"):
                gr.HTML(
                    """
                    <h3>Prediction Results</h3>
                    <div class="section-note">
                        Confidence scores for all four classes.
                    </div>
                    """
                )

                prediction_output = gr.Label(
                    num_top_classes=4,
                    label="Model Prediction",
                    elem_id="prediction-output"
                )

                gr.HTML(
                    """
                    <div id="safety-note">
                        <strong>Clinical safety notice:</strong>
                        Educational use only — not for medical diagnosis.
                    </div>
                    """
                )

        gr.HTML(
            """
            <div id="footer-note">
                Brain MRI Classification • TensorFlow / Keras • Gradio
            </div>
            """
        )

    analyze_button.click(
        fn=predict_mri,
        inputs=image_input,
        outputs=prediction_output
    )

    clear_button.click(
        fn=clear_app,
        inputs=None,
        outputs=[image_input, prediction_output]
    )


if __name__ == "__main__":
    import os

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
