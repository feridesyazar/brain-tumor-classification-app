# 🧠 Brain Tumor Classification App

An AI-powered web application for classifying **brain tumor MRI images** using a trained deep learning model.

The application analyzes uploaded MRI images and predicts one of four supported classes.

## 🚀 Live Demos

### 🤗 Hugging Face Spaces

👉 [Open the Brain Tumor MRI Classifier on Hugging Face](https://huggingface.co/spaces/ferides/brain-tumor-classification-app)

### 🌐 Render

👉 [Open the Brain Tumor MRI Classifier on Render](https://brain-tumor-mri-classifier-6t98.onrender.com)

> The free Render instance may take a short time to start after a period of inactivity.

## 🩺 Supported Classes

The model can classify MRI images into the following categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

## ✨ Features

- Upload a brain MRI image
- Analyze the image using a trained deep learning model
- Display the predicted MRI class
- Show prediction confidence
- Interactive web interface
- Clinical-style user interface
- Deployment on Hugging Face Spaces and Render

## 🧠 Model

The application uses a trained **TensorFlow / Keras deep learning model** for MRI image classification.

The prediction workflow is:

```text
MRI Image
    ↓
Image Preprocessing
    ↓
Deep Learning Model
    ↓
Classification
    ↓
Prediction Result
```

## 🛠 Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Gradio
- GitHub
- Hugging Face Spaces
- Render

## 📁 Project Structure

```text
brain-tumor-classification-app/
│
├── app.py
├── class_names.json
├── requirements.txt
├── .gitignore
├── .python-version
├── LICENSE
└── README.md
```

Additional deployment-related files may also be included in the repository.

The trained `.keras` model files are hosted with the deployed application.

## 🔍 Example Prediction

```text
Prediction: Glioma
Confidence: 98.xx%
```

The exact prediction depends on the uploaded MRI image.

## 🎯 Project Purpose

This project demonstrates an end-to-end computer vision workflow combining:

**Artificial Intelligence • Deep Learning • Medical Image Classification • Web Deployment**

It was developed as an educational, demonstration and portfolio project.

## ⚠️ Disclaimer

This application is intended for **educational and demonstration purposes only**.

It is not a medical diagnostic tool and should not be used as a substitute for professional medical evaluation.
