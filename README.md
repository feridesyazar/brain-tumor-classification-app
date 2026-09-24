# 🧠 Brain Tumor Classification App

An AI-powered web application for classifying **brain tumor MRI images** using a trained deep learning model.

The application allows users to upload an MRI image and receive a predicted tumor class together with a confidence score. The project demonstrates an end-to-end computer vision workflow, from image preprocessing and deep learning inference to web deployment.

## Live Demo

### Hugging Face Spaces

[Open the Brain Tumor MRI Classifier on Hugging Face](https://huggingface.co/spaces/ferides/brain-tumor-classification-app)

### Render

[Open the Brain Tumor MRI Classifier on Render](https://brain-tumor-mri-classifier-6t98.onrender.com)

> The Render instance may require a short startup time after a period of inactivity.

## Supported Classes

The model classifies MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

## Features

- Upload brain MRI images
- Deep learning-based image classification
- Prediction of four MRI classes
- Confidence score display
- Interactive web interface
- Clinical-style user interface
- Deployment on Hugging Face Spaces
- Deployment on Render

## Model

The application uses a trained **TensorFlow / Keras deep learning model** for MRI image classification.

The prediction workflow follows this general process:

```text
MRI Image
    ↓
Image Preprocessing
    ↓
Deep Learning Model
    ↓
Class Prediction
    ↓
Confidence Score
    ↓
Prediction Result
```

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Gradio
- GitHub
- Hugging Face Spaces
- Render

## Application Workflow

1. The user uploads a brain MRI image.
2. The image is prepared for model inference.
3. The trained neural network processes the image.
4. The model calculates prediction probabilities.
5. The most likely MRI class is selected.
6. The result and confidence score are displayed in the web interface.

## Key Project Files

```text
brain-tumor-classification-app/
│
├── app.py
├── class_names.json
├── requirements.txt
├── README.md
├── .gitignore
└── deployment-related files
```

The repository contains the application code and configuration required for deployment. Trained model files are used by the deployed application.

## Example Prediction

```text
Prediction: Glioma
Confidence: 98.xx%
```

Prediction results depend on the uploaded MRI image and the model output.

## Deployment

The application is available through two deployment environments:

**Hugging Face Spaces**  
Used to provide an interactive Gradio-based AI application.

**Render**  
Used as an additional live web deployment of the application.

This setup demonstrates how a machine learning model can be integrated into a web interface and deployed as an accessible online application.

## Project Purpose

This project demonstrates practical experience with:

- Computer Vision
- Deep Learning
- Medical Image Classification
- TensorFlow / Keras
- Model Inference
- Gradio Application Development
- Cloud Deployment
- GitHub-based Project Management

The project was developed for educational, demonstration and portfolio purposes.

## Disclaimer

This application is intended for **educational and demonstration purposes only**.

It is not a medical diagnostic tool and should not be used as a substitute for professional medical evaluation.
