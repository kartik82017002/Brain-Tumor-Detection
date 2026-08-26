# Brain Tumor Detection using ResNet50 and Flask

This project detects brain tumors from MRI images using a ResNet50-based CNN model. The trained model is deployed through a simple Flask web application where users can upload an MRI scan and get a prediction.

## Overview
I used transfer learning with ResNet50 to classify MRI brain scan images and check for the presence of a tumor. The model was trained and tested on Google Colab, and later integrated into a Flask app for real-time predictions.

## Features
- Image classification using ResNet50 (Transfer Learning)
- Model trained and tested on Google Colab
- Flask-based web interface to upload MRI images and view predictions

## Tech Stack
- Python
- TensorFlow / Keras
- ResNet50 (Pretrained Model)
- Flask
- HTML/CSS

## Project Structure
Untitled4.ipynb - Main notebook containing data preprocessing, training and evaluation
flask_app/ - Flask application files
README.md

## How to Run
1. Clone the repository

git clone https://github.com/kartik82017002/Brain-Tumor-Detection.git

2. Install dependencies

pip install -r requirements.txt

3. Run the Flask app

python app.py

## Results
Add your model's accuracy or other metrics here, for example "Achieved 95% accuracy on test set".

## Dataset
Mention where the dataset was taken from, for example a Kaggle brain MRI dataset link.

## Acknowledgements
Built as a learning project to understand how deep learning can be applied to medical imaging problems.
