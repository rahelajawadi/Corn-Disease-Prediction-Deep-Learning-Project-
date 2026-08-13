# 🌽 Corn Disease Detection

<p align="center">
  <strong>Deep Learning-Based Corn Leaf Disease Classification</strong>
</p>

<p align="center">
  An interactive Streamlit application that uses a trained TensorFlow Lite model to identify common corn leaf diseases from images.
</p>

---

## 📌 Overview

This project implements a **deep learning image classification system** for detecting corn leaf diseases. Users can upload a corn leaf image and receive an instant prediction along with a confidence score.

## ✨ Highlights

🌱 **Disease Classification**  
Detects **Blight, Common Rust, Gray Leaf Spot,** and **Healthy** leaves.

🧠 **Deep Learning Model**  
Uses a trained **TensorFlow Lite** model for image classification.

📊 **Confidence Scores**  
Displays the predicted class and confidence level for each prediction.

🖥️ **Interactive App**  
Streamlit-powered interface with image uploads and built-in sample images.

## 🛠️ Tech Stack

**Python** · **TensorFlow Lite / LiteRT** · **Streamlit** · **NumPy** · **Pillow**

## 📂 Project Structure

```text
├── streamlit_app.py
├── model.tflite
├── requirements.txt
├── run_app.sh
└── static/
