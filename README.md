# Corn Disease Detection

A Streamlit web app that classifies corn (maize) leaf images as **Blight**,
**Common Rust**, **Gray Leaf Spot**, or **Healthy** using a deep learning model.

## Features

- Upload a leaf image and get an instant prediction
- Confidence score for each class
- Built-in sample images to try

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open http://localhost:8501.

## Files

- `streamlit_app.py` — the app
- `model.tflite` — the trained model used for predictions
- `requirements.txt` — dependencies
- `static/` — sample images
