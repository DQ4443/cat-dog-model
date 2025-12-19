# Cat or Dog Classifier

A web app that uses your webcam to classify images as cat or dog using a TFLite model.

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
cd backend
source venv/bin/activate
python app.py
```

Open http://127.0.0.1:5000 in browser.

## Usage

1. Allow webcam access when prompted
2. Click "Capture & Classify"
3. View the prediction and confidence score

## API

- `GET /` - Web interface
- `POST /classify` - Classify an image (JSON body: `{"image": "<base64>"}`)
- `GET /history` - View classification history
