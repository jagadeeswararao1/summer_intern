### Histopathological Cancer Detection – Flask App

A web application for detecting cancer in histopathological images using a CNN model built with PyTorch. It provides a simple drag-and-drop UI, real-time predictions, and an HTTP API.

## Features
- AI-powered cancer detection with a pretrained CNN
- Modern UI with drag-and-drop upload
- Real-time results with probability and confidence
- Supports PNG, JPG, JPEG, TIF, TIFF
- Uploaded images are stored permanently in the `uploads` folder

## Tech Stack
- Flask (Python)
- PyTorch + Torchvision
- Bootstrap + Vanilla JS frontend

## Project Structure
```
Histopathological_cancer_detection/
├── app.py                 # Flask application (API + model inference)
├── best_model.pt          # Trained PyTorch model weights (required)
├── batch_predict.py       # Batch prediction script (for folders of images)
├── test_model.py          # Basic model tests/utilities
├── templates/
│   └── index.html         # Web UI
├── static/
│   └── c1.webp            # Background/logo image
├── uploads/               # Saved uploads (auto-created)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Prerequisites
- Python 3.8+
- pip
- Virtual environment tool (recommended)
- PyTorch per your environment (CPU or GPU)

## Installation
1) Clone or copy the project files into a folder.

2) (Recommended) Create and activate a virtual environment:
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3) Install dependencies:
```bash
pip install -r requirements.txt
```

4) Ensure the model file exists:
- Place `best_model.pt` in the project root (same folder as `app.py`).

## Running the App (Development)
```bash
python app.py
```
- Local: http://127.0.0.1:5000
- Network: printed as http://<your-ip>:5000

Health check:
```bash
curl http://127.0.0.1:5000/health
```
Expected:
```json
{"status":"healthy","model_loaded":true}
```

## Usage (Web UI)
1) Open the app in your browser.
2) Drag-and-drop or select an image (PNG/JPG/JPEG/TIF/TIFF, up to 16MB).
3) View results:
   - “Cancer Detected” or “No Cancer Detected”
   - Probability score and Confidence level
4) Uploaded files are saved in the `uploads` folder permanently.

## HTTP API
- GET `/` – Web UI
- GET `/health` – Health check
  - Response: `{"status":"healthy","model_loaded": true}`
- POST `/predict` – Image prediction
  - Form-data: `file` = image file (PNG/JPG/JPEG/TIF/TIFF)
  - Response:
```json
{
  "prediction": 1,
  "probability": 0.8731,
  "result": "Cancer Detected",
  "confidence": "High"
}
```

Example request:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@/path/to/image.jpg"
```

## Configuration
Edit these in `app.py` if needed:
- `UPLOAD_FOLDER` (default: `uploads`)
- `MAX_CONTENT_LENGTH` (default: 16MB)
- Model file path: `best_model.pt`
- Host/port: `host='0.0.0.0', port=5000`
- Threshold: fixed at 0.5 for binary classification
- Note: Uploaded images are kept permanently (no auto-delete)

## Batch Prediction (Optional)
Use `batch_predict.py` for folder-based inference.
- Run:
```bash
python batch_predict.py --help
```
- Typical usage:
```bash
python batch_predict.py --input_dir /path/to/images --output_csv predictions.csv
```
Adjust to match the script’s help output.

## Testing (Optional)
Basic checks are available in `test_model.py`. Run:
```bash
python test_model.py
```

## Troubleshooting
- Model not found:
  - Error: `FileNotFoundError: best_model.pt`
  - Fix: Ensure `best_model.pt` is in the same folder as `app.py`.
- Model architecture mismatch:
  - Ensure the weights match the CNN defined in `app.py`.
- Static images (background/logo) not visible:
  - Verify `static/c1.webp` exists.
  - Access via `http://127.0.0.1:5000/`, not a file:// URL.
- Large files rejected:
  - Increase `app.config['MAX_CONTENT_LENGTH']` in `app.py`.
- GPU/CPU:
  - The app auto-detects CUDA. If PyTorch CUDA isn’t installed, it will run on CPU.

## Production (Optional)
Use a production WSGI server and reverse proxy.
- Windows (example with Waitress):
```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```
- Linux (example with Gunicorn):
```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```
Place behind Nginx/Apache as needed.

## Privacy Notice
- Uploaded images are stored permanently in `uploads`. Ensure you comply with your data retention policy or modify the code to delete images after processing.

## License
Add your preferred license (e.g., MIT) to a `LICENSE` file.

## Acknowledgements
- PyTorch, Torchvision, Flask, Bootstrap

If you want, I can save this into `README.md` for you.
