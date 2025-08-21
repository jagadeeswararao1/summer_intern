# Histopathological Cancer Detection Flask App

A web-based application for detecting cancer in histopathological images using a deep learning CNN model.

## Features

- **AI-Powered Analysis**: Uses a trained CNN model to analyze histopathological images
- **Modern Web Interface**: Beautiful, responsive UI with drag-and-drop functionality
- **Real-time Results**: Instant cancer detection with confidence levels
- **Multiple Format Support**: Supports PNG, JPG, JPEG, TIF, and TIFF formats
- **Secure Processing**: Images are processed securely and not stored permanently

## Prerequisites

- Python 3.8 or higher
- PyTorch (CPU or GPU version)
- Flask
- Other dependencies listed in `requirements.txt`

## Installation

1. **Clone or download the project files**
   ```bash
   # Make sure you have the following files in your project directory:
   # - app.py
   # - model.pt (your trained model)
   # - templates/index.html
   # - requirements.txt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify your model file**
   - Ensure `model.pt` is in the same directory as `app.py`
   - The model should be compatible with the CNN architecture defined in the code

## Usage

1. **Start the Flask application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your web browser
   - Navigate to `http://localhost:5000`
   - You should see the cancer detection interface

3. **Upload and analyze images**
   - Drag and drop an image or click "Choose File"
   - Supported formats: PNG, JPG, JPEG, TIF, TIFF
   - Maximum file size: 16MB
   - Wait for the analysis to complete
   - View the results with confidence levels

## API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Upload image and get prediction
- `GET /health` - Health check endpoint

## Model Architecture

The application uses a CNN model with the following architecture:
- 5 convolutional layers with batch normalization and ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Fully connected layers with sigmoid activation for binary classification

## File Structure

```
Histopathological_cancer_detection/
├── app.py                 # Main Flask application
├── model.pt              # Trained CNN model
├── templates/
│   └── index.html        # Web interface template
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── uploads/             # Temporary upload directory (created automatically)
```

## Configuration

The application can be configured by modifying the following parameters in `app.py`:

- `UPLOAD_FOLDER`: Directory for temporary file storage
- `MAX_CONTENT_LENGTH`: Maximum file size limit
- Model threshold: Currently set to 0.5 for binary classification

## Troubleshooting

### Common Issues

1. **Model loading error**
   - Ensure `model.pt` is in the correct location
   - Check if the model architecture matches the code

2. **CUDA/GPU issues**
   - The app automatically uses CPU if CUDA is not available
   - For GPU usage, ensure PyTorch with CUDA support is installed

3. **Port already in use**
   - Change the port in `app.py`: `app.run(debug=True, host='0.0.0.0', port=5001)`

4. **File upload errors**
   - Check file size (max 16MB)
   - Ensure file format is supported
   - Verify upload directory permissions

### Performance Tips

- For production deployment, consider using a WSGI server like Gunicorn
- Enable GPU acceleration if available for faster inference
- Implement caching for frequently accessed model predictions

## Security Considerations

- Images are processed temporarily and deleted after analysis
- File type validation prevents malicious uploads
- File size limits prevent DoS attacks
- Consider implementing authentication for production use

## Development

To modify the application:

1. **Update model architecture**: Modify the `CNN` class in `app.py`
2. **Change preprocessing**: Update the `transform` variable
3. **Modify UI**: Edit `templates/index.html`
4. **Add new endpoints**: Extend the Flask routes in `app.py`

## License

This project is for educational and research purposes. Please ensure compliance with relevant medical and data protection regulations when using this application.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are correctly installed
3. Ensure your model file is compatible with the expected architecture 