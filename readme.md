# Highway Vehicle Tracker

A web-based application for detecting and tracking vehicles using YOLOv8 object detection models. This application combines real-time object detection with advanced vehicle tracking capabilities, S3 file browsing, and model management functionality.

## Features

- **Real-time Object Detection**: Process video from a webcam or file using YOLOv8 models
- **Vehicle Tracking**: Count vehicles crossing a customizable line with directional tracking
- **Multiple YOLOv8 Models**: Support for nano, small, medium, large, and extra-large models
- **S3 File Browser**: Browse and download files from your AWS S3
- **Model Management**: Download and manage YOLOv8 models
- **Custom Model Support**: Upload and use your own custom YOLO models
- **Performance Controls**: Adjust quality, resolution, and frame rate for optimal performance
- **Statistics Display**: View detection counts, processing times, and vehicle statistics

## Screenshots

![Vehicle Tracker Main Screen](screenshots/main_screen.png)
![Model Download Page](screenshots/model_download.png)
![S3 Browser](screenshots/s3_browser.png)

## Requirements

- Python 3.8+ 
- PyTorch 2.0+
- OpenCV 4.5+
- Ultralytics YOLOv8
- AWS account with S3 bucket (for S3 functionality)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/highway-vehicle-tracker.git
   cd highway-vehicle-tracker
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure AWS credentials:
   Create a `.env` file in the project root with your AWS credentials:
   ```
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   S3_BUCKET_NAME=your_bucket_name
   ```

## Usage

1. Start the server:
   ```
   python server.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8081/
   ```

3. To use the application:
   - Select a video file or use your webcam
   - Choose a YOLO model for detection
   - Adjust performance settings as needed
   - Click "Start" to begin detection and tracking
   - Enable vehicle counting to track objects crossing a line

4. To download models:
   - Navigate to "Download Models" page
   - Click the download button for the desired model
   - Models will be stored locally for future use

5. To browse S3 files:
   - Navigate to "S3 Browser" page
   - Browse through folders and files in your S3 bucket
   - Download files as needed

## Project Structure

```
/highway-vehicle-tracker/
│
├── server.py               # Main server application
├── index.html              # Main web interface
├── .env                    # AWS configuration (create this)
│
├── models/                 # Downloaded YOLO models
│   ├── yolov8n.pt          # Nano model
│   ├── yolov8s.pt          # Small model
│   └── ...
│
├── custom_models/          # User-uploaded custom models
│
├── templates/              # HTML templates
│   ├── s3_browser.html     # S3 file browser interface
│   └── download_models.html # Model download interface
│
└── static/                 # Static assets
    └── client.js           # Client-side JavaScript
```

## API Endpoints

- `/ws` - WebSocket for real-time processing
- `/models` - Get available YOLO models
- `/test_models` - Test YOLO models
- `/upload_model` - Upload custom models
- `/s3-browser` - S3 file browser page
- `/api/s3/files` - List files in S3 bucket
- `/api/s3/download` - Generate presigned URL for S3 file download
- `/download-model` - Model download page
- `/api/yolo-models` - List available YOLO models for download
- `/api/download-model` - Download a specific YOLO model

## Performance Optimization

For optimal performance:

1. **Model Selection**:
   - Smaller models (nano, small) work best for real-time processing
   - Larger models provide higher accuracy but require more powerful hardware

2. **Resolution and Quality**:
   - Lower resolution and quality settings improve processing speed
   - On less powerful hardware, use "Low" resolution setting

3. **Frame Rate**:
   - Adjust the target frame rate based on your hardware capabilities
   - 5-10 FPS is typically sufficient for vehicle tracking

## Customization

### Custom YOLO Models

You can upload your own YOLOv8-compatible models:

1. Navigate to the "Custom Model" tab in the model selector
2. Choose a .pt file containing your custom model
3. Click "Upload & Use Model"

### Vehicle Tracking Settings

Adjust vehicle tracking:

1. Enable "Vehicle Counting" in the settings
2. Use the slider to position the counting line
3. Reset counts as needed using the "Reset Counts" button

## Troubleshooting

- **Model Loading Errors**: Ensure model files are in the correct format and location
- **WebSocket Connection Issues**: Check for firewall blocking WebSocket connections
- **AWS Credential Errors**: Verify your AWS credentials in the .env file
- **Slow Performance**: Try a smaller model or reduce resolution/quality settings
- **CUDA Errors**: Check CUDA installation if using GPU acceleration

## License

[MIT License](LICENSE)

## Acknowledgments

- YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics)
- Frontend based on modern web technologies

## Contact

For questions and support, please [open an issue](https://github.com/yourusername/highway-vehicle-tracker/issues) on GitHub.
