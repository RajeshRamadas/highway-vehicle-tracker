# Highway Vehicle Tracker

## Overview

Highway Vehicle Tracker is an advanced real-time object detection and vehicle tracking application using YOLOv8 (You Only Look Once) neural network models. The application provides a comprehensive web interface for processing video streams from highways, with support for multiple YOLO model variants and custom model uploads.

## Key Features

### Vehicle Detection and Tracking
- Real-time vehicle detection using YOLOv8 models
- Multiple pre-trained model variants:
  - YOLOv8 Nano (fastest)
  - YOLOv8 Small
  - YOLOv8 Medium
  - YOLOv8 Large
  - YOLOv8 XLarge (most accurate)

### Intelligent Tracking Capabilities
- Directional vehicle counting
- Detailed statistics by vehicle type
- Customizable tracking line position
- Support for tracking multiple vehicle classes
  - Cars
  - Trucks
  - Buses
  - Motorcycles
  - Other transportation vehicles

### Flexible Input Sources
- Camera input
- Highway video file input
- Adjustable resolution and frame rate

### Performance Controls
- Confidence threshold adjustment
- Frame quality control
- Target frame rate selection

### Custom Model Support
- Upload and use custom PyTorch (.pt) models
- Seamless integration with standard YOLO models

## Potential Use Cases

- Traffic flow analysis
- Highway infrastructure planning
- Vehicle counting and classification
- Transportation research
- Smart city monitoring

## Prerequisites

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- aiohttp
- OpenCV
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/highway-vehicle-tracker.git
   cd highway-vehicle-tracker
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Ultralytics YOLO models:
   ```bash
   pip install -U ultralytics
   ```

## Running the Application

```bash
python server.py
```

The application will start and be accessible at `http://localhost:8081`

## Usage Guide

### Connecting a Video Source
1. Choose between camera or highway video file input
2. Select a YOLO model (standard or custom)
3. Configure tracking and performance settings
4. Click "Start" to begin processing

### Vehicle Tracking
- Enable vehicle tracking checkbox
- Adjust the counting line position
- View real-time counts by vehicle type and direction

### Performance Tuning
- Adjust confidence threshold
- Modify frame quality
- Select target frame rate
- Choose resolution

## Custom Model Upload

1. Prepare a PyTorch (.pt) object detection model
2. Go to the "Custom Model" tab
3. Upload the model file
4. Optionally provide a custom name
5. Use the uploaded model for processing

## Technical Details

- WebSocket-based real-time communication
- Asynchronous frame processing
- GPU acceleration support
- Flexible model loading

## Security and Performance Notes

- Supports CUDA for GPU acceleration
- Handles various model sizes efficiently
- Configurable processing parameters

## Troubleshooting

- Ensure all dependencies are installed
- Check console logs for detailed error messages
- Verify camera/video file permissions
- Confirm model file compatibility

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the GitHub repository.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Ultralytics for YOLO implementation
- OpenCV for image processing
- PyTorch for deep learning framework
