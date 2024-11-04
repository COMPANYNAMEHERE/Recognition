# Video Recognition Program

A PyQt6-based desktop application for real-time video analysis using YOLOv8 object detection and facial expression recognition.

## Features

- **Object Detection**: Using YOLOv8 for accurate real-time object detection.
- **Facial Expression Recognition**: Integrated DeepFace for emotion analysis.
- **GPU Acceleration**: CUDA support for improved performance.
- **User-Friendly Interface**: Clean, modern PyQt6 GUI.
- **Progress Tracking**: Real-time progress indicators and status updates.
- **Configurable Settings**: Adjustable parameters for detection confidence, model selection, and more.
- **Results Export**: Save detection results for further analysis.

## Requirements

- Python 3.10 or higher.
- NVIDIA GPU with CUDA support (optional but recommended).
- Windows/Linux/macOS.

## To-Do List

- Dialogue detection.
- Optimization.
- Better UI.
- Generic scene context recognition.
- Working cancel button.
- Persistence.
- Progress bar ETA.
- Fix save results.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/video-recognition-program.git
   cd video-recognition-program
   ```

2. Create and activate a virtual environment:

   ```bash
   # Using conda
   conda create -n recognizer python=3.10
   conda activate recognizer

   # Or using venv
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Unix or macOS
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install CUDA support (if using GPU):

   ```bash
   # For PyTorch with CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For TensorFlow
   pip install tensorflow[and-cuda]
   ```

## Project Structure

```
project_root/
├── src/
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── settings_dialog.py
│   │   └── video_widget.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── model_manager.py
│   │   ├── video_converter.py
│   │   └── video_processor.py
│   └── main.py
├── models/          # Created at runtime
├── logs/            # Application logs
├── cache/           # Temporary files
├── output/          # Processed results
├── requirements.txt
└── config.json
```

## Usage

1. Start the application:

   ```bash
   cd src
   python main.py
   ```

2. Load a video file using the "Load Video" button.

3. Click "Analyze" to start processing.

4. Use the settings dialog to configure:

   - Detection confidence threshold.
   - GPU usage.
   - Model selection.
   - Save location.
   - Other processing parameters.

## Features in Detail

### Object Detection

- Uses YOLOv8 for real-time object detection.
- Supports multiple object classes.
- Configurable confidence thresholds.
- GPU acceleration when available.

### Facial Expression Recognition

- Detects faces using MediaPipe.
- Analyzes expressions using DeepFace.
- Recognizes emotions:
  - Happy
  - Sad
  - Angry
  - Surprised
  - Neutral
  - Fear
  - Disgust

### Video Processing

- Supports multiple video formats.
- Automatic video format conversion.
- Progress tracking.
- Memory-efficient processing.

## Configuration

Edit `config.json` to modify default settings:

```json
{
    "confidence_threshold": 0.5,
    "use_gpu": true,
    "model": "yolov8n",
    "save_path": "detections",
    "max_image_size": 1280,
    "batch_size": 4,
    "enable_optimization": true,
    "cache_results": true
}
```

## Logging

- Application logs are stored in `logs/app.log`.
- Error logs are stored in `logs/errors.log`.
- View logs for debugging and monitoring.

## Development

To contribute to development:

1. Install development dependencies:

   ```bash
   pip install pytest black pylint
   ```

2. Run tests:

   ```bash
   pytest
   ```

3. Format code:

   ```bash
   black src/
   ```

4. Check code quality:

   ```bash
   pylint src/
   ```

## Troubleshooting

Common issues and solutions:
**Can't install requirements**
   - Run Python 3.10

**GPU not detected**:

   - Ensure NVIDIA drivers are installed.
   - Verify CUDA installation.
   - Check PyTorch/TensorFlow GPU support.

**Memory issues**:

   - Reduce batch size in settings.
   - Lower maximum image size.
   - Close other GPU applications.

**Video loading fails**:

   - Check supported video formats.
   - Ensure video file isn't corrupted.
   - Check available disk space.

## Contributors

Joost van Tiggelen

## Acknowledgments

- YOLOv8 by Ultralytics.
- DeepFace for facial expression recognition.
- PyQt6 for the GUI framework.
- OpenCV for video processing.
- Claude & ChatGPT

---

Created by [Your Name/Organization]

