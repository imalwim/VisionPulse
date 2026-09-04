# VisionPulse - Real-Time Vision Analytics Pipeline

VisionPulse is an end-to-end computer vision application engineered for low-latency object detection, tracking, and crowd monitoring using state-of-the-art YOLOv8 architecture and OpenCV.

## Key Features
- **Real-Time Detection & Tracking:** Leverages YOLOv8 with built-in multi-object tracking.
- **Dynamic Density Analysis:** Actively monitors and calculates the real-time presence of people in the feed.
- **Performance Metrics:** On-screen real-time FPS (Frames Per Second) benchmarking for edge optimization.
- **Edge Deployment Ready:** Optimized using lightweight network backbones suitable for constrained environments.

## Tech Stack
- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch, Ultralytics YOLOv8
- **Image Processing:** OpenCV, NumPy

## Getting Started

### 1. Clone the repository
\`\`\`bash
git clone https://github.com/imalwim/VisionPulse.git
cd VisionPulse
\`\`\`

### 2. Install dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Run the analytics stream
\`\`\`bash
python app.py
\`\`\`
*(Press `q` at any time to close the video feed window)*
