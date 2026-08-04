# 🤖 ObjectSight AI - Visual Object Detection & Internet Knowledge Search Engine

**ObjectSight AI** is a real-time object detection software designed for robotics and computer vision workflows. It identifies objects from live webcam feeds, uploaded photos/videos, or custom cropped image regions and automatically queries the internet (Wikipedia, DuckDuckGo, technical databases) for in-depth object details, specifications, history, and key facts.

---

## 🌟 Key Features

1. **Dual AI Detection Pipeline**:
   - **Client-Side Browser AI**: Powered by TensorFlow.js (COCO-SSD) for ultra-fast, zero-latency live camera bounding box rendering.
   - **Server-Side Vision Engine**: Powered by OpenCV DNN (MobileNet-SSD) and contour ROI extractors for server backend image processing.

2. **Automated Internet Search Engine**:
   - Fetches live Wikipedia page summaries, main image thumbnails, technical specifications, and key facts for any detected object.
   - Leverages DuckDuckGo Instant Answer API for extra definitions, taxonomy, and related topics.

3. **Interactive Robotics HUD Interface**:
   - Futuristic Dark/Cyberpunk Glassmorphism UI styled with custom CSS (`Outfit` & `JetBrains Mono` typography).
   - Real-time animated bounding box overlays with confidence score tags.
   - Click on **any** detected bounding box or object chip to instantly fetch internet details.

4. **Manual Region Crop Tool**:
   - Click and drag a custom bounding box over any unrecognized region on the camera feed or image to perform a targeted crop search.

5. **Voice Readout (Text-To-Speech)**:
   - Listen to object descriptions and summaries spoken aloud with built-in Web Speech API synthesis.

6. **Preset Media & Search History**:
   - Includes preset sample images (Smartphone, Laptop, Coffee Cup, Dog, Robot) for instant testing.
   - Persistent search history log.

---

## 🚀 How to Run

### Method 1: Easy Launch Script
```bash
cd /home/jerome/Robotics/Project/ObjectSight_AI
./run.sh
```

### Method 2: Python Command
```bash
python3 app.py
```

Once started, open your web browser and navigate to:
👉 **`http://localhost:5000`**

---

## 📂 Project Architecture

```
ObjectSight_AI/
├── app.py              # Flask server & REST API endpoints (/api/detect, /api/search, /api/history)
├── search_engine.py    # Internet search module (Wikipedia API, DuckDuckGo, specs generator)
├── detector.py          # OpenCV DNN & contour ROI object detection module
├── requirements.txt    # Python dependencies
├── run.sh              # Bash startup script
├── README.md           # Documentation
└── static/
    ├── index.html      # Robotics HUD web application interface
    ├── css/
    │   └── style.css   # Glassmorphic dark cyberpunk design system & HUD animations
    └── js/
        └── app.js      # Client application logic & TensorFlow.js COCO-SSD pipeline
```

---

## 🛠 Tech Stack
- **Frontend**: HTML5, CSS3 (Vanilla Glassmorphism), JavaScript (ES6+), TensorFlow.js COCO-SSD, Web Speech API
- **Backend**: Python 3, Flask, Flask-CORS, OpenCV (`cv2`), NumPy
- **Data Sources**: Wikipedia REST & Python API, DuckDuckGo Instant Answer API
