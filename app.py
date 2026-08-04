"""
ObjectSight AI - Flask Application & API Server
Serves the web UI and handles HTTP API endpoints for Object Detection & Internet Intelligence Search.
"""

import os
import io
import time
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from detector import ObjectDetector
from search_engine import search_object_details

app = Flask(__name__, static_folder="static")
CORS(app)

# Initialize detector
detector = ObjectDetector(model_dir=os.path.join(os.path.dirname(__file__), "models"))

# Search history log (in-memory)
search_history = []


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "system": "ObjectSight AI Robotics Vision System",
        "detector_status": "ready" if detector.net is not None else "fallback_heuristic",
        "timestamp": time.time()
    })


@app.route('/api/detect', methods=['POST'])
def detect_objects():
    """
    Endpoint receiving image data (Base64 string or File upload).
    Returns list of detected objects with bounding boxes and cropped image thumbnails.
    """
    try:
        image_np = None
        
        # 1. Base64 Input
        if request.is_json and 'image_b64' in request.json:
            b64_str = request.json['image_b64']
            if ',' in b64_str:
                b64_str = b64_str.split(',')[1]
            img_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. File Upload Input
        elif 'file' in request.files:
            file = request.files['file']
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_np is None or image_np.size == 0:
            return jsonify({"error": "Invalid or missing image payload"}), 400

        conf_thresh = float(request.json.get('confidence', 0.35)) if request.is_json else 0.35
        results = detector.detect(image_np, confidence_threshold=conf_thresh)

        (h, w) = image_np.shape[:2]

        return jsonify({
            "success": True,
            "image_dimensions": {"width": w, "height": h},
            "count": len(results),
            "objects": results
        })

    except Exception as e:
        print(f"Detect endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/search', methods=['GET', 'POST'])
def search_object():
    """
    Endpoint to search internet for object details.
    Accepts query string param 'q' or JSON body {"query": "object_name"}.
    """
    try:
        query = ""
        if request.method == 'GET':
            query = request.args.get('q', '').strip()
        elif request.is_json:
            query = request.json.get('query', '').strip()

        if not query:
            return jsonify({"error": "Missing query parameter 'q'"}), 400

        details = search_object_details(query)

        # Save to search history
        entry = {
            "query": query,
            "title": details.get("title", query.title()),
            "category": details.get("category", "General"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_url": details.get("image_url", ""),
            "summary_snippet": details.get("summary", "")[:120] + "..."
        }
        # Avoid duplicate consecutive entries
        if not search_history or search_history[0]["query"].lower() != query.lower():
            search_history.insert(0, entry)
            if len(search_history) > 30:
                search_history.pop()

        return jsonify({
            "success": True,
            "data": details
        })

    except Exception as e:
        print(f"Search endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/crop_search', methods=['POST'])
def crop_search():
    """
    Accepts image b64 + bounding box [x, y, w, h] + optional label hint.
    Crops region and triggers internet search for details.
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request payload must be JSON"}), 400

        data = request.json
        b64_str = data.get('image_b64', '')
        bbox = data.get('bbox', [0, 0, 100, 100])
        label_hint = data.get('label', 'object').strip()

        if ',' in b64_str:
            b64_str = b64_str.split(',')[1]

        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        crop_b64 = detector.crop_region(img_np, bbox[0], bbox[1], bbox[2], bbox[3])

        # Fetch details for label hint
        details = search_object_details(label_hint)
        details["cropped_region_b64"] = crop_b64

        return jsonify({
            "success": True,
            "label": label_hint,
            "data": details
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/history', methods=['GET', 'DELETE'])
def history():
    global search_history
    if request.method == 'DELETE':
        search_history = []
        return jsonify({"success": True, "message": "History cleared"})
    return jsonify({
        "success": True,
        "history": search_history
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 ObjectSight AI Server starting on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
