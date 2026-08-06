"""
ObjectSight AI - Object Detector Module
Handles image preprocessing, OpenCV DNN / heuristic object detection, ROI cropping, and bounding box computation.
"""

import cv2
import numpy as np
import base64
import os
import urllib.request

# COCO 80 Class Labels for Object Detection
COCO_CLASSES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class ObjectDetector:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_dir = model_dir
        self.net = None
        self.load_model()

    def load_model(self):
        """Attempts to load MobileNetSSD Caffe model if available, or initialize DNN."""
        prototxt_path = os.path.join(self.model_dir, "MobileNetSSD_deploy.prototxt")
        caffemodel_path = os.path.join(self.model_dir, "MobileNetSSD_deploy.caffemodel")
        
        # Download lightweight Caffe MobileNet-SSD weights if missing
        if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
            try:
                os.makedirs(self.model_dir, exist_ok=True)
                print("Downloading MobileNet-SSD prototxt...")
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt",
                    prototxt_path
                )
                print("Downloading MobileNet-SSD caffe model (~22MB)...")
                urllib.request.urlretrieve(
                    "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel",
                    caffemodel_path
                )
            except Exception as e:
                print(f"Model download notice: {e}. Fallback detection modes active.")

        if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
            try:
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
                print("MobileNetSSD OpenCV DNN successfully loaded!")
            except Exception as err:
                print(f"Error loading Caffe net: {err}")
                self.net = None

    def detect(self, image_np, confidence_threshold=0.35):
        """
        Runs object detection on numpy image array (BGR format).
        Returns list of dicts with: label, confidence, bbox [x, y, width, height], and cropped_b64.
        """
        (h, w) = image_np.shape[:2]
        detections_output = []

        if self.net is not None:
            try:
                blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 0.007843, (300, 300), 127.5)
                self.net.setInput(blob)
                detections = self.net.forward()

                # VOC 21 class subset for MobileNetSSD
                voc_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                               "sofa", "train", "tvmonitor"]

                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > confidence_threshold:
                        idx = int(detections[0, 0, i, 1])
                        if idx < len(voc_classes):
                            label = voc_classes[idx]
                            if label == "background":
                                continue
                            
                            # Map VOC label to friendly label
                            if label == "tvmonitor": label = "tv / monitor"
                            if label == "aeroplane": label = "airplane"
                            if label == "motorbike": label = "motorcycle"

                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")

                            startX = max(0, startX)
                            startY = max(0, startY)
                            endX = min(w, endX)
                            endY = min(h, endY)

                            bw = max(10, endX - startX)
                            bh = max(10, endY - startY)

                            # Crop region ROI
                            crop = image_np[startY:endY, startX:endX]
                            crop_b64 = ""
                            if crop.size > 0:
                                _, buffer = cv2.imencode('.jpg', crop)
                                crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

                            detections_output.append({
                                "id": f"det_{i}_{idx}",
                                "label": label,
                                "confidence": round(float(confidence) * 100, 1),
                                "bbox": [int(startX), int(startY), int(bw), int(bh)],
                                "crop_b64": crop_b64
                            })
            except Exception as e:
                print(f"Detection inference exception: {e}")

        # Fallback Contour Object Segmenter if DNN yields 0 objects
        if not detections_output:
            detections_output = self._heuristic_segmentation(image_np)

        return detections_output

    def _heuristic_segmentation(self, img_np):
        """Segment prominent object contours in the scene as fallback."""
        (h, w) = img_np.shape[:2]
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        results = []
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < (h * w * 0.03):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            crop = img_np[y:y+bh, x:x+bw]
            crop_b64 = ""
            if crop.size > 0:
                _, buffer = cv2.imencode('.jpg', crop)
                crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

            results.append({
                "id": f"cnt_{idx}",
                "label": "Detected Object",
                "confidence": 75.0,
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "crop_b64": crop_b64
            })
        return results

    def crop_region(self, img_np, x, y, width, height):
        """Crops a user-selected bounding rectangle [x, y, w, h]."""
        (h, w) = img_np.shape[:2]
        startX = max(0, min(w - 1, int(x)))
        startY = max(0, min(h - 1, int(y)))
        endX = max(startX + 5, min(w, int(x + width)))
        endY = max(startY + 5, min(h, int(y + height)))

        crop = img_np[startY:endY, startX:endX]
        if crop.size == 0:
            return ""
        _, buffer = cv2.imencode('.jpg', crop)
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
