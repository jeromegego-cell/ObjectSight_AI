/**
 * ObjectSight AI - Client Engine
 * Integrates TensorFlow.js COCO-SSD real-time object detection with backend Internet Search Engine APIs.
 */

// Global State
let cocoModel = null;
let currentStream = null;
let isWebcamActive = false;
let currentMode = 'webcam'; // 'webcam' | 'upload'
let activeObjects = [];
let confidenceThreshold = 0.40;
let isCropping = false;
let cropStartPoint = null;
let cropEndPoint = null;
let currentKnowledgeData = null;

// DOM Elements
const videoEl = document.getElementById('webcam-feed');
const imageEl = document.getElementById('image-feed');
const canvasEl = document.getElementById('vision-canvas');
const ctx = canvasEl.getContext('2d');
const scannerEl = document.getElementById('scanner');
const loaderEl = document.getElementById('loader');
const chipsContainer = document.getElementById('chips-container');
const historyList = document.getElementById('history-list');

// Initialize App
window.addEventListener('DOMContentLoaded', async () => {
  setupCanvasSize();
  window.addEventListener('resize', setupCanvasSize);
  
  // Attach Canvas Mouse Listeners for Bounding Box Click & Manual Cropping
  canvasEl.addEventListener('mousedown', handleCanvasMouseDown);
  canvasEl.addEventListener('mousemove', handleCanvasMouseMove);
  canvasEl.addEventListener('mouseup', handleCanvasMouseUp);

  // Load TensorFlow.js COCO-SSD Model
  try {
    updateStatus('LOADING AI MODEL...');
    if (window.cocoSsd) {
      cocoModel = await cocoSsd.load();
      updateStatus('AI MODEL READY');
      console.log('COCO-SSD model loaded successfully!');
    } else {
      updateStatus('BACKEND VISION READY');
    }
  } catch (err) {
    console.warn('TF.js model load issue, using server backend vision:', err);
    updateStatus('SERVER VISION ACTIVE');
  }

  // Load history
  fetchHistory();
});

function updateStatus(msg) {
  const statusTxt = document.getElementById('status-text');
  if (statusTxt) statusTxt.textContent = msg;
}

function setupCanvasSize() {
  const box = document.getElementById('viewport-box');
  canvasEl.width = box.clientWidth;
  canvasEl.height = box.clientHeight;
  redrawCanvas();
}

// ----------------------------------------------------
// Mode Switcher: Webcam vs Upload
// ----------------------------------------------------
function switchMode(mode) {
  currentMode = mode;
  document.getElementById('tab-webcam').classList.toggle('active', mode === 'webcam');
  document.getElementById('tab-upload').classList.toggle('active', mode === 'upload');

  if (mode === 'webcam') {
    imageEl.style.display = 'none';
    videoEl.style.display = 'block';
    if (!isWebcamActive) {
      startWebcam();
    }
  } else {
    stopWebcam();
    videoEl.style.display = 'none';
    imageEl.style.display = 'block';
  }
}

// ----------------------------------------------------
// Live Webcam Handler
// ----------------------------------------------------
async function toggleWebcam() {
  if (isWebcamActive) {
    stopWebcam();
  } else {
    switchMode('webcam');
    await startWebcam();
  }
}

async function startWebcam() {
  try {
    const btn = document.getElementById('btn-toggle-cam');
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Stop Camera`;
    
    currentStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    videoEl.srcObject = currentStream;
    isWebcamActive = true;
    scannerEl.style.display = 'block';

    videoEl.onloadedmetadata = () => {
      runWebcamDetectionLoop();
    };
  } catch (err) {
    alert('Webcam access error or denied: ' + err.message);
    stopWebcam();
  }
}

function stopWebcam() {
  isWebcamActive = false;
  scannerEl.style.display = 'none';
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
    currentStream = null;
  }
  videoEl.srcObject = null;
  const btn = document.getElementById('btn-toggle-cam');
  btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Start Camera`;
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
}

// ----------------------------------------------------
// Real-time Detection Loop
// ----------------------------------------------------
async function runWebcamDetectionLoop() {
  if (!isWebcamActive) return;

  if (cocoModel && videoEl.readyState === 4) {
    try {
      const predictions = await cocoModel.detect(videoEl);
      processPredictions(predictions, videoEl.videoWidth, videoEl.videoHeight);
    } catch (e) {
      console.error('Detection loop error:', e);
    }
  } else if (!cocoModel && isWebcamActive) {
    // Call server backend every 1.5 seconds if local model unavailable
    await captureAndSendToServer();
    await new Promise(r => setTimeout(r, 1200));
  }

  if (isWebcamActive) {
    requestAnimationFrame(runWebcamDetectionLoop);
  }
}

// Map predictions to viewport canvas coordinates
function processPredictions(predictions, sourceW, sourceH) {
  const canvasW = canvasEl.width;
  const canvasH = canvasEl.height;
  const scaleX = canvasW / sourceW;
  const scaleY = canvasH / sourceH;

  activeObjects = predictions
    .filter(p => p.score >= confidenceThreshold)
    .map((p, idx) => {
      const [x, y, w, h] = p.bbox;
      return {
        id: `obj_${idx}`,
        label: p.class,
        confidence: Math.round(p.score * 100),
        bbox: [x * scaleX, y * scaleY, w * scaleX, h * scaleY],
        rawBbox: p.bbox
      };
    });

  updateDetectedChips();
  redrawCanvas();
}

// ----------------------------------------------------
// Canvas Drawing & Bounding Boxes
// ----------------------------------------------------
function redrawCanvas() {
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  // Draw detected object bounding boxes
  activeObjects.forEach(obj => {
    const [x, y, w, h] = obj.bbox;

    // Glowing box border
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 8;
    ctx.strokeRect(x, y, w, h);

    // Semi-transparent fill
    ctx.fillStyle = 'rgba(0, 240, 255, 0.08)';
    ctx.fillRect(x, y, w, h);

    // Label tag badge
    const labelText = `${obj.label.toUpperCase()} (${obj.confidence}%)`;
    ctx.font = 'bold 12px "JetBrains Mono", monospace';
    const textWidth = ctx.measureText(labelText).width;

    ctx.fillStyle = '#00f0ff';
    ctx.shadowBlur = 0;
    ctx.fillRect(x, Math.max(0, y - 24), textWidth + 14, 22);

    ctx.fillStyle = '#050811';
    ctx.fillText(labelText, x + 7, Math.max(15, y - 8));
  });

  // Draw manual crop selection box if active
  if (cropStartPoint && cropEndPoint) {
    const x = Math.min(cropStartPoint.x, cropEndPoint.x);
    const y = Math.min(cropStartPoint.y, cropEndPoint.y);
    const w = Math.abs(cropEndPoint.x - cropStartPoint.x);
    const h = Math.abs(cropEndPoint.y - cropStartPoint.y);

    ctx.strokeStyle = '#ff0055';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255, 0, 85, 0.15)';
    ctx.fillRect(x, y, w, h);
  }
}

// Update list of detected chips under viewport
function updateDetectedChips() {
  if (activeObjects.length === 0) {
    chipsContainer.innerHTML = `<span style="color: var(--text-muted); font-size: 0.85rem; font-family: var(--font-mono);">No objects detected. Point camera or pick a sample image.</span>`;
    return;
  }

  chipsContainer.innerHTML = activeObjects.map(obj => `
    <div class="object-chip" onclick="searchObject('${obj.label}')">
      <span>${obj.label}</span>
      <span class="chip-conf">${obj.confidence}%</span>
    </div>
  `).join('');
}

// ----------------------------------------------------
// File Upload & Sample Media
// ----------------------------------------------------
function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(e) {
    loadSampleImage(e.target.result, 'Uploaded Media');
  };
  reader.readAsDataURL(file);
}

function loadSampleImage(src, labelHint) {
  switchMode('upload');
  imageEl.src = src;
  imageEl.onload = async () => {
    setupCanvasSize();
    scannerEl.style.display = 'block';
    
    // Process with TF.js or Server
    if (cocoModel) {
      const predictions = await cocoModel.detect(imageEl);
      processPredictions(predictions, imageEl.naturalWidth || imageEl.width, imageEl.naturalHeight || imageEl.height);
      scannerEl.style.display = 'none';

      // Auto-trigger search for top predicted object
      if (predictions.length > 0) {
        searchObject(predictions[0].class);
      } else if (labelHint) {
        searchObject(labelHint);
      }
    } else {
      await sendImageB64ToServer(src, labelHint);
      scannerEl.style.display = 'none';
    }
  };
}

// ----------------------------------------------------
// Server Backend API Fallback
// ----------------------------------------------------
async function captureAndSendToServer() {
  if (!videoEl || videoEl.videoWidth === 0) return;

  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = videoEl.videoWidth;
  tempCanvas.height = videoEl.videoHeight;
  const tCtx = tempCanvas.getContext('2d');
  tCtx.drawImage(videoEl, 0, 0);

  const b64 = tempCanvas.toDataURL('image/jpeg', 0.7);
  await sendImageB64ToServer(b64);
}

async function sendImageB64ToServer(b64, labelHint = '') {
  try {
    const resp = await fetch('/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_b64: b64, confidence: confidenceThreshold })
    });
    const data = await resp.json();

    if (data.success && data.objects) {
      const imgW = data.image_dimensions ? data.image_dimensions.width : canvasEl.width;
      const imgH = data.image_dimensions ? data.image_dimensions.height : canvasEl.height;

      const scaleX = canvasEl.width / imgW;
      const scaleY = canvasEl.height / imgH;

      activeObjects = data.objects.map((obj, idx) => ({
        id: obj.id || `obj_${idx}`,
        label: obj.label,
        confidence: obj.confidence,
        bbox: [obj.bbox[0] * scaleX, obj.bbox[1] * scaleY, obj.bbox[2] * scaleX, obj.bbox[3] * scaleY]
      }));

      updateDetectedChips();
      redrawCanvas();

      if (activeObjects.length > 0) {
        searchObject(activeObjects[0].label);
      } else if (labelHint) {
        searchObject(labelHint);
      }
    }
  } catch (err) {
    console.error('Server detect error:', err);
  }
}

// ----------------------------------------------------
// Bounding Box Click & Manual Region Crop
// ----------------------------------------------------
function toggleCropMode() {
  isCropping = !isCropping;
  const btn = document.getElementById('btn-crop-mode');
  btn.style.borderColor = isCropping ? 'var(--accent-pink)' : '';
  btn.style.color = isCropping ? 'var(--accent-pink)' : '';
  if (isCropping) {
    alert('Crop Mode Active: Click and drag on the camera/image viewport to select any region to search!');
  }
}

function handleCanvasMouseDown(e) {
  const rect = canvasEl.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  if (isCropping) {
    cropStartPoint = { x: clickX, y: clickY };
    cropEndPoint = null;
    return;
  }

  // Check if click hits a detected bounding box
  const clickedObj = activeObjects.find(obj => {
    const [x, y, w, h] = obj.bbox;
    return clickX >= x && clickX <= (x + w) && clickY >= y && clickY <= (y + h);
  });

  if (clickedObj) {
    searchObject(clickedObj.label);
  }
}

function handleCanvasMouseMove(e) {
  if (isCropping && cropStartPoint) {
    const rect = canvasEl.getBoundingClientRect();
    cropEndPoint = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
    redrawCanvas();
  }
}

function handleCanvasMouseUp(e) {
  if (isCropping && cropStartPoint && cropEndPoint) {
    const w = Math.abs(cropEndPoint.x - cropStartPoint.x);
    const h = Math.abs(cropEndPoint.y - cropStartPoint.y);

    if (w > 15 && h > 15) {
      const queryPrompt = prompt("Enter object name or description for this cropped region:", "Object Detail");
      if (queryPrompt) {
        searchObject(queryPrompt);
      }
    }
    cropStartPoint = null;
    cropEndPoint = null;
    isCropping = false;
    toggleCropMode();
    redrawCanvas();
  }
}

// ----------------------------------------------------
// Internet Intelligence Search Engine
// ----------------------------------------------------
function handleManualSearch(e) {
  e.preventDefault();
  const input = document.getElementById('manual-search-input');
  const val = input.value.trim();
  if (val) {
    searchObject(val);
  }
}

function updateConfidence(val) {
  confidenceThreshold = parseFloat(val) / 100.0;
  document.getElementById('conf-val').textContent = `${val}%`;
}

async function searchObject(queryTerm) {
  if (!queryTerm) return;

  loaderEl.style.display = 'flex';

  try {
    const resp = await fetch(`/api/search?q=${encodeURIComponent(queryTerm)}`);
    const result = await resp.json();

    if (result.success && result.data) {
      renderKnowledgeCard(result.data);
      fetchHistory();
    }
  } catch (err) {
    console.error('Search API failed:', err);
    alert('Internet search failed: ' + err.message);
  } finally {
    loaderEl.style.display = 'none';
  }
}

function renderKnowledgeCard(data) {
  currentKnowledgeData = data;

  document.getElementById('obj-title').textContent = data.title || data.query;
  document.getElementById('obj-category').textContent = data.category || 'Object Class';
  document.getElementById('obj-summary').textContent = data.summary || 'No summary available.';

  // Image Thumbnail
  const imgEl = document.getElementById('obj-image');
  if (data.image_url) {
    imgEl.src = data.image_url;
    imgEl.style.display = 'block';
  } else {
    imgEl.style.display = 'none';
  }

  // Key Facts List
  const factsEl = document.getElementById('obj-facts');
  if (data.key_facts && data.key_facts.length > 0) {
    factsEl.innerHTML = data.key_facts.map(f => `<li>${f}</li>`).join('');
  } else {
    factsEl.innerHTML = `<li>No specific key facts recorded.</li>`;
  }

  // Specs Table
  const specsEl = document.getElementById('obj-specs');
  if (data.specifications && Object.keys(data.specifications).length > 0) {
    specsEl.innerHTML = Object.entries(data.specifications).map(([k, v]) => `
      <tr>
        <td>${k}</td>
        <td>${v}</td>
      </tr>
    `).join('');
  }

  // Wikipedia Link
  const linkEl = document.getElementById('obj-link');
  if (data.source_url) {
    linkEl.href = data.source_url;
    linkEl.style.display = 'inline-block';
  } else {
    linkEl.style.display = 'none';
  }
}

// ----------------------------------------------------
// Voice Reader (Text-To-Speech API)
// ----------------------------------------------------
function speakDetails() {
  if (!currentKnowledgeData) return;

  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel(); // Stop any ongoing speech

    const textToSpeak = `${currentKnowledgeData.title}. Category: ${currentKnowledgeData.category}. ${currentKnowledgeData.summary}`;
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  } else {
    alert('Text-to-Speech is not supported in this browser.');
  }
}

// ----------------------------------------------------
// Search History
// ----------------------------------------------------
async function fetchHistory() {
  try {
    const resp = await fetch('/api/history');
    const data = await resp.json();

    if (data.success && data.history) {
      if (data.history.length === 0) {
        historyList.innerHTML = `<div style="color: var(--text-muted); font-size: 0.8rem; font-family: var(--font-mono); text-align: center; padding: 10px;">History log is empty</div>`;
        return;
      }

      historyList.innerHTML = data.history.map(item => `
        <div class="history-item" onclick="searchObject('${item.query}')">
          <span>${item.title}</span>
          <span class="history-time">${item.timestamp.split(' ')[1] || ''}</span>
        </div>
      `).join('');
    }
  } catch (err) {
    console.error('History fetch error:', err);
  }
}

async function clearSearchHistory() {
  await fetch('/api/history', { method: 'DELETE' });
  fetchHistory();
}
