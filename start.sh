#!/usr/bin/env bash
# Portable Auto-Launcher for ObjectSight_AI
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "   🚀 Launching ObjectSight AI Robotics Vision System"
echo "=========================================================="

if [ ! -d "venv" ] || ! ./venv/bin/python -c "import flask" 2>/dev/null; then
    echo "📦 Creating fresh local virtual environment..."
    rm -rf venv
    python3 -m venv venv
    echo "📥 Installing requirements..."
    ./venv/bin/pip install -r requirements.txt
fi

echo "✅ Environment ready! Launching Flask server..."
./venv/bin/python app.py
