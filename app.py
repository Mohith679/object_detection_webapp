from flask import Flask, render_template, Response, jsonify, request
from object_detector import ObjectDetector
from tts import speak
import numpy as np
import cv2
import base64
import logging
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
detector = ObjectDetector()

@app.route('/')
def index():
    """Render the main interface"""
    app.logger.info("Serving index page")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route that serves the processed frames"""
    app.logger.info("Video feed endpoint accessed")
    return Response(
        detector.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/start', methods=['POST'])
def start_detection():
    """Start the object detection system"""
    app.logger.info("Start detection request received")
    if not detector.running:
        success = detector.start_detection()
        return jsonify({
            "status": "started" if success else "error",
            "message": "Detection started" if success else "Failed to start detection"
        })
    return jsonify({"status": "already running", "message": "Detection is already running"})

@app.route('/stop', methods=['POST'])
def stop_detection():
    """Stop the object detection system"""
    app.logger.info("Stop detection request received")
    if detector.running:
        detector.stop_detection()
        return jsonify({"status": "stopped", "message": "Detection stopped"})
    return jsonify({"status": "not running", "message": "Detection is not running"})

@app.route('/status')
def detection_status():
    """Check if detection is running"""
    return jsonify({"running": detector.running})

@app.route('/test_tts', methods=['POST'])
def test_tts():
    """Test the text-to-speech functionality"""
    try:
        speak("This is a test voice alert")
        return jsonify({
            "status": "success",
            "message": "Voice alert test initiated"
        })
    except Exception as e:
        app.logger.error(f"TTS test failed: {e}")
        return jsonify({
            "status": "error",
            "message": f"Voice alert test failed: {str(e)}"
        })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)