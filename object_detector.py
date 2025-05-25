import cv2
import numpy as np
from ultralytics import YOLO
import time
import logging
from threading import Lock
from tts import speak

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ObjectDetector:
    def __init__(self):
        # Distance calculation parameters
        self.FOCAL_LENGTH = 1000  # Camera focal length in pixels
        self.KNOWN_WIDTH_CM = 7  # Known width of objects in centimeters
        self.SAFE_DISTANCE_CM = 100  # Minimum safe distance in cm

        # Video stream settings
        self.FRAME_WIDTH = 640  # Output frame width
        self.FRAME_HEIGHT = 480  # Output frame height
        self.FPS_TARGET = 24  # Target frames per second
        self.FRAME_SKIP = 2  # Process every Nth frame (for performance)

        # Alert configuration
        self.CONTINUOUS_ALERT_INTERVAL = 2.0  # Seconds between repeat alerts
        self.MIN_ALERT_GAP = 0.5  # Minimum seconds between any two alerts
        self.ALERT_PERSISTENCE = 1.0  # Seconds to remember unseen objects

        # System state
        self.cap = None  # Video capture object
        self.running = False  # Detection running flag
        self.lock = Lock()  # Thread safety lock
        self.frame_count = 0  # Frame counter
        self.last_alert_time = 0  # Last time any alert was played
        self.tracked_objects = {}  # {obj_type: {'distance': float,
        #             'last_seen': float,
        #             'last_alert': float}}

        # Initialize YOLO model
        try:
            self.model = YOLO("yolov8n.pt").to('cpu')
            logging.info("YOLO model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            raise

    def estimate_distance(self, pixel_width):
        """Estimate distance to object using pixel width"""
        if pixel_width == 0:
            return 0
        return (self.KNOWN_WIDTH_CM * self.FOCAL_LENGTH) / pixel_width

    def process_frame(self, frame):
        """Process a single frame for object detection and alerts"""
        try:
            # Skip processing some frames for better performance
            self.frame_count += 1
            if self.frame_count % self.FRAME_SKIP != 0:
                return frame

            # Resize and detect objects
            frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            results = self.model.predict(frame, conf=0.5, imgsz=320, verbose=False)

            current_time = time.time()
            annotated = frame.copy()
            current_objects = {}

            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    # Extract object information
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pixel_width = x2 - x1
                    distance = self.estimate_distance(pixel_width)
                    obj_type = self.model.names[int(box.cls)]
                    current_objects[obj_type] = distance

                    # Visual feedback
                    if distance < self.SAFE_DISTANCE_CM:
                        color = (0, 0, 255)  # Red for unsafe
                        label = f"{obj_type} - UNSAFE ({int(distance)}cm)"
                    else:
                        color = (0, 255, 0)  # Green for safe
                        label = f"{obj_type} - SAFE ({int(distance)}cm)"

                    # Draw bounding box and label
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Update tracked objects and trigger alerts
            self.update_tracked_objects(current_objects, current_time)
            self.trigger_alerts(current_time)

            return annotated
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return frame

    def update_tracked_objects(self, current_objects, current_time):
        """Maintain state of tracked objects"""
        # Update existing objects
        for obj_type in list(self.tracked_objects.keys()):
            if obj_type in current_objects:
                self.tracked_objects[obj_type]['distance'] = current_objects[obj_type]
                self.tracked_objects[obj_type]['last_seen'] = current_time
            elif current_time - self.tracked_objects[obj_type]['last_seen'] > self.ALERT_PERSISTENCE:
                del self.tracked_objects[obj_type]  # Remove stale objects

        # Add new objects
        for obj_type, distance in current_objects.items():
            if obj_type not in self.tracked_objects:
                self.tracked_objects[obj_type] = {
                    'distance': distance,
                    'last_seen': current_time,
                    'last_alert': 0  # Never alerted before
                }

    def trigger_alerts(self, current_time):
        """Trigger voice alerts for unsafe objects"""
        for obj_type, data in self.tracked_objects.items():
            if data['distance'] < self.SAFE_DISTANCE_CM:
                # Check if time for a new alert
                if (current_time - data['last_alert'] >= self.CONTINUOUS_ALERT_INTERVAL and
                        current_time - self.last_alert_time >= self.MIN_ALERT_GAP):
                    speak(f"Warning! {obj_type} at {int(data['distance'])} centimeters")
                    self.tracked_objects[obj_type]['last_alert'] = current_time
                    self.last_alert_time = current_time

    def generate_frames(self):
        """Generator function for video frames"""
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.1)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Frame read failed, reinitializing camera")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    time.sleep(0.1)
                    continue

                processed_frame = self.process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', processed_frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ret:
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                time.sleep(1 / self.FPS_TARGET)
            except Exception as e:
                logging.error(f"Error in frame generation: {e}")
                break

    def start_detection(self):
        """Start the object detection system"""
        with self.lock:
            if not self.running:
                try:
                    # Try different camera indices
                    for i in range(3):
                        self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                        if self.cap.isOpened():
                            break

                    if not self.cap.isOpened():
                        logging.error("Could not open any camera device")
                        return False

                    # Optimize camera settings
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
                    self.cap.set(cv2.CAP_PROP_FPS, self.FPS_TARGET)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    self.running = True
                    logging.info("Detection started successfully")
                    return True
                except Exception as e:
                    logging.error(f"Error starting detection: {e}")
                    return False
            return True

    def stop_detection(self):
        """Stop the object detection system"""
        with self.lock:
            self.running = False
            try:
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                self.tracked_objects.clear()
                logging.info("Detection stopped")
            except Exception as e:
                logging.error(f"Error stopping detection: {e}")