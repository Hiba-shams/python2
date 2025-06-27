import cv2
import numpy as np
import os

class ImageQualityInspector:
    def __init__(self, 
                 face_cascade_path=None, 
                 face_region_threshold=0.6, 
                 blur_threshold=100.0, 
                 debug=False):
        """Initialize the Inspector with thresholds and settings."""
        self.face_cascade = cv2.CascadeClassifier(
            face_cascade_path or (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        )
        self.face_region_threshold = face_region_threshold
        self.blur_threshold = blur_threshold
        self.debug = debug

    def _load_image(self, image_path):
        """Load an image from a file path and handle errors."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img

    def detect_single_face(self, image_path):
        """Detect if exactly one face exists and it's in the expected region."""
        img = self._load_image(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)  # Improve reliability
        )

        if self.debug:
            print(f"Detected {len(faces)} face(s).")

        if len(faces) != 1:
            return False, "Expected exactly one face"

        x, y, w, h = faces[0]
        img_height, img_width = img.shape[:2]

        # Intelligent face position validation
        upper_region_limit = self.face_region_threshold * img_height
        if (y + h/2) > upper_region_limit:
            return False, "Face is too low in the image"

        if self.debug:
            # Visualize detected face
            img_copy = img.copy()
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Face Detection", img_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return True, "Face detected properly"

    def is_image_blurry(self, image_path):
        """Check if the image is blurry using variance of Laplacian."""
        img = self._load_image(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        if self.debug:
            print(f"Blurriness score (variance of Laplacian): {variance:.2f}")

        return variance < self.blur_threshold, variance

    def evaluate_image(self, image_path):
        """Combined evaluation: face detection and blur detection."""
        results = {}

        # Check face
        try:
            face_ok, face_message = self.detect_single_face(image_path)
            results['face_check'] = (face_ok, face_message)
        except Exception as e:
            results['face_check'] = (False, f"Face check failed: {str(e)}")

        # Check blur
        try:
            blurry, variance = self.is_image_blurry(image_path)
            results['blurry_check'] = (not blurry, f"Blurriness score: {variance:.2f}")
        except Exception as e:
            results['blurry_check'] = (False, f"Blurry check failed: {str(e)}")

        return results
