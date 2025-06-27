import cv2

def preprocess_image(image_path):
    """Preprocess image: convert to grayscale, blur, and threshold."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    processed_path = image_path.replace(".jpg", "_processed.jpg")
    cv2.imwrite(processed_path, thresh)
    return processed_path
