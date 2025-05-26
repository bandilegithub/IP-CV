import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Constants
DATA_DIR = "images"
RADIUS = 1
N_POINTS = 8 * RADIUS

# Feature extraction function
def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    edges = cv2.adaptiveThreshold(equalized, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coin_features = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = gray[y:y+h, x:x+w]

        # Shape features
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter != 0 else 0

        # Hu Moments
        hu_moments = cv2.HuMoments(cv2.moments(cnt)).flatten()

        # LBP features
        lbp = local_binary_pattern(roi, N_POINTS, RADIUS, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3),
                                 range=(0, N_POINTS + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)

        features = [area, perimeter, circularity] + hu_moments.tolist() + hist.tolist()
        coin_features.append(features)

    return coin_features

# Example structure to simulate dataset
labels = []
features = []

# Assume folder structure: /coin_images/{5c,10c,20c,...}/image.jpg
for coin_label in os.listdir(DATA_DIR):
    coin_dir = os.path.join(DATA_DIR, coin_label)
    if os.path.isdir(coin_dir):
        for image_name in os.listdir(coin_dir):
            image_path = os.path.join(coin_dir, image_name)
            feats = extract_features(image_path)
            for f in feats:
                features.append(f)
                labels.append(coin_label)


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=False)
report
