import cv2
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return img, blur

def load_dataset(image_root):
    X = []
    y = []
    for label in os.listdir(image_root):
        label_folder = os.path.join(image_root, label)
        if not os.path.isdir(label_folder):
            continue
        for fname in os.listdir(label_folder):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(label_folder, fname)
            img, blur = preprocess_image(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contours = segment_coins(blur)
            feats = extract_features(gray, contours)

            for f in feats:
                X.append(f)
                y.append(label)
    return np.array(X), np.array(y)

def segment_coins(blurred_img):
    edges = cv2.adaptiveThreshold(blurred_img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_glcm_features(gray_img, mask):
    coin = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    x, y, w, h = cv2.boundingRect(mask)
    coin_crop = coin[y:y+h, x:x+w]
    glcm = graycomatrix(coin_crop, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return [contrast, energy, homogeneity, correlation]

def extract_shape_features(image):
    # Threshold the image to get a binary image
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None

    # Assume the largest contour is the coin
    c = max(contours, key=cv2.contourArea)

    # Hu Moments (7 values)
    hu_moments = cv2.HuMoments(cv2.moments(c)).flatten()

    # Contour descriptors
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0
    # Eccentricity calculation
    (x, y), (MA, ma), angle = cv2.fitEllipse(c) if len(c) >= 5 else ((0,0),(0,0),0)
    eccentricity = np.sqrt(1 - (MA / ma) ** 2) if ma != 0 else 0

    contour_features = {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "eccentricity": eccentricity
    }

    return hu_moments, contour_features

def extract_features(gray_img, contours):
    features = []
    for cnt in contours:
        mask = np.zeros(gray_img.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        feats = extract_glcm_features(gray_img, mask)
        features.append(feats)
    return np.array(features)

def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return clf

def classify_coin(features, classifier):
    return classifier.predict(features)

def main(image_path, classifier):
    img, blur = preprocess_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours = segment_coins(blur)
    features = extract_features(gray, contours)
    labels = classify_coin(features, classifier)
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, labels[i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.imshow("Detected Coins", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    X, y = load_dataset("images")
    classifier = train_classifier(X, y)
    main("images/R5/R5heads4.png", classifier)