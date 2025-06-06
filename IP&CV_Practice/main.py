import cv2
import numpy as np
import matplotlib.pyplot as plt



def read_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def compute_and_plot_histogram(image):
    # Compute histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Plot histogram as bars
    plt.bar(range(256), hist.flatten(), width=1.0, edgecolor='black')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Grayscale Image Histogram')
    plt.show()

def filter_smoothing(image):
    gaussian_blurred = cv2.GaussianBlur(image, (3, 3), 0)
    median_filtered = cv2.medianBlur(image, 3)
    return gaussian_blurred, median_filtered

def thresholding(image):
    adaptive_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    _, global_thresholding = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return adaptive_threshold, global_thresholding

def laplacian_of_gaussian(image, sigma=1.0):
    # Create a Gaussian kernel
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = gaussian_kernel * gaussian_kernel.T

    # Apply Gaussian filter
    smoothed_image = cv2.filter2D(image, -1, gaussian_kernel)

    # Compute Laplacian
    laplacian = cv2.Laplacian(smoothed_image, cv2.CV_64F)

    return laplacian

def preprocess_image(image):
    gaussian_blurred, median_filtered = filter_smoothing(image)

    equalised = cv2.equalizeHist(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(image)
    laplacian_of_gaussian_result = laplacian_of_gaussian(cl1)
    return equalised, cl1, median_filtered, gaussian_blurred, laplacian_of_gaussian_result

def segment_image(image):
    adaptive_thresh, global_thresh = thresholding(image)
    laplacian = cv2.Laplacian(global_thresh, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    canny = cv2.Canny(image, 180, 230)
    return laplacian_abs, global_thresh, canny

def laplacian_sharpening(image):
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    # Add Laplacian to original image for sharpening
    sharpened = cv2.addWeighted(image, 1.0, laplacian_abs, 1.0, 0)
    return sharpened, laplacian_abs

def sobel_edges(image):
    # Sobel edge detection in x and y directions
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # Combine the two gradients
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    return sobel_combined

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


if __name__ == "__main__":
    #../IP&CV_Project/images/10c/10cHeads2.png
    image = read_image("../IP&CV_Project/images/R2/R2Heads1.png")
    equalised, cl1, median_filtered, blur, laplacian_of_gaussian = preprocess_image(image)
    gaussian, median = filter_smoothing(image)

    laplacian_sharp, laplacian_edges = laplacian_sharpening(image)
    laplacian, thresh, edges = segment_image(cl1)
    sobel = sobel_edges(image)

    hu, contour_desc = extract_shape_features(image)
    print("Hu Moments:", hu)
    print("Contour Descriptors:", contour_desc)
    cv2.imshow("Original Image", image)
    cv2.imshow("Sobel", sobel)
    #cv2.imshow("laplacian of gaussian", gaussian)
    #cv2.imshow("median", median)
    #cv2.imshow("Equalised", equalised)
    #cv2.imshow("CLAHE", cl1)
    #cv2.imshow("Thresholding", thresh)
    #cv2.imshow("canny", edges)
    #cv2.imshow("Laplacian", laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()