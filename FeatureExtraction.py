import cv2
import numpy as np
from scipy.signal import convolve2d


# Region of Interest (ROI): 48*512
def extract_roi(img):
    """
    Extract the top 48 rows as the Region of Interest (ROI) from enhanced image.

    Parameters:
        img (np.ndarray): The input grayscale image.

    Returns:
        np.ndarray: ROI image (48 * 512).
    """
    return img[:48, :]


# Defined Gabor filter
def modulate_frequency(x, y, freq_y):
    """
    Modulating function M1(x,y,f) based on frequency in the y-axis. Defined filter.

    Parameters:
        x, y (float): Coordinates in the filter.
        freq_y (float): Frequency parameter for y-axis.

    Returns:
        float: Modulated value at (x, y).
    """
    freq = 1 / freq_y
    return np.cos(2 * np.pi * freq * np.sqrt(x ** 2 + y ** 2))


def gabor_function(x, y, sigma_x, sigma_y):
    """
    Calculate Gabor function G(x,y,f) for given coordinates and standard deviations.

    Parameters:
        x, y (float): Coordinates in the filter.
        sigma_x, sigma_y (float): Standard deviations in x and y directions.

    Returns:
        float: Gabor filter value at (x, y).
    """
    gaussian = np.exp(-(x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2) / 2)
    return (1 / (2 * np.pi * sigma_x * sigma_y)) * gaussian * modulate_frequency(x, y, sigma_y)


def create_gabor_kernel(sigma_x, sigma_y, size=9):
    """
    Generate a Gabor kernel with specified sigma values and size.

    Parameters:
        sigma_x, sigma_y (float): Standard deviations for Gaussian envelope.
        size (int): Size of the kernel (size x size).

    Returns:
        np.ndarray: Gabor kernel.
    """
    kernel = np.zeros((size, size), dtype=np.float32)
    offset = size // 2
    for i in range(size):
        for j in range(size):
            kernel[i, j] = gabor_function(j - offset, i - offset, sigma_x, sigma_y)
    return kernel


def apply_gabor_filter(image, sigma_x, sigma_y):
    """
    Apply Gabor filter to an image using specified sigma values.

    Parameters:
        image (np.ndarray): Input grayscale image.
        sigma_x, sigma_y (float): Standard deviations for the Gabor filter.

    Returns:
        np.ndarray: Filtered image.
    """
    roi_img = extract_roi(image)
    gabor_kernel = create_gabor_kernel(sigma_x, sigma_y)
    filtered_img = convolve2d(roi_img, gabor_kernel, mode='same')
    return filtered_img


def compute_feature_vector(img1, img2, block_size=8):
    """
    Compute feature vector (1536*1) from two filtered images by calculating mean and
    standard deviation for each block.

    Parameters:
        img1, img2 (np.ndarray): Two filtered images.
        block_size (int): Size of each block for statistical analysis.

    Returns:
        np.ndarray: 1D feature vector.
    """
    rows = img1.shape[0] // block_size
    cols = img1.shape[1] // block_size
    feature_vec = []

    for img in [img1, img2]:
        for r in range(rows):
            for c in range(cols):
                block = img[r * block_size:(r + 1) * block_size, c * block_size:(c + 1) * block_size]
                mean_val = np.mean(np.abs(block))
                std_dev = np.mean(np.abs(block - mean_val))
                feature_vec.extend([mean_val, std_dev])

    return np.array(feature_vec)


# Example Usage
def extract_feature(iris_img):
    '''
    Logic:
    1. Applies two Gabor filters with different parameters
    2. Computes feature vector from filtered images

    Parameters:
    - `iris_img`: Input enhanced iris image
    - `sigma_x1=3.0, sigma_y1=1.5`: Parameters for first filter
    - `sigma_x2=4.5, sigma_y2=1.5`: Parameters for second filter
    - Returns: 1536-dimensional feature vector
    '''
    # Define parameters for 2 Gabor filters channels
    sigma_x1, sigma_y1 = 3.0, 1.5
    sigma_x2, sigma_y2 = 4.5, 1.5

    # Apply Gabor filters
    filtered_img1 = apply_gabor_filter(iris_img, sigma_x1, sigma_y1)
    filtered_img2 = apply_gabor_filter(iris_img, sigma_x2, sigma_y2)

    # Extract feature vector
    features = compute_feature_vector(filtered_img1, filtered_img2)

    # Check output feature vector
    # np.set_printoptions(threshold=np.inf)   # Option to print all elements
    # print("Feature vector length:", len(features))
    # print("Feature vector:", features)
    return features

