import numpy as np
import cv2
from IrisLocalization import detect_iris_and_pupil

def normalize_iris(image, final_pupil_center, pupil_radius, iris_circle, M=64, N=512, initial_angle=0):
    """
    Normalize the iris by unwrapping it to a rectangular block with fixed size (M, N).

    Logic:
    1. Converts angle offset from degrees to radians
    2. Creates empty array for normalized image
    3. Extracts pupil and iris parameters
    4. For each point in normalized image:
        - Calculates corresponding theta angle
        - Computes coordinates on pupil boundary
        - Computes coordinates on iris boundary
        - Uses linear interpolation between boundaries
        - Maps pixel values to normalized image
    
    Parameters:
        image (numpy.ndarray): Grayscale input image.
        final_pupil_center (tuple): Coordinates of the pupil center (x, y).
        pupil_radius (int): Radius of the pupil.
        iris_circle (tuple): Coordinates and radius of the iris circle (x, y, radius).
        M (int): Height of the normalized image (default 64).
        N (int): Width of the normalized image (default 512).
        initial_angle (float): Initial angle offset in degrees (e.g., -9, -6, -3, 0, 3, 6, 9).
    
    Returns:
        normalized_iris (numpy.ndarray): Normalized iris image of size MxN.
    """
    # Calculate the angle offset in radians
    initial_angle_rad = np.deg2rad(initial_angle)
    
    # Create an empty array for the normalized iris
    normalized_iris = np.zeros((M, N), dtype=image.dtype)

    # Extract the pupil and iris coordinates and radius
    x_p, y_p = final_pupil_center
    r_p = pupil_radius
    x_i, y_i, r_i = iris_circle

    # Iterate over each point in the normalized rectangular block
    for X in range(N):
        theta = 2 * np.pi * X / N + initial_angle_rad  # Add initial angle offset
        
        x_p_theta = x_p + r_p * np.cos(theta)
        y_p_theta = y_p + r_p * np.sin(theta)
        
        x_i_theta = x_i + r_i * np.cos(theta)
        y_i_theta = y_i + r_i * np.sin(theta)
        
        for Y in range(M):
            # Linear interpolation between pupil and iris boundary
            x = int(x_p_theta + (x_i_theta - x_p_theta) * Y / M)
            y = int(y_p_theta + (y_i_theta - y_p_theta) * Y / M)

            # Assign pixel value if coordinates are within image bounds
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                normalized_iris[Y, X] = image[y, x]
    
    return normalized_iris



