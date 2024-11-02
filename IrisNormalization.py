import numpy as np
import cv2
from IrisLocalization import detect_iris_and_pupil

def normalize_iris(image, final_pupil_center, pupil_radius, iris_circle, M=64, N=512, initial_angle=0):
    """
    Normalize the iris by unwrapping it to a rectangular block with fixed size (M, N).
    
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


# Example usage function
def main():
    # Load and detect iris/pupil on an example image
    image_path = r"C:\Users\chris\OneDrive\Desktop\24fall\5293\Assignments_export\GroupProject\datasets\CASIA Iris Image Database (version 1.0)\023\1\023_1_1.bmp"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Detect iris and pupil
    final_pupil_center, pupil_radius, iris_circle = detect_iris_and_pupil(image_path)
    
    if final_pupil_center and iris_circle:
        # Normalize the iris with different initial angles
        for angle in [-9, -6, -3, 0, 3, 6, 9]:
            normalized_iris = normalize_iris(image, final_pupil_center, pupil_radius, iris_circle, initial_angle=angle)
            
            # Save or display the normalized iris image
            #cv2.imwrite(f"normalized_iris_angle_{angle}.png", normalized_iris)
            cv2.imshow(f"Normalized Iris Angle {angle}", normalized_iris)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Iris or pupil detection failed.")


if __name__ == "__main__":
    main()
