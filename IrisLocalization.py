import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def extract_roi(image, center_x, center_y, size=120):
    """
    Extract a region of interest (ROI) centered at the given point.

    Logic:
    - Extracts a square region of interest (ROI) centered at given coordinates.
    - Handles boundary cases to ensure the ROI stays within image borders.
    - Returns both the ROI and its position in the original image.

    Parameters:
    - image: Input grayscale image array.
    - center_x, center_y: Center coordinates of the ROI.
    - size: Size of the ROI window (default=120).
    
    Returns:
    - (ROI image, (x_start, y_start))
    """
    half_size = size // 2
    h, w = image.shape

    # Calculate ROI boundaries with padding
    x_start = max(0, center_x - half_size)
    x_end = min(w, center_x + half_size)
    y_start = max(0, center_y - half_size)
    y_end = min(h, center_y + half_size)

    return image[y_start:y_end, x_start:x_end], (x_start, y_start)

def preprocess_pupil_roi(roi, kernel_size=5):
    """
    Enhanced preprocessing for better pupil segmentation.

    Logic:
    1. Enhances contrast using CLAHE for better pupil visibility.
    2. Applies bilateral filtering to reduce noise while preserving edges.
    3. Uses blackhat morphology to enhance dark regions (pupil).
    4. Applies median blur for final noise reduction.

    Parameters:
    - roi: Input ROI image.
    - kernel_size: Kernel size for morphological operations (default=5).
    - clipLimit=3.0: CLAHE contrast limit.
    - tileGridSize=(8,8): CLAHE grid size.
    - Bilateral filter params: d=9, sigmaColor=75, sigmaSpace=75.

    Returns:
    - Preprocessed ROI image.
    """
    # Start with contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    roi_enhanced = clahe.apply(roi)
    
    # Apply bilateral filter to reduce noise while preserving edges
    roi_filtered = cv2.bilateralFilter(roi_enhanced, 9, 75, 75)
    
    # Create a larger kernel for more aggressive morphology
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size*2, kernel_size*2))
    
    # Apply black hat operation to enhance dark regions (pupil)
    blackhat = cv2.morphologyEx(roi_filtered, cv2.MORPH_BLACKHAT, kernel_large)
    
    # Add blackhat to original image to enhance dark regions
    roi_enhanced = cv2.add(roi_filtered, blackhat)
    
    # Apply median blur to reduce remaining noise
    roi_blurred = cv2.medianBlur(roi_enhanced, kernel_size)
    
    return roi_blurred

def otsu_threshold(image):
    """
    Logic:

    Preprocesses the input image using preprocess_pupil_roi.
    Applies Otsu's method for automatic thresholding.
    Cleans the binary image using morphological operations, to reduce the effect of eyelashes that affects finding countours of pupil

    Parameters:

    image: Input preprocessed image.
    kernel: 5x5 elliptical structuring element for morphology.
    Returns: Binary image with the pupil region highlighted.
    
    """
    # Apply preprocessing
    processed_image = preprocess_pupil_roi(image)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean up the binary image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def find_pupil_initial(image):
    """
    Find initial pupil location using projection method
    Logic:

    Extracts a subsection of the image where the pupil is likely located.
    Calculates vertical and horizontal projections.
    Finds the minimum projection values to estimate the pupil center.
    Adjusts coordinates to be relative to the original image.

    Parameters:

    image: Input grayscale image.
    subImage: Region [60:240, 100:220] for initial search.
    Returns: (x_center, y_center) as the estimated pupil center.
    
    """
    # First, take a subsection of the image
    subImage = image[60:240, 100:220]
    
    # Find minimum projections in the subImage
    vertical_projection = np.sum(subImage, axis=0)
    horizontal_projection = np.sum(subImage, axis=1)
    
    # Get coordinates relative to subImage
    x_center_sub = np.argmin(vertical_projection)
    y_center_sub = np.argmin(horizontal_projection)
    
    # Adjust coordinates to be relative to original image
    x_center = x_center_sub + 100  # Add x offset
    y_center = y_center_sub + 60   # Add y offset
    
    return x_center, y_center


def find_pupil_contour(binary_image, fallback_radius=20):
    """
    Find pupil center and radius using contour fitting with validation and fallback options.
    
    Logic:

    Finds contours in the binary image.
    Filters contours based on area and circularity.
    Fits an ellipse or minimum enclosing circle to the best contour.
    Includes a fallback mechanism for failed detection.
    
    Parameters:

    binary_image: Thresholded image.
    fallback_radius: Default radius if detection fails (default=20).
    area > 500: Minimum contour area threshold.
    circularity > 0.3: Minimum circularity threshold.
    Returns: (center, radius, contour, circularity).
    
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if not contours:
        print("No contours found in binary image. Using fallback.")
        return None, fallback_radius, None, 0.0  # Fallback values

    # Filter contours based on area and circularity
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        # Lower thresholds if no valid contours are found initially
        if area > 500 and circularity > 0.3:  # Adjusted circularity threshold
            valid_contours.append((contour, circularity))

    # Check if any valid contours were found after filtering
    if not valid_contours:
        print("No valid contours found after filtering. Trying fallback method.")
        return None, fallback_radius, None, 0.0  # Fallback values if no contours

    # Get the most circular contour
    pupil_contour, circularity = max(valid_contours, key=lambda x: x[1])

    # Use an ellipse or minimum enclosing circle as fallback if needed
    if len(pupil_contour) >= 5:  # Ellipse fitting requires at least 5 points
        ellipse = cv2.fitEllipse(pupil_contour)
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        radius = int((ellipse[1][0] + ellipse[1][1]) / 4)  # Average of major/minor axes
    else:
        # Fallback to minimum enclosing circle if ellipse fitting fails
        (x, y), radius = cv2.minEnclosingCircle(pupil_contour)
        center = (int(x), int(y))
        radius = int(radius)

    return center, radius, pupil_contour, circularity


def preprocess_iris_roi(roi):
    """
    Preprocess the ROI for iris boundary detection with enhanced contrast
    
    Logic:

    Enhances contrast using CLAHE.
    Applies bilateral filtering for noise reduction.
    Uses unsharp masking for edge enhancement.
    Applies adaptive thresholding for binarization.

    Parameters:

    roi: Input ROI image.
    clipLimit=3.0: CLAHE parameter.
    tileGridSize=(8,8): CLAHE grid size.
    Bilateral filter params: d=9, sigmaColor=75, sigmaSpace=75.
    Returns: (enhanced_image, thresholded_image).

    """
    # Apply CLAHE instead of simple histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    roi_eq = clahe.apply(roi)
    
    # Apply bilateral filter with adjusted parameters
    roi_filtered = cv2.bilateralFilter(roi_eq, 9, 75, 75)
    
    # Enhance edges using unsharp masking
    gaussian = cv2.GaussianBlur(roi_filtered, (0, 0), 3.0)
    roi_sharp = cv2.addWeighted(roi_filtered, 1.5, gaussian, -0.5, 0)
    
    # Apply adaptive thresholding with adjusted parameters
    roi_thresh = cv2.adaptiveThreshold(
        roi_sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    return roi_sharp, roi_thresh


def find_iris_boundary(image, pupil_center, pupil_radius):
    """
    Find iris boundary using edge detection and Hough transform with enhanced fallback mechanism
    
    Logic:

    Extracts a larger ROI around the pupil.
    Preprocesses the ROI for edge detection.
    Creates a mask focusing on the expected iris region.
    Detects the iris boundary using the Hough transform.
    Implements a scoring system for circle selection.
    
    Parameters:

    image: Input grayscale image.
    pupil_center: Coordinates of the detected pupil center.
    pupil_radius: Detected pupil radius.
    roi_size=240: Size of the iris ROI window.
    ideal_radius=95: Expected iris radius.
    max_allowed_center_dist: Maximum allowed distance from the pupil center.
    Hough parameters: dp=1, minDist=pupil_radius*1.5, param1=60, param2=20.
    
    Returns: (iris_circle, roi_filtered, roi_thresh, edges).

    """
    # Extract larger ROI for iris detection
    roi_size = 240  # Increased ROI size to accommodate larger iris
    roi, (roi_x, roi_y) = extract_roi(image, pupil_center[0], pupil_center[1], size=roi_size)
    
    # Preprocess the ROI
    roi_filtered, roi_thresh = preprocess_iris_roi(roi)
    
    # Create a mask to focus on the region around the pupil
    mask = np.zeros_like(roi_filtered)
    local_center = (roi_size // 2, roi_size // 2)  # Center of ROI
    
    # Create ring mask with adjusted proportions
    outer_radius = max(int(pupil_radius * 2), 180)  # Ensure minimum outer radius
    inner_radius = int(pupil_radius * 0)  # No inner mask to allow full detection
    
    cv2.circle(mask, local_center, outer_radius, 255, -1)
    cv2.circle(mask, local_center, inner_radius, 0, -1)
    
    # Apply mask
    masked_image = cv2.bitwise_and(roi_filtered, mask)
    
    # Edge detection with adjusted parameters
    edges = cv2.Canny(masked_image, 60, 100)
    
    # Hough circle detection with adjusted parameters
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=pupil_radius * 1.5,
        param1=60,
        param2=20,
        minRadius=80,
        maxRadius=int(pupil_radius * 4)
    )
    
    detected_iris = None
    best_score = 0.0
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Enhanced circle selection with scoring system
        valid_circles_with_scores = []
        ideal_radius = 95  # Target iris radius
        max_allowed_center_dist = pupil_radius * 0.6  # Maximum allowed distance from pupil center
        
        for circle in circles[0]:
            # Calculate center distance
            center_dist = np.sqrt((circle[0] - local_center[0]) ** 2 +
                                (circle[1] - local_center[1]) ** 2)
            
            # Basic validation criteria
            if (center_dist <= max_allowed_center_dist and 
                circle[2] >= 80 and
                circle[2] >= pupil_radius * 1.4):
                
                # Calculate scores for both metrics
                
                # 1. Radius score: how close is the radius to ideal_radius
                radius_diff = abs(circle[2] - ideal_radius)
                max_radius_diff = ideal_radius * 0.5  # Allow up to 50% deviation
                radius_score = 1.0 - min(radius_diff / max_radius_diff, 1.0)
                
                # 2. Center distance score: how close is the center to pupil center
                center_score = 1.0 - (center_dist / max_allowed_center_dist)
                
                # Calculate weighted final score
                radius_weight = 0.01  # Very small weight for radius
                center_weight = 0.99  # Almost all weight on center position
                
                final_score = (radius_score * radius_weight + 
                             center_score * center_weight)
                
                # Convert to global coordinates
                global_circle = [
                    circle[0] + roi_x,  # x coordinate
                    circle[1] + roi_y,  # y coordinate
                    circle[2]  # radius
                ]
                
                # Store circle with its score and individual metrics for debugging
                valid_circles_with_scores.append({
                    'circle': global_circle,
                    'score': final_score,
                    'radius_score': radius_score,
                    'center_score': center_score,
                    'radius': circle[2],
                    'center_dist': center_dist
                })
        
        if valid_circles_with_scores:
            # Sort by final score and get the best one
            valid_circles_with_scores.sort(key=lambda x: x['score'], reverse=True)
            best_match = valid_circles_with_scores[0]
            best_score = best_match['score']
            
            # Print debug information about the selected circle
            # print(f"\nSelected circle metrics:")
            # print(f"Final score: {best_match['score']:.3f}")
            # print(f"Radius score: {best_match['radius_score']:.3f} (radius: {best_match['radius']})")
            # print(f"Center score: {best_match['center_score']:.3f} (distance: {best_match['center_dist']:.1f})")
            
            # Only use detected iris if it meets all criteria:
            # 1. Score is good enough
            # 2. Radius is not too large
            if best_score >= 0.62 and best_match['radius'] <= 110:
                detected_iris = best_match['circle']
            # else:
            #     if best_score < 0.62:
            #         print(f"Best detection score ({best_score:.3f}) below threshold (0.62), using fallback.")
            #     if best_match['radius'] > 110:
            #         print(f"Detected radius ({best_match['radius']}) exceeds maximum (110), using fallback.")
    
    # Use fallback mechanism if no iris detected or score too low
    if detected_iris is None:
        #reason = "No iris detected" if circles is None or not valid_circles_with_scores else "Low detection score"
        #print(f"\n{reason}. Using fallback mechanism (pupil_radius + 57).")
        fallback_iris = [
            pupil_center[0],  # Use pupil center x
            pupil_center[1],  # Use pupil center y
            pupil_radius + 60  # Estimated radius using fallback
        ]
        return fallback_iris, roi_filtered, roi_thresh, edges
    
    return detected_iris, roi_filtered, roi_thresh, edges

def detect_iris_and_pupil(image_path):
    """
    Main function to detect both iris and pupil
    
    Logic:

    Loads the image in grayscale.
    Finds the initial pupil location.
    Extracts and processes the pupil ROI.
    Detects the pupil boundary using find_pupil_contour.
    Uses the pupil information to detect the iris boundary.
    Includes visualization capability (commented out).

    Parameters:

    image_path: Path to the input image file.
    Returns: (final_pupil_center, pupil_radius, iris_circle).
    
    """
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read the image at path: {image_path}")

    # Find initial pupil location
    x_center, y_center = find_pupil_initial(image)

    # Extract 120x120 ROI for pupil detection
    pupil_roi, (roi_x, roi_y) = extract_roi(image, x_center, y_center, size=120)

    # Apply Otsu's thresholding for pupil detection
    binary_roi = otsu_threshold(pupil_roi)

    # Find pupil using contour fitting
    center, pupil_radius, contour, circularity = find_pupil_contour(binary_roi)

    if center is None:
        print("Warning: Initial pupil detection failed. Using fallback values.")
        final_pupil_center = (x_center, y_center)
        pupil_radius = 45
        circularity = 0.0  # Set default circularity when detection fails
    else:
        # Convert ROI coordinates back to original image coordinates
        pupil_x = roi_x + center[0]
        pupil_y = roi_y + center[1]
        final_pupil_center = (pupil_x, pupil_y)

   
    # Find iris boundary
    iris_data = find_iris_boundary(image, final_pupil_center, pupil_radius)
    
    if iris_data is None:
        print("Warning: Iris boundary detection failed completely")
        return final_pupil_center, pupil_radius, None
        
    iris_circle, roi_filtered, roi_thresh, edges = iris_data

    # # Visualize results
    # plt.figure(figsize=(15, 10))

    # # Original image with detections
    # plt.subplot(2, 3, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('Original Image with Detections')

    # # Draw pupil
    # pupil = plt.Circle(final_pupil_center, pupil_radius, color='r',
    #                    fill=False, label='Pupil')
    # plt.gca().add_artist(pupil)

    # # Draw iris if found
    # if iris_circle is not None:
    #     iris = plt.Circle((iris_circle[0], iris_circle[1]), iris_circle[2],
    #                       color='g', fill=False, label='Iris')
    #     plt.gca().add_artist(iris)

    # plt.legend()

    # # Preprocessed ROI
    # plt.subplot(2, 3, 2)
    # if roi_filtered is not None:
    #     plt.imshow(roi_filtered, cmap='gray')
    #     plt.title('Preprocessed ROI')
    # else:
    #     plt.text(0.5, 0.5, 'ROI Processing Failed', ha='center')

    # # Thresholded ROI
    # plt.subplot(2, 3, 3)
    # if roi_thresh is not None:
    #     plt.imshow(roi_thresh, cmap='gray')
    #     plt.title('Thresholded ROI')
    # else:
    #     plt.text(0.5, 0.5, 'Thresholding Failed', ha='center')

    # # Edge detection
    # plt.subplot(2, 3, 4)
    # if edges is not None:
    #     plt.imshow(edges, cmap='gray')
    #     plt.title('Edge Detection')
    # else:
    #     plt.text(0.5, 0.5, 'Edge Detection Failed', ha='center')

    # # Binary pupil ROI
    # plt.subplot(2, 3, 5)
    # if binary_roi is not None:
    #     plt.imshow(binary_roi, cmap='gray')
    #     if contour is not None:
    #         plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2)
    #     plt.title(f'Binary Pupil ROI\nCircularity: {circularity:.3f}')
    # else:
    #     plt.text(0.5, 0.5, 'Binary ROI Failed', ha='center')

    # plt.tight_layout()
    # plt.show()

    return final_pupil_center, pupil_radius, iris_circle


