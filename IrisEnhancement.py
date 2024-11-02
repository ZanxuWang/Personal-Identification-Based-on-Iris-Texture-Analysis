import cv2
import numpy as np
from IrisLocalization import detect_iris_and_pupil
from IrisNormalization import normalize_iris

def enhance_iris(normalized_iris, block_size=16, enhancement_block_size=32):
    """
    Enhance the normalized iris image by compensating for lighting variations
    and improving contrast using histogram equalization.
    
    Parameters:
        normalized_iris (numpy.ndarray): The normalized iris image (MxN).
        block_size (int): Size of each block for background illumination estimation (default 16x16).
        enhancement_block_size (int): Size of each region for local histogram equalization (default 32x32).
    
    Returns:
        enhanced_iris (numpy.ndarray): The enhanced iris image.
    """
    # Step 1: Estimate background illumination using block-wise averaging
    h, w = normalized_iris.shape
    background = np.zeros((h // block_size, w // block_size), dtype=np.float32)
    
    # Calculate mean intensity for each block
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = normalized_iris[i:i+block_size, j:j+block_size]
            mean_intensity = np.mean(block)
            background[i // block_size, j // block_size] = mean_intensity
    
    # Resize the estimated background illumination to the size of the normalized image using bicubic interpolation
    background_resized = cv2.resize(background, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Subtract the background illumination from the normalized iris image to obtain lighting-compensated image
    lighting_corrected = cv2.subtract(normalized_iris.astype(np.float32), background_resized)
    lighting_corrected = np.clip(lighting_corrected, 0, 255).astype(np.uint8)
    
    # Step 2: Apply local histogram equalization for contrast enhancement
    enhanced_iris = np.zeros_like(lighting_corrected)
    for i in range(0, h, enhancement_block_size):
        for j in range(0, w, enhancement_block_size):
            # Extract 32x32 block
            block = lighting_corrected[i:i+enhancement_block_size, j:j+enhancement_block_size]
            
            # Perform histogram equalization on the block
            block_equalized = cv2.equalizeHist(block)
            
            # Place back the equalized block
            enhanced_iris[i:i+enhancement_block_size, j:j+enhancement_block_size] = block_equalized
    
    return enhanced_iris

# Example usage function
def main():
    # Load and detect iris/pupil on an example image
    image_path = r"C:\Users\chris\OneDrive\Desktop\24fall\5293\Assignments_export\GroupProject\datasets\CASIA Iris Image Database (version 1.0)\023\1\023_1_1.bmp"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Detect iris and pupil
    final_pupil_center, pupil_radius, iris_circle = detect_iris_and_pupil(image_path)
    
    if final_pupil_center and iris_circle:
        # Normalize the iris
        normalized_iris = normalize_iris(image, final_pupil_center, pupil_radius, iris_circle, initial_angle=0)
        
        # Enhance the normalized iris image
        enhanced_iris = enhance_iris(normalized_iris)
        
        # Save or display the enhanced iris image
        cv2.imwrite("enhanced_iris.png", enhanced_iris)
        cv2.imshow("Enhanced Iris", enhanced_iris)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Iris or pupil detection failed.")


if __name__ == "__main__":
    main()
