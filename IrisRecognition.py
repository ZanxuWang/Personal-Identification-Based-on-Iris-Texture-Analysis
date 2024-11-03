# main.py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from IrisLocalization import detect_iris_and_pupil
from IrisNormalization import normalize_iris
from IrisEnhancement import enhance_iris
from FeatureExtraction import extract_feature
from PerformanceEvaluation import *
from IrisMatching import run_lda_analysis

def main():
    """
    Orchestrates the entire iris recognition process by executing all major steps:
    1. Data Preparation:
        - Defines the dataset path.
        - Initializes lists to store training and testing features and labels.
        - Processes training and testing images to extract features.
    2. Iris Localization and Normalization:
        - Detects the iris and pupil boundaries in each image.
        - Normalizes the iris region to a fixed size.
        - Enhances the normalized iris image for better feature extraction.
    3. Feature Extraction:
        - Extracts discriminative features from the enhanced iris images using Gabor filters.
        - Handles rotation compensation by processing images at multiple angles.
    4. Template Generation:
        - Computes mean feature vectors (templates) for each subject from the training data.
    5. Recognition Performance Evaluation:
        - Evaluates the system's identification performance using different distance metrics.
        - Performs dimensionality reduction using LDA and re-evaluates performance.
        - Visualizes the impact of dimensionality reduction on accuracy.
    6. Verification Performance Analysis:
        - Analyzes the system's verification performance by plotting ROC curves.
        - Compares performance before and after applying LDA.

    Parameters:
        None

    Returns:
        None

    Notes:
        - The function assumes a specific directory structure for the dataset.
        - It uses several helper functions from different modules to perform specialized tasks.
        - Intermediate results like feature vectors and templates are stored in memory.
        - Outputs include printed accuracy metrics and plotted figures for analysis.
    """
    # Dataset path，plz set to your own dataset path before running it
    base_dir = r".\datasets\CASIA Iris Image Database (version 1.0)"
    
    # Extract training features
    print("Extracting training features...")
    train_features = []
    train_labels = []
    base = Path(base_dir)
    
    # Process training images
    for subject in tqdm(range(1, 109), desc="Processing training subjects"):
        subject_dir = f"{subject:03d}"
        current_dir = base / subject_dir / '1'
        
        if not current_dir.exists():
            continue
            
        for img_file in current_dir.glob('*.bmp'):
            image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            final_pupil_center, pupil_radius, iris_circle = detect_iris_and_pupil(str(img_file))
            
            if final_pupil_center and iris_circle:
                normalized_iris = normalize_iris(image, final_pupil_center, 
                                              pupil_radius, iris_circle, 
                                              initial_angle=0)
                enhanced_iris = enhance_iris(normalized_iris)
                feature = extract_feature(enhanced_iris)
                train_features.append(feature)
                train_labels.append(subject)
    
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    
    # Extract testing features
    print("Extracting testing features...")
    test_features = []
    test_labels = []
    angles = [-9, -6, -3, 0, 3, 6, 9]
    
    # Process testing images
    for subject in tqdm(range(1, 109), desc="Processing testing subjects"):
        subject_dir = f"{subject:03d}"
        current_dir = base / subject_dir / '2'
        
        if not current_dir.exists():
            continue
            
        for img_file in current_dir.glob('*.bmp'):
            image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            final_pupil_center, pupil_radius, iris_circle = detect_iris_and_pupil(str(img_file))
            
            if final_pupil_center and iris_circle:
                for angle in angles:
                    normalized_iris = normalize_iris(image, final_pupil_center, 
                                                  pupil_radius, iris_circle, 
                                                  initial_angle=angle)
                    enhanced_iris = enhance_iris(normalized_iris)
                    feature = extract_feature(enhanced_iris)
                    test_features.append(feature)
                    test_labels.append(subject)
    
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)
    
    # Compute templates for original features
    templates = compute_templates(train_features, train_labels)
    
    # Evaluate original features
    print("Evaluating original features...")
    acc_l1, acc_l2, acc_cos = evaluate_recognition(test_features, test_labels, templates)
    print(f"Original Features Accuracies:")
    print(f"L1: {acc_l1:.4f}")
    print(f"L2: {acc_l2:.4f}")
    print(f"Cosine: {acc_cos:.4f}")
    
    # LDA Analysis
    dimensions = [107, 100, 90, 80, 70, 60, 50, 40, 30, 20]
    print("Performing LDA analysis...")
    lda_results = run_lda_analysis(train_features, train_labels, test_features, test_labels, dimensions)
    
    # Create accuracy vs dimensions plot
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, lda_results[:, 3], marker='o')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Accuracy (Cosine Similarity)')
    plt.title('LDA Dimension Reduction: Accuracy vs Dimensions')
    plt.grid(True)
    plt.show()
    
    # Print comparison table
    print("\nComparison Results:")
    print("-" * 60)
    print(f"{'Metric':<15} {'Original':<15} {'LDA (107d)':<15} {'Difference':<15}")
    print("-" * 60)
    
    # Get LDA results for 107 dimensions (first row of lda_results)
    lda_l1, lda_l2, lda_cos = lda_results[0, 1], lda_results[0, 2], lda_results[0, 3]
    
    # Print results with differences
    print(f"{'L1 Distance':<15} {acc_l1:>14.4f} {lda_l1:>14.4f} {(lda_l1-acc_l1):>14.4f}")
    print(f"{'L2 Distance':<15} {acc_l2:>14.4f} {lda_l2:>14.4f} {(lda_l2-acc_l2):>14.4f}")
    print(f"{'Cosine':<15} {acc_cos:>14.4f} {lda_cos:>14.4f} {(lda_cos-acc_cos):>14.4f}")
    print("-" * 60)
    
    # Verification performance analysis
    template_labels = np.arange(1, 109)
    
    # For original features
    print("Analyzing verification performance for original features using cosine similarity score...")
    templates = compute_templates(train_features, train_labels)
    original_results, genuine_scores, impostor_scores = plot_roc_and_metrics(
        test_features, test_labels, templates, template_labels
    )
    
    # For LDA features (using 100 dimensions)
    print("\nAnalyzing verification performance for LDA features using cosine similarity score...")
    lda = LinearDiscriminantAnalysis(n_components=107)
    lda.fit(train_features, train_labels)
    train_transformed = lda.transform(train_features)
    test_transformed = lda.transform(test_features)
    templates_lda = compute_templates(train_transformed, train_labels)
    lda_results, genuine_scores_lda, impostor_scores_lda = plot_roc_and_metrics(
        test_transformed, test_labels, templates_lda, template_labels
    )

if __name__ == "__main__":
    main()


