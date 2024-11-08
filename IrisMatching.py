import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from PerformanceEvaluation import compute_templates, evaluate_recognition

def run_lda_analysis(train_features, train_labels, test_features, test_labels, dimensions):

    """
    Perform LDA analysis with different dimensions

    Logic:
    1. For each target dimension:
        - Creates LDA model with specified components
        - Fits model to training data
        - Transforms both training and test features
        - Computes templates from transformed features
        - Evaluates recognition performance

    Parameters:
    - `train_features`: Training set feature matrix
    - `train_labels`: Training set class labels
    - `test_features`: Test set feature matrix
    - `test_labels`: Test set class labels
    - `dimensions`: List of target dimensions to evaluate
    - Returns: Array of [dimension, acc_l1, acc_l2, acc_cos] for each dimension

    """
    results = []
    for n_components in dimensions:
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(train_features, train_labels)
        
        # Transform features
        train_transformed = lda.transform(train_features)
        test_transformed = lda.transform(test_features)
        
        # Compute templates and evaluate
        templates = compute_templates(train_transformed, train_labels)
        acc_l1, acc_l2, acc_cos = evaluate_recognition(test_transformed, test_labels, templates)
        results.append([n_components, acc_l1, acc_l2, acc_cos])
        
    return np.array(results)
 

