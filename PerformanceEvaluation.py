import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt

def compute_templates(train_features, train_labels):
    """
    Compute average template for each class

    Logic:
    1. Iterates through all classes (1-108)
    2. Extracts features belonging to each class
    3. Computes mean feature vector as template

    Parameters:
    - `train_features`: Training feature matrix
    - `train_labels`: Training class labels
    - Returns: Array of class templates

    """
    templates = []
    for class_id in range(1, 109):
        class_features = train_features[train_labels == class_id]
        if len(class_features) > 0:
            template = np.mean(class_features, axis=0)
            templates.append(template)
    return np.array(templates)

def compute_distances(test_feature, templates):
    """
    Compute l1, l2, and cosine distances between a test feature and all templates

    Logic:
    1. Computes L1 distance (Manhattan)
    2. Computes L2 distance (Euclidean)
    3. Computes Cosine distance (1 - cosine similarity)

    Parameters:
    - `test_feature`: Single test feature vector
    - `templates`: Array of class templates
    - Returns: (l1_distances, l2_distances, cos_distances)

    """
    # L1 distance
    l1_distances = np.sum(np.abs(templates - test_feature), axis=1)
    
    # L2 distance
    l2_distances = np.sqrt(np.sum((templates - test_feature)**2, axis=1))
    
    # Cosine similarity (convert to distance)
    cos_similarities = cosine_similarity(test_feature.reshape(1, -1), templates)
    cos_distances = 1 - cos_similarities[0]
    
    return l1_distances, l2_distances, cos_distances

def evaluate_recognition(test_features, test_labels, templates, n_angles=7):
    """
    Evaluate recognition accuracy using different distance metrics

    Logic:
    1. For each test sample:
        - Processes features from all rotation angles
        - Finds best matching angle for each distance metric
        - Compares predictions with true labels
    2. Computes accuracy for each metric

    Parameters:
    - test_features: Test feature matrix
    - test_labels: Test class labels
    - templates: Class templates
    - n_angles: Number of rotation angles (default=7)
    - Returns: (acc_l1, acc_l2, acc_cos)

    """
    correct_l1 = 0
    correct_l2 = 0
    correct_cos = 0
    total = len(test_labels) // n_angles

    for i in range(0, len(test_features), n_angles):
        test_angles = test_features[i:i+n_angles]
        true_label = test_labels[i]

        # Store distances for each angle
        l1_distances = []
        l2_distances = []
        cos_distances = []

        for test_feature in test_angles:
            l1_dist, l2_dist, cos_dist = compute_distances(test_feature, templates)
            l1_distances.append(l1_dist)
            l2_distances.append(l2_dist)
            cos_distances.append(cos_dist)

        # Find best angle for each metric
        min_l1_distances = [np.min(dists) for dists in l1_distances]
        min_l2_distances = [np.min(dists) for dists in l2_distances]
        min_cos_distances = [np.min(dists) for dists in cos_distances]

        best_l1_idx = np.argmin(min_l1_distances)
        best_l2_idx = np.argmin(min_l2_distances)
        best_cos_idx = np.argmin(min_cos_distances)

        # Get predictions for best angles
        pred_l1 = np.argmin(l1_distances[best_l1_idx]) + 1
        pred_l2 = np.argmin(l2_distances[best_l2_idx]) + 1
        pred_cos = np.argmin(cos_distances[best_cos_idx]) + 1

        if pred_l1 == true_label:
            correct_l1 += 1
        if pred_l2 == true_label:
            correct_l2 += 1
        if pred_cos == true_label:
            correct_cos += 1

    return correct_l1/total, correct_l2/total, correct_cos/total



def calculate_verification_metrics(test_features, test_labels, templates, template_labels, n_angles=7):
    """
    Calculate verification metrics (genuine and impostor scores)
    Returns the best (highest) similarity score for each genuine/impostor pair

    Logic:
    1. For each test sample:
        - Computes similarity scores across all angles
        - Records best genuine match score
        - Records all impostor match scores
    2. Collects scores for ROC analysis

    Parameters:
    - `test_features`, `test_labels`: Test data
    - `templates`, `template_labels`: Template data
    - `n_angles`: Number of rotation angles
    - Returns: (genuine_scores, impostor_scores)
    """
    genuine_scores = []
    impostor_scores = []

    for i in range(0, len(test_features), n_angles):
        test_angles = test_features[i:i + n_angles]
        true_label = test_labels[i]

        # For each test image, find its best matching score
        best_genuine_score = -1
        best_impostor_scores = []

        # Try all angles and keep the best scores
        for test_feature in test_angles:
            similarities = cosine_similarity(test_feature.reshape(1, -1), templates)[0]

            # For each template
            for template_idx, similarity in enumerate(similarities):
                if template_labels[template_idx] == true_label:
                    best_genuine_score = max(best_genuine_score, similarity)
                else:
                    best_impostor_scores.append(similarity)

        # After checking all angles, store the best scores
        genuine_scores.append(best_genuine_score)
        # For impostor scores, we'll take the highest score for each comparison
        # (worst case scenario - most likely to cause false match)
        impostor_scores.extend(best_impostor_scores)

    return np.array(genuine_scores), np.array(impostor_scores)


def calculate_fmr_fnmr(genuine_scores, impostor_scores, threshold):
    """
    Calculate FMR and FNMR at a given threshold
    For cosine similarity, we accept matches when score is ABOVE threshold

    Logic:
    1. Calculates False Match Rate (FMR)
    - Percentage of impostor scores above threshold
    2. Calculates False Non-Match Rate (FNMR)
    - Percentage of genuine scores below threshold

    Parameters:
    - `genuine_scores`: Scores from genuine matches
    - `impostor_scores`: Scores from impostor matches
    - `threshold`: Decision threshold
    - Returns: (fmr, fnmr) as percentages

    """
    # False Match Rate: percentage of impostor scores above threshold
    fmr = np.mean(impostor_scores > threshold) * 100

    # False Non-Match Rate: percentage of genuine scores below or equal to threshold
    fnmr = np.mean(genuine_scores <= threshold) * 100

    return fmr, fnmr


def plot_roc_and_metrics(test_features, test_labels, templates, template_labels):
    """
    Plot ROC curve and calculate metrics at specific operating points
    Logic:
    1. Calculates verification scores
    2. Computes ROC curve points using multiple thresholds
    3. Finds operating points at specific FMR values
    4. Generates ROC plot and performance table

    Parameters:
    - `test_features`, `test_labels`: Test data
    - `templates`, `template_labels`: Template data
    - Returns: (results, genuine_scores, impostor_scores)

    """
    # Calculate genuine and impostor scores
    genuine_scores, impostor_scores = calculate_verification_metrics(
        test_features, test_labels, templates, template_labels
    )

    # Calculate ROC curve points
    # Use thresholds based on the range of actual scores
    min_score = min(np.min(genuine_scores), np.min(impostor_scores))
    max_score = max(np.max(genuine_scores), np.max(impostor_scores))
    thresholds = np.linspace(min_score, max_score, 1000)

    fmr_list = []
    fnmr_list = []

    for threshold in thresholds:
        fmr, fnmr = calculate_fmr_fnmr(genuine_scores, impostor_scores, threshold)
        fmr_list.append(fmr)
        fnmr_list.append(fnmr)

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    # plt.plot(fmr_list, 100 - np.array(fnmr_list), label='ROC Curve')
    plt.plot(fmr_list, fnmr_list, label='ROC Curve')
    plt.xlabel('False Match Rate (%)')
    # plt.ylabel('True Match Rate (%)')
    plt.ylabel('False Non-Match Rate (%)')
    plt.title('ROC Curve for Iris Verification')
    plt.grid(True)
    plt.legend()

    # Find operating points
    target_fmrs = [0.01, 0.001, 0.0001]  # Target FMR values in percentage
    results = []

    for target_fmr in target_fmrs:
        # Find threshold that gives closest FMR to target
        fmr_array = np.array(fmr_list)
        idx = np.argmin(np.abs(fmr_array - target_fmr))
        threshold = thresholds[idx]

        # Calculate actual FMR and FNMR at this threshold
        fmr, fnmr = calculate_fmr_fnmr(genuine_scores, impostor_scores, threshold)

        results.append({
            'threshold': threshold,
            'fmr': fmr,
            'fnmr': fnmr
        })

    # Create results table
    print("\nVerification Results at Different Thresholds:")
    print("Threshold    False match rate (%)    False non-match rate (%)")
    print("-" * 60)

    for res in results:
        print(f"{res['threshold']:.3f}        {res['fmr']:.3f}                   {res['fnmr']:.2f}")

    plt.show()
    return results, genuine_scores, impostor_scores