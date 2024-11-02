import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt

def compute_templates(train_features, train_labels):
    """
    Compute average template for each class
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
    """
    correct_l1 = 0
    correct_l2 = 0
    correct_cos = 0
    total = len(test_labels) // n_angles
    
    for i in range(0, len(test_features), n_angles):
        test_angles = test_features[i:i+n_angles]
        true_label = test_labels[i]
        
        # Store minimum distances for each angle
        min_l1_distances = []
        min_l2_distances = []
        min_cos_distances = []
        
        for test_feature in test_angles:
            l1_dist, l2_dist, cos_dist = compute_distances(test_feature, templates)
            min_l1_distances.append(np.min(l1_dist))
            min_l2_distances.append(np.min(l2_dist))
            min_cos_distances.append(np.min(cos_dist))
            
        # Find best angle for each metric
        best_l1_idx = np.argmin(min_l1_distances)
        best_l2_idx = np.argmin(min_l2_distances)
        best_cos_idx = np.argmin(min_cos_distances)
        
        # Get predictions for best angles
        l1_dist, l2_dist, cos_dist = compute_distances(test_angles[best_l1_idx], templates)
        pred_l1 = np.argmin(l1_dist) + 1
        pred_l2 = np.argmin(l2_dist) + 1
        pred_cos = np.argmin(cos_dist) + 1
        
        if pred_l1 == true_label:
            correct_l1 += 1
        if pred_l2 == true_label:
            correct_l2 += 1
        if pred_cos == true_label:
            correct_cos += 1
    
    return correct_l1/total, correct_l2/total, correct_cos/total


plt.show()


# %%
def calculate_verification_metrics(test_features, test_labels, templates, template_labels, n_angles=7):
    """
    Calculate verification metrics (genuine and impostor scores)
    Returns the best (highest) similarity score for each genuine/impostor pair
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
    """
    # False Match Rate: percentage of impostor scores above threshold
    fmr = np.mean(impostor_scores > threshold) * 100

    # False Non-Match Rate: percentage of genuine scores below or equal to threshold
    fnmr = np.mean(genuine_scores <= threshold) * 100

    return fmr, fnmr


def plot_roc_and_metrics(test_features, test_labels, templates, template_labels, n_bootstrap=1000):
    """
    Plot ROC curve and calculate metrics at specific operating points
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