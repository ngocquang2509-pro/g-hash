import torch
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def hamming_distance(code1, code2):
    """
    Compute Hamming distance between two binary codes
    
    Args:
        code1: Binary codes (N1, hash_bits) in {-1, +1}
        code2: Binary codes (N2, hash_bits) in {-1, +1}
    
    Returns:
        Hamming distance matrix (N1, N2)
    """
    if isinstance(code1, torch.Tensor):
        code1 = code1.cpu().numpy()
    if isinstance(code2, torch.Tensor):
        code2 = code2.cpu().numpy()
    
    # XOR and count differences
    # For binary codes in {-1, +1}, (code1 != code2).sum() gives hamming distance
    # Equivalent: (hash_bits - code1 @ code2.T) / 2
    hash_bits = code1.shape[1]
    hamming_dist = (hash_bits - code1 @ code2.T) / 2
    
    return hamming_dist.astype(int)


def compute_similarity_matrix(labels1, labels2):
    """
    Compute semantic similarity matrix between samples based on labels
    
    Args:
        labels1: Multi-hot labels (N1, num_classes)
        labels2: Multi-hot labels (N2, num_classes)
    
    Returns:
        Binary similarity matrix (N1, N2): 1 if share any label, 0 otherwise
    """
    if isinstance(labels1, torch.Tensor):
        labels1 = labels1.cpu().numpy()
    if isinstance(labels2, torch.Tensor):
        labels2 = labels2.cpu().numpy()
    
    # Two samples are similar if they share at least one label
    similarity = (labels1 @ labels2.T) > 0
    
    return similarity.astype(np.float32)


def mean_average_precision(query_codes, database_codes, query_labels, database_labels, top_k=None):
    """
    Compute Mean Average Precision (mAP)
    
    Args:
        query_codes: Binary codes for queries (N_query, hash_bits)
        database_codes: Binary codes for database (N_db, hash_bits)
        query_labels: Labels for queries (N_query, num_classes)
        database_labels: Labels for database (N_db, num_classes)
        top_k: Compute mAP@K (if None, use all retrieved samples)
    
    Returns:
        mAP score
    """
    num_query = query_codes.shape[0]
    
    # Compute Hamming distances
    hamming_dist = hamming_distance(query_codes, database_codes)
    
    # Compute ground truth similarity
    similarity = compute_similarity_matrix(query_labels, database_labels)
    
    # Compute AP for each query
    aps = []
    for i in range(num_query):
        # Get distances and similarities for this query
        distances = hamming_dist[i]
        sims = similarity[i]
        
        # Sort by distance (ascending)
        sorted_indices = np.argsort(distances)
        sorted_sims = sims[sorted_indices]
        
        # Truncate to top-K if specified
        if top_k:
            sorted_sims = sorted_sims[:top_k]
        
        # Compute average precision
        num_relevant = sorted_sims.sum()
        if num_relevant == 0:
            continue
        
        # Positions of relevant items
        relevant_positions = np.where(sorted_sims == 1)[0] + 1
        
        # Precision at each relevant position
        precisions = np.arange(1, len(relevant_positions) + 1) / relevant_positions
        
        # Average precision
        ap = precisions.sum() / num_relevant
        aps.append(ap)
    
    # Mean AP
    map_score = np.mean(aps)
    
    return map_score


def precision_at_k(query_codes, database_codes, query_labels, database_labels, k=100):
    """
    Compute Precision@K
    
    Args:
        query_codes: Binary codes for queries
        database_codes: Binary codes for database
        query_labels: Labels for queries
        database_labels: Labels for database
        k: Number of top retrieved samples
    
    Returns:
        Precision@K score
    """
    num_query = query_codes.shape[0]
    
    # Compute Hamming distances
    hamming_dist = hamming_distance(query_codes, database_codes)
    
    # Compute ground truth similarity
    similarity = compute_similarity_matrix(query_labels, database_labels)
    
    # Compute precision for each query
    precisions = []
    for i in range(num_query):
        distances = hamming_dist[i]
        sims = similarity[i]
        
        # Get top-K nearest neighbors
        top_k_indices = np.argsort(distances)[:k]
        top_k_sims = sims[top_k_indices]
        
        # Precision = (# relevant in top-K) / K
        precision = top_k_sims.sum() / k
        precisions.append(precision)
    
    return np.mean(precisions)


def recall_at_k(query_codes, database_codes, query_labels, database_labels, k=100):
    """
    Compute Recall@K
    """
    num_query = query_codes.shape[0]
    
    hamming_dist = hamming_distance(query_codes, database_codes)
    similarity = compute_similarity_matrix(query_labels, database_labels)
    
    recalls = []
    for i in range(num_query):
        distances = hamming_dist[i]
        sims = similarity[i]
        
        num_relevant_total = sims.sum()
        if num_relevant_total == 0:
            continue
        
        # Get top-K
        top_k_indices = np.argsort(distances)[:k]
        top_k_sims = sims[top_k_indices]
        
        # Recall = (# relevant in top-K) / (# total relevant)
        recall = top_k_sims.sum() / num_relevant_total
        recalls.append(recall)
    
    return np.mean(recalls)


def precision_recall_curve_at_hamming_radius(query_codes, database_codes, 
                                              query_labels, database_labels, max_radius=4):
    """
    Compute precision-recall at different Hamming radii
    
    Returns:
        Dictionary with precision and recall at each radius
    """
    hamming_dist = hamming_distance(query_codes, database_codes)
    similarity = compute_similarity_matrix(query_labels, database_labels)
    
    results = {}
    for radius in range(max_radius + 1):
        precisions = []
        recalls = []
        
        for i in range(len(query_codes)):
            # Retrieve samples within Hamming radius
            retrieved_mask = hamming_dist[i] <= radius
            num_retrieved = retrieved_mask.sum()
            
            if num_retrieved == 0:
                continue
            
            # Compute precision and recall
            num_relevant = similarity[i].sum()
            if num_relevant == 0:
                continue
            
            num_relevant_retrieved = (similarity[i] * retrieved_mask).sum()
            
            precision = num_relevant_retrieved / num_retrieved
            recall = num_relevant_retrieved / num_relevant
            
            precisions.append(precision)
            recalls.append(recall)
        
        results[radius] = {
            'precision': np.mean(precisions) if precisions else 0,
            'recall': np.mean(recalls) if recalls else 0
        }
    
    return results


def compute_retrieval_metrics(query_codes, database_codes, query_labels, database_labels, top_k_list=[100, 500, 1000]):
    """
    Compute comprehensive retrieval metrics
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # mAP at different K values
    for k in top_k_list:
        if k <= len(database_codes):
            metrics[f'mAP@{k}'] = mean_average_precision(
                query_codes, database_codes, query_labels, database_labels, top_k=k
            )
    
    # Overall mAP
    metrics['mAP'] = mean_average_precision(
        query_codes, database_codes, query_labels, database_labels
    )
    
    # Precision and Recall at K
    for k in top_k_list:
        if k <= len(database_codes):
            metrics[f'P@{k}'] = precision_at_k(
                query_codes, database_codes, query_labels, database_labels, k=k
            )
            metrics[f'R@{k}'] = recall_at_k(
                query_codes, database_codes, query_labels, database_labels, k=k
            )
    
    # Precision-Recall at Hamming radius 2 (common in papers)
    pr_at_radius = precision_recall_curve_at_hamming_radius(
        query_codes, database_codes, query_labels, database_labels, max_radius=4
    )
    metrics['PR_at_radius'] = pr_at_radius
    
    return metrics
