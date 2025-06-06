from .utils import Thresholds

import numpy as np
import scipy
import sklearn

def get_optimal_threshold(alpha_id, alpha_ood, metric: Thresholds = Thresholds.DIFF_ENTROPY):
    if metric == Thresholds.DIFF_ENTROPY:
        id_scores = diff_entropy(alpha_id)
        ood_scores = diff_entropy(alpha_ood)
    elif metric == Thresholds.PRED_ENTROPY:
        id_scores = pred_entropy(alpha_id)
        ood_scores = pred_entropy(alpha_ood) 
    elif metric == Thresholds.TOTAL_ALPHA:
        id_scores = total_alpha(alpha_id)
        ood_scores = total_alpha(alpha_ood)
    elif metric == Thresholds.MUTUAL_INFO:
        id_scores = mutual_info(alpha_id)
        ood_scores = mutual_info(alpha_ood)

    corrects = np.concatenate([np.ones(len(alpha_id)), np.zeros(len(alpha_ood))], axis=0)
    scores = np.concatenate([id_scores, ood_scores], axis=0)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(corrects, scores)
    auc_score = sklearn.metrics.auc(fpr, tpr)
    
    index = tpr - fpr
    optimal_idx = np.argmax(index)
    optimal_threshold = thresholds[optimal_idx]
    
    return auc_score, fpr, tpr, thresholds, optimal_threshold

def mutual_info(alpha):
    eps = 1e-6
    alpha = np.asarray(alpha, dtype=np.float64) + eps
    alpha0 = np.sum(alpha, axis=1, keepdims=True)
    probs = alpha / alpha0 
    total_uncertainty = -np.sum(probs * np.log(probs + eps), axis=1)
    digamma_alpha = scipy.special.digamma(alpha + 1.0)
    digamma_alpha0 = scipy.special.digamma(alpha0 + 1.0)
    exp_data_uncertainty = -np.sum(probs * (digamma_alpha - digamma_alpha0), axis=1)
    mi = total_uncertainty - exp_data_uncertainty
    return (-mi).squeeze()   

def total_alpha(alpha):
    return np.sum(alpha, axis=1)

def pred_entropy(probabilities):
    eps = 1e-6
    entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
    normalized_entropy = entropy / np.log(probabilities.shape[1])  # Normalize by log(C)
    return -normalized_entropy

def diff_entropy(alpha):
    eps = 1e-6
    alpha = alpha + eps
    alpha0 = np.sum(alpha, axis=1)
    log_term = np.sum(scipy.special.gammaln(alpha), axis=1) - scipy.special.gammaln(alpha0)
    digamma_term = np.sum((alpha - 1.0) * (scipy.special.digamma(alpha) - scipy.special.digamma(alpha0[:, None])), axis=1)
    differential_entropy = log_term - digamma_term
    return -differential_entropy