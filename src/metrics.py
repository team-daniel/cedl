import numpy as np
import scipy
import sklearn

def get_optimal_threshold(alpha_id, alpha_ood, metric="diff_entropy"):
    if metric == "diff_entropy":
        id_scores = diff_entropy(alpha_id)
        ood_scores = diff_entropy(alpha_ood)
    elif metric == "pred_entropy":
        id_scores = pred_entropy(alpha_id)
        ood_scores = pred_entropy(alpha_ood)     

    corrects = np.concatenate([np.ones(len(alpha_id)), np.zeros(len(alpha_ood))], axis=0)
    scores = np.concatenate([id_scores, ood_scores], axis=0)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(corrects, scores)
    auc_score = sklearn.metrics.auc(fpr, tpr)
    
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    
    return auc_score, fpr, tpr, thresholds, optimal_threshold

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