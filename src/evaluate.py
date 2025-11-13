"""Evaluation metrics"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

def calculate_content_effect(y_true, y_pred, plausibility):
    """Calculate content effect (lower is better)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    plausibility = np.array(plausibility)
    
    plausible_mask = plausibility == True
    implausible_mask = plausibility == False
    
    if plausible_mask.sum() == 0 or implausible_mask.sum() == 0:
        return 0.0
    
    acc_plausible = accuracy_score(y_true[plausible_mask], y_pred[plausible_mask])
    acc_implausible = accuracy_score(y_true[implausible_mask], y_pred[implausible_mask])
    
    return abs(acc_plausible - acc_implausible)

def evaluate(y_true, y_pred, plausibility=None, detailed=False):
    """Evaluate predictions"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='binary'),
    }
    
    if plausibility is not None:
        metrics['content_effect'] = calculate_content_effect(y_true, y_pred, plausibility)
        if metrics['content_effect'] > 0:
            metrics['ranking_score'] = metrics['accuracy'] / metrics['content_effect']
        else:
            metrics['ranking_score'] = metrics['accuracy'] * 100
    
    if detailed:
        print("\n" + classification_report(y_true, y_pred, target_names=['Invalid', 'Valid']))
    
    return metrics
