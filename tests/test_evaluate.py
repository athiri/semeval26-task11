#!/usr/bin/env python3
"""Test evaluation metrics"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluate import evaluate, calculate_content_effect
import numpy as np

def test_evaluation():
    """Test evaluation functions"""
    print("Testing evaluation metrics...")
    
    # Test basic accuracy
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])
    
    metrics = evaluate(y_true, y_pred)
    
    assert 'accuracy' in metrics, "Should have accuracy"
    assert 'f1' in metrics, "Should have f1"
    assert metrics['accuracy'] == 1.0, "Perfect predictions should have accuracy 1.0"
    
    print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ✓ F1 Score: {metrics['f1']:.4f}")
    
    # Test content effect
    plausibility = np.array([True, True, False, False, True])
    metrics_with_ce = evaluate(y_true, y_pred, plausibility)
    
    assert 'content_effect' in metrics_with_ce, "Should have content_effect"
    assert 'ranking_score' in metrics_with_ce, "Should have ranking_score"
    
    print(f"  ✓ Content Effect: {metrics_with_ce['content_effect']:.4f}")
    print(f"  ✓ Ranking Score: {metrics_with_ce['ranking_score']:.2f}")
    
    # Test content effect calculation
    ce = calculate_content_effect(y_true, y_pred, plausibility)
    assert isinstance(ce, float), "Content effect should be float"
    assert ce >= 0, "Content effect should be non-negative"
    
    print("  ✅ Evaluation tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_evaluation()
        sys.exit(0)
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
