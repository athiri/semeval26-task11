#!/usr/bin/env python3
"""Test pipeline integration"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import SyllogismDataLoader
from features import extract_features_from_dataframe
from models import get_model
from evaluate import evaluate
import pandas as pd

def test_pipeline_integration():
    """Test full pipeline integration"""
    print("Testing pipeline integration...")
    
    # Test model creation
    model = get_model('random_forest')
    assert model is not None, "Should create model"
    print("  ✓ Model creation works")
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'syllogism': [
            "All A are B. All B are C. Therefore, all A are C.",
            "Some A are B. All B are C. Therefore, some A are C."
        ],
        'validity': [True, True]
    })
    
    # Extract features
    X = extract_features_from_dataframe(sample_data)
    y = sample_data['validity'].values
    
    assert len(X) == 2, "Should have 2 feature vectors"
    assert len(y) == 2, "Should have 2 labels"
    print(f"  ✓ Feature extraction: {X.shape}")
    
    # Train model
    model.fit(X, y)
    assert model.is_trained, "Model should be trained"
    print("  ✓ Model training works")
    
    # Make predictions
    y_pred = model.predict(X)
    assert len(y_pred) == 2, "Should have 2 predictions"
    print("  ✓ Model prediction works")
    
    # Evaluate
    metrics = evaluate(y, y_pred)
    assert 'accuracy' in metrics, "Should have metrics"
    print(f"  ✓ Evaluation: accuracy={metrics['accuracy']:.4f}")
    
    print("  ✅ Pipeline integration tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_pipeline_integration()
        sys.exit(0)
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
