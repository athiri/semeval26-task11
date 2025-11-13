#!/usr/bin/env python3
"""Test feature extraction"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features import extract_all_features, extract_features_from_dataframe
import pandas as pd

def test_feature_extraction():
    """Test feature extraction functions"""
    print("Testing feature extraction...")
    
    # Test basic feature extraction
    test_text = "All dogs are mammals. All mammals breathe. Therefore, all dogs breathe."
    features = extract_all_features(test_text)
    
    assert isinstance(features, dict), "Should return dictionary"
    assert len(features) > 0, "Should extract some features"
    
    # Check for expected feature types
    assert 'text_length' in features, "Should have text_length"
    assert 'num_words' in features, "Should have num_words"
    assert 'count_all' in features, "Should have logical features"
    
    print(f"  ✓ Extracted {len(features)} features")
    print(f"  ✓ Sample features: {list(features.keys())[:5]}")
    
    # Test DataFrame extraction
    df = pd.DataFrame({
        'syllogism': [test_text, test_text]
    })
    feature_df = extract_features_from_dataframe(df)
    
    assert isinstance(feature_df, pd.DataFrame), "Should return DataFrame"
    assert len(feature_df) == 2, "Should have 2 rows"
    assert len(feature_df.columns) == len(features), "Should have same number of features"
    
    print(f"  ✓ DataFrame extraction works")
    print("  ✅ Feature extraction tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_feature_extraction()
        sys.exit(0)
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
