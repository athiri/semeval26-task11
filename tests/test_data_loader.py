#!/usr/bin/env python3
"""Test data loader functionality"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import SyllogismDataLoader
import pandas as pd

def test_data_loader():
    """Test that data loader can load existing data"""
    print("Testing SyllogismDataLoader...")
    
    loader = SyllogismDataLoader(subtask=1)
    
    # Test loading train split
    train_df = loader.load_split('train')
    assert isinstance(train_df, pd.DataFrame), "Should return DataFrame"
    
    if not train_df.empty:
        print(f"  ✓ Loaded {len(train_df)} training samples")
        assert 'syllogism' in train_df.columns, "Should have 'syllogism' column"
        assert 'validity' in train_df.columns, "Should have 'validity' column"
    else:
        print("  ⚠️  No training data found (run generate_data.py first)")
    
    # Test statistics
    stats = loader.get_statistics()
    if stats:
        print(f"  ✓ Statistics: {stats['num_samples']} samples")
    
    print("  ✅ Data loader tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_data_loader()
        sys.exit(0)
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        sys.exit(1)
