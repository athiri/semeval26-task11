"""Data loader for syllogism datasets"""
import json
from pathlib import Path
import pandas as pd

class SyllogismDataLoader:
    """Load syllogism data for different subtasks"""
    
    def __init__(self, subtask: int):
        self.subtask = subtask
        self.data_dir = Path("data")
    
    def load_split(self, split: str) -> pd.DataFrame:
        """Load train/val/test split"""
        file_path = self.data_dir / f"{split}_subtask{self.subtask}.json"
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            print(f"   Run: python3 src/generate_data.py --subtask {self.subtask}")
            return pd.DataFrame()
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return pd.DataFrame(data)
    
    def get_statistics(self):
        """Get dataset statistics"""
        train_df = self.load_split('train')
        
        if train_df.empty:
            return {}
        
        stats = {
            'num_samples': len(train_df),
            'num_valid': int(train_df['validity'].sum()),
            'num_invalid': int((~train_df['validity']).sum()),
        }
        
        if 'plausibility' in train_df.columns:
            stats['num_plausible'] = int(train_df['plausibility'].sum())
            stats['num_implausible'] = int((~train_df['plausibility']).sum())
        
        return stats
