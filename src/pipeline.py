#!/usr/bin/env python3
"""
Main pipeline for SemEval 2026 Task 11
Simple baseline implementation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import SyllogismDataLoader
from features import extract_features_from_dataframe
from models import get_model
from evaluate import evaluate
import numpy as np

# Set seed
np.random.seed(42)

def train_baseline(subtask=1, model_type='random_forest'):
    """Train baseline model"""
    print(f"\n{'='*60}")
    print(f"Training Baseline - Subtask {subtask}")
    print(f"{'='*60}\n")
    
    # Load data
    print("📥 Loading data...")
    loader = SyllogismDataLoader(subtask)
    train_df = loader.load_split('train')
    val_df = loader.load_split('val')
    
    if train_df.empty or val_df.empty:
        print("❌ No data found. Run: python3 src/generate_data.py --subtask 1")
        return
    
    print(f"✓ Loaded {len(train_df)} training samples")
    print(f"✓ Loaded {len(val_df)} validation samples")
    
    # Extract features
    print("\n🔧 Extracting features...")
    X_train = extract_features_from_dataframe(train_df)
    X_val = extract_features_from_dataframe(val_df)
    y_train = train_df['validity'].values
    y_val = val_df['validity'].values
    
    print(f"✓ Extracted {X_train.shape[1]} features")
    
    # Train model
    print(f"\n🤖 Training {model_type}...")
    model = get_model(model_type)
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n📊 Evaluating...")
    y_pred = model.predict(X_val)
    
    plausibility = val_df['plausibility'].values if 'plausibility' in val_df.columns else None
    metrics = evaluate(y_val, y_pred, plausibility, detailed=True)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"F1 Score:       {metrics['f1']:.4f}")
    
    if 'content_effect' in metrics:
        print(f"Content Effect: {metrics['content_effect']:.4f} (lower is better)")
        print(f"Ranking Score:  {metrics['ranking_score']:.2f} (higher is better)")
    
    print(f"{'='*60}\n")
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"subtask{subtask}_{model_type}.pkl"
    model.save(model_path)
    print(f"✅ Model saved to: {model_path}\n")

def show_info(subtask=1):
    """Show dataset info"""
    print(f"\n{'='*60}")
    print(f"Subtask {subtask} Information")
    print(f"{'='*60}\n")
    
    loader = SyllogismDataLoader(subtask)
    stats = loader.get_statistics()
    
    if not stats:
        print("❌ No data found. Run: python3 src/generate_data.py --subtask 1")
        return
    
    print(f"Total samples:      {stats['num_samples']:,}")
    print(f"Valid syllogisms:   {stats['num_valid']:,}")
    print(f"Invalid syllogisms: {stats['num_invalid']:,}")
    
    if 'num_plausible' in stats:
        print(f"Plausible:          {stats['num_plausible']:,}")
        print(f"Implausible:        {stats['num_implausible']:,}")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SemEval 2026 Task 11 Pipeline")
    parser.add_argument('command', choices=['train', 'info'], help='Command to run')
    parser.add_argument('--subtask', type=int, default=1, help='Subtask number (1-4)')
    parser.add_argument('--model-type', default='random_forest',
                       choices=['logistic_regression', 'random_forest', 'gradient_boosting'],
                       help='Model type')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_baseline(args.subtask, args.model_type)
    elif args.command == 'info':
        show_info(args.subtask)
