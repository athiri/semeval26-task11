"""Baseline models"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle

class BaselineModel:
    """Baseline sklearn models"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.is_trained = False
        
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    def fit(self, X, y):
        """Train the model"""
        print(f"\nTraining {self.model_type}...")
        print(f"  Training samples: {len(X):,}")
        print(f"  Features: {X.shape[1]}")
        
        self.model.fit(X, y)
        self.is_trained = True
        
        train_pred = self.model.predict(X)
        train_acc = accuracy_score(y, train_pred)
        print(f"  Training Accuracy: {train_acc:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probabilities"""
        return self.model.predict_proba(X)
    
    def save(self, path):
        """Save model"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        """Load model"""
        with open(path, 'rb') as f:
            return pickle.load(f)
