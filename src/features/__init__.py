"""Feature extraction"""
from .basic import extract_basic_features
from .logical import extract_logical_features
import pandas as pd

def extract_all_features(text: str) -> dict:
    features = {}
    features.update(extract_basic_features(text))
    features.update(extract_logical_features(text))
    return features

def extract_features_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([extract_all_features(t) for t in df['syllogism']])
