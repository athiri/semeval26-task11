"""Models"""
from .baseline import BaselineModel

def get_model(model_type: str, **kwargs):
    if model_type in ['logistic_regression', 'random_forest', 'gradient_boosting']:
        return BaselineModel(model_type=model_type)
    raise ValueError(f'Unknown model: {model_type}')
