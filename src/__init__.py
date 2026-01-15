"""
Student Mental Health - Depression Prediction
Module principal pour le preprocessing et la modélisation
"""

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    TRAIN_PATH,
    TEST_PATH,
    SUBMISSION_PATH,
    TARGET_FEATURE,
    RANDOM_STATE,
    OPTIMAL_THRESHOLD
)
from .preprocessing import encodage, imputation, preprocessing, test_imputation, test_preprocessing, set_trainset
from .modeling import create_models, get_best_model
from .evaluation import evaluation, plot_precision_recall_curve, model_final

__all__ = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'TRAIN_PATH',
    'TEST_PATH',
    'SUBMISSION_PATH',
    'TARGET_FEATURE',
    'RANDOM_STATE',
    'OPTIMAL_THRESHOLD',
    'encodage',
    'imputation',
    'preprocessing',
    'test_imputation',
    'test_preprocessing',
    'set_trainset',
    'create_models',
    'get_best_model',
    'evaluation',
    'plot_precision_recall_curve',
    'model_final'
]
