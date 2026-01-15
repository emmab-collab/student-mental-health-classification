"""
Student Mental Health - Depression Prediction
Module principal pour le preprocessing, la modélisation et l'EDA
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
from .eda import (
    get_data_shape, plot_missing_values, get_missing_values_summary,
    plot_target_distribution, create_target_subsets,
    plot_feature_by_target, plot_correlation_matrix, perform_ttest
)

__all__ = [
    # Config
    'PROJECT_ROOT',
    'DATA_DIR',
    'TRAIN_PATH',
    'TEST_PATH',
    'SUBMISSION_PATH',
    'TARGET_FEATURE',
    'RANDOM_STATE',
    'OPTIMAL_THRESHOLD',
    # Preprocessing
    'encodage',
    'imputation',
    'preprocessing',
    'test_imputation',
    'test_preprocessing',
    'set_trainset',
    # Modeling
    'create_models',
    'get_best_model',
    # Evaluation
    'evaluation',
    'plot_precision_recall_curve',
    'model_final',
    # EDA
    'get_data_shape',
    'plot_missing_values',
    'get_missing_values_summary',
    'plot_target_distribution',
    'create_target_subsets',
    'plot_feature_by_target',
    'plot_correlation_matrix',
    'perform_ttest'
]
