"""
Configuration du projet - Chemins et constantes
"""

import os

# Chemin racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemins des données
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
SUBMISSION_PATH = os.path.join(DATA_DIR, 'submission.csv')

# Variable cible
TARGET_FEATURE = 'Depression'

# Paramètres du modèle
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 4

# Seuil optimal pour les prédictions
OPTIMAL_THRESHOLD = -0.01
