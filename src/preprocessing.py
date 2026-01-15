"""
Module de preprocessing pour la prédiction de dépression
Reproduit exactement les étapes du notebook pre-processing-mental-health.ipynb
"""

import numpy as np
import pandas as pd

# Variable globale pour la target
TARGET_FEATURE = 'Depression'

# Variables globales pour stocker le trainset
_trainset = None
_trainset_encoded = None


def set_trainset(trainset):
    """Définit le trainset global pour l'encodage basé sur les moyennes"""
    global _trainset, _trainset_encoded
    _trainset = trainset.copy()
    # Encoder le trainset pour pouvoir calculer les médianes plus tard
    _trainset_encoded = _encode_trainset_for_medians(trainset.copy())


def get_trainset():
    """Retourne le trainset global (non encodé)"""
    return _trainset


def get_trainset_encoded():
    """Retourne le trainset encodé pour les calculs de médiane"""
    return _trainset_encoded


def _encode_trainset_for_medians(trainset):
    """
    Encode le trainset pour pouvoir calculer les médianes.
    Applique les mêmes transformations que encodage() mais sur le trainset lui-même.
    """
    df = trainset.copy()

    sommeil = {
        "More than 8 hours": 9,
        'Less than 5 hours': 4,
        '5-6 hours': 5.5,
        '7-8 hours': 7.5,
        '1-2 hours': 1.5,
        '6-8 hours': 7,
        '4-6 hours': 5,
        '6-7 hours': 6.5,
        '10-11 hours': 10.5,
        '8-9 hours': 8.5,
        '9-11 hours': 10,
        '2-3 hours': 2.5,
        '3-4 hours': 3.5,
        'Moderate': 6,
        '4-5 hours': 4.5,
        '9-6 hours': 7.5,
        '1-3 hours': 2,
        '1-6 hours': 4,
        'Healthy': 1,
        'Less than Healthy': 2,
        'Less Healthy': 2,
        'Unhealthy': 4,
        'No Healthy': 4
    }

    diet = {
        'More Healty': 0,
        'Healthy': 1,
        'Less than Healthy': 2,
        'Less Healthy': 2,
        'Moderate': 3,
        'Unhealthy': 4,
        'No Healthy': 4
    }

    degree_map = {
        "BCom": "B.Com", "B.Com": "B.Com", "B.Comm": "B.Com",
        "B.Tech": "B.Tech", "BTech": "B.Tech", "B.T": "B.Tech",
        "BSc": "B.Sc", "B.Sc": "B.Sc", "Bachelor of Science": "B.Sc",
        "BArch": "B.Arch", "B.Arch": "B.Arch",
        "BA": "B.A", "B.A": "B.A",
        "BBA": "BBA", "BB": "BBA",
        "BCA": "BCA",
        "BE": "BE",
        "BEd": "B.Ed", "B.Ed": "B.Ed",
        "BPharm": "B.Pharm", "B.Pharm": "B.Pharm",
        "BHM": "BHM",
        "LLB": "LLB", "LL B": "LLB", "LL BA": "LLB", "LL.Com": "LLB", "LLCom": "LLB",
        "MCom": "M.Com", "M.Com": "M.Com",
        "M.Tech": "M.Tech", "MTech": "M.Tech", "M.T": "M.Tech",
        "MSc": "M.Sc", "M.Sc": "M.Sc", "Master of Science": "M.Sc",
        "MBA": "MBA",
        "MCA": "MCA",
        "MD": "MD",
        "ME": "ME",
        "MEd": "M.Ed", "M.Ed": "M.Ed",
        "MArch": "M.Arch", "M.Arch": "M.Arch",
        "MPharm": "M.Pharm", "M.Pharm": "M.Pharm",
        "MA": "MA", "M.A": "MA",
        "MPA": "MPA",
        "LLM": "LLM",
        "PhD": "PhD",
        "MBBS": "MBBS",
        "CA": "CA",
        "Class 12": "Class 12", "12th": "Class 12",
        "Class 11": "Class 11", "11th": "Class 11"
    }

    # Target encoding basé sur le trainset original
    mean_per_city = trainset.groupby('City')[TARGET_FEATURE].mean()
    mean_per_name = trainset.groupby('Name')[TARGET_FEATURE].mean()
    mean_per_profession = trainset.groupby('Profession')[TARGET_FEATURE].mean()

    # Pour Degree, on doit d'abord normaliser puis calculer la moyenne
    trainset_temp = trainset.copy()
    trainset_temp['Degree'] = trainset_temp['Degree'].map(degree_map)
    mean_per_degree = trainset_temp.groupby('Degree')[TARGET_FEATURE].mean()

    code = {
        'Female': 1,
        'Male': 0,
        'Working Professional': 1,
        'Student': 0,
        'Yes': 1,
        'No': 0
    }

    # Application des encodages
    df['Dietary Habits'] = df['Dietary Habits'].map(diet)
    df['Sleep Duration'] = df['Sleep Duration'].map(sommeil)
    df['Degree'] = df['Degree'].map(degree_map)

    # Target encoding
    df['City'] = df['City'].map(mean_per_city)
    df['Name'] = df['Name'].map(mean_per_name)
    df['Profession'] = df['Profession'].map(mean_per_profession)
    df['Degree'] = df['Degree'].map(mean_per_degree)

    # Encodage binaire
    for col in ['Gender', 'Working Professional or Student',
                'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']:
        df[col] = df[col].map(code)

    # Combinaison des colonnes Pressure et Satisfaction
    df['Pressure'] = np.where(
        df['Working Professional or Student'] == 0,
        df['Academic Pressure'],
        df['Work Pressure']
    )
    df['Satisfaction'] = np.where(
        df['Working Professional or Student'] == 0,
        df['Study Satisfaction'],
        df['Job Satisfaction']
    )

    return df


def encodage(df):
    """
    Encode les variables catégorielles en variables numériques.
    Reproduit exactement la fonction encodage() du notebook original.
    """
    trainset = get_trainset()
    if trainset is None:
        raise ValueError("Le trainset n'a pas été défini. Utilisez set_trainset() d'abord.")

    sommeil = {
        "More than 8 hours": 9,
        'Less than 5 hours': 4,
        '5-6 hours': 5.5,
        '7-8 hours': 7.5,
        '1-2 hours': 1.5,
        '6-8 hours': 7,
        '4-6 hours': 5,
        '6-7 hours': 6.5,
        '10-11 hours': 10.5,
        '8-9 hours': 8.5,
        '9-11 hours': 10,
        '2-3 hours': 2.5,
        '3-4 hours': 3.5,
        'Moderate': 6,
        '4-5 hours': 4.5,
        '9-6 hours': 7.5,
        '1-3 hours': 2,
        '1-6 hours': 4,
        '8 hours': 8,
        '10-6 hours': 8,
        'Unhealthy': 3,
        'Work_Study_Hours': 6,
        '3-6 hours': 3.5,
        '9-5': 7,
        '9-5 hours': 7
    }

    diet = {
        'More Healty': 0,
        'Healthy': 1,
        'Less than Healthy': 2,
        'Less Healthy': 2,
        'Moderate': 3,
        'Unhealthy': 4,
        'No Healthy': 4
    }

    # Second sommeil dict from original (overwrites first, includes diet values for edge cases)
    sommeil = {
        "More than 8 hours": 9,
        'Less than 5 hours': 4,
        '5-6 hours': 5.5,
        '7-8 hours': 7.5,
        '1-2 hours': 1.5,
        '6-8 hours': 7,
        '4-6 hours': 5,
        '6-7 hours': 6.5,
        '10-11 hours': 10.5,
        '8-9 hours': 8.5,
        '9-11 hours': 10,
        '2-3 hours': 2.5,
        '3-4 hours': 3.5,
        'Moderate': 6,
        '4-5 hours': 4.5,
        '9-6 hours': 7.5,
        '1-3 hours': 2,
        '1-6 hours': 4,
        'Healthy': 1,
        'Less than Healthy': 2,
        'Less Healthy': 2,
        'Unhealthy': 4,
        'No Healthy': 4
    }

    degree = {
        "BCom": "B.Com", "B.Com": "B.Com", "B.Comm": "B.Com",
        "B.Tech": "B.Tech", "BTech": "B.Tech", "B.T": "B.Tech",
        "BSc": "B.Sc", "B.Sc": "B.Sc", "Bachelor of Science": "B.Sc",
        "BArch": "B.Arch", "B.Arch": "B.Arch",
        "BA": "B.A", "B.A": "B.A",
        "BBA": "BBA", "BB": "BBA",
        "BCA": "BCA",
        "BE": "BE",
        "BEd": "B.Ed", "B.Ed": "B.Ed",
        "BPharm": "B.Pharm", "B.Pharm": "B.Pharm",
        "BHM": "BHM",
        "LLB": "LLB", "LL B": "LLB", "LL BA": "LLB", "LL.Com": "LLB", "LLCom": "LLB",
        "MCom": "M.Com", "M.Com": "M.Com",
        "M.Tech": "M.Tech", "MTech": "M.Tech", "M.T": "M.Tech",
        "MSc": "M.Sc", "M.Sc": "M.Sc", "Master of Science": "M.Sc",
        "MBA": "MBA",
        "MCA": "MCA",
        "MD": "MD",
        "ME": "ME",
        "MEd": "M.Ed", "M.Ed": "M.Ed",
        "MArch": "M.Arch", "M.Arch": "M.Arch",
        "MPharm": "M.Pharm", "M.Pharm": "M.Pharm",
        "MA": "MA", "M.A": "MA",
        "MPA": "MPA",
        "LLM": "LLM",
        "PhD": "PhD",
        "MBBS": "MBBS",
        "CA": "CA",
        "Class 12": "Class 12", "12th": "Class 12",
        "Class 11": "Class 11", "11th": "Class 11"
    }

    # Target encoding basé sur le trainset original
    mean_per_city = trainset.groupby('City')[TARGET_FEATURE].mean()
    mean_per_name = trainset.groupby('Name')[TARGET_FEATURE].mean()
    mean_per_profession = trainset.groupby('Profession')[TARGET_FEATURE].mean()

    # Pour Degree, on doit d'abord normaliser les valeurs du trainset puis calculer la moyenne
    trainset_temp = trainset.copy()
    trainset_temp['Degree'] = trainset_temp['Degree'].map(degree)
    mean_per_degree = trainset_temp.groupby('Degree')[TARGET_FEATURE].mean()

    code = {
        'Female': 1,
        'Male': 0,
        'Working Professional': 1,
        'Student': 0,
        'Yes': 1,
        'No': 0
    }

    # Application des encodages
    df['Dietary Habits'] = df['Dietary Habits'].map(diet)
    df['Sleep Duration'] = df['Sleep Duration'].map(sommeil)
    df['Degree'] = df['Degree'].map(degree)

    # Target encoding
    df['City'] = df['City'].map(mean_per_city)
    df['Name'] = df['Name'].map(mean_per_name)
    df['Profession'] = df['Profession'].map(mean_per_profession)
    df['Degree'] = df['Degree'].map(mean_per_degree)

    # Encodage binaire
    for col in ['Gender', 'Working Professional or Student',
                'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']:
        df[col] = df[col].map(code)

    # Suppression de l'id
    df.drop('id', axis=1, inplace=True)

    return df


def imputation(df):
    """
    Impute les valeurs manquantes et combine les colonnes similaires.
    Reproduit exactement la fonction imputation() du notebook original.
    """
    trainset_encoded = get_trainset_encoded()

    # Combinaison des colonnes Pressure et Satisfaction
    df['Pressure'] = np.where(
        df['Working Professional or Student'] == 0,
        df['Academic Pressure'],
        df['Work Pressure']
    )
    df['Satisfaction'] = np.where(
        df['Working Professional or Student'] == 0,
        df['Study Satisfaction'],
        df['Job Satisfaction']
    )

    # Suppression des colonnes originales
    df.drop(['Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction'],
            axis=1, inplace=True)

    # Imputation par la médiane du trainset encodé
    nan_columns = ['Profession', 'Name', 'City', 'Financial Stress',
                   'Sleep Duration', 'Dietary Habits', 'Degree', 'Pressure', 'Satisfaction']
    for col in nan_columns:
        df[col] = df[col].fillna(np.nanmedian(trainset_encoded[col]))

    # Suppression des lignes sans CGPA
    df = df.dropna(subset=['CGPA'])

    return df


def preprocessing(df):
    """
    Pipeline complet de preprocessing.
    Reproduit exactement la fonction preprocessing() du notebook original.
    """
    df = encodage(df)
    df = imputation(df)

    X = df.drop(TARGET_FEATURE, axis=1)
    y = df[TARGET_FEATURE]

    print(y.value_counts())

    return X, y


def test_imputation(df):
    """
    Imputation pour les données de test (sans suppression des NaN CGPA).
    Reproduit exactement la fonction test_imputation() du notebook original.
    """
    trainset_encoded = get_trainset_encoded()

    # Combinaison des colonnes
    df['Pressure'] = np.where(
        df['Working Professional or Student'] == 0,
        df['Academic Pressure'],
        df['Work Pressure']
    )
    df['Satisfaction'] = np.where(
        df['Working Professional or Student'] == 0,
        df['Study Satisfaction'],
        df['Job Satisfaction']
    )

    df.drop(['Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction'],
            axis=1, inplace=True)

    # Imputation par la médiane du trainset encodé (incluant CGPA pour le test)
    nan_columns = ['CGPA', 'Profession', 'Name', 'City', 'Financial Stress',
                   'Sleep Duration', 'Dietary Habits', 'Degree', 'Pressure', 'Satisfaction']
    for col in nan_columns:
        df[col] = df[col].fillna(np.nanmedian(trainset_encoded[col]))

    return df


def test_preprocessing(df):
    """
    Pipeline de preprocessing pour les données de test.
    Reproduit exactement la fonction test_preprocessing() du notebook original.
    """
    df = encodage(df)
    df = test_imputation(df)
    return df
