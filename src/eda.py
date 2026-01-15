"""
Module d'Analyse Exploratoire des Données (EDA)
Fonctions pour l'exploration et la visualisation des données de santé mentale
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuration par défaut
TARGET_FEATURE = 'Depression'


# =============================================================================
# 1. ANALYSE DE LA FORME
# =============================================================================

def get_data_shape(df):
    """
    Affiche les informations de base sur le dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser

    Returns:
    --------
    dict : Informations sur la forme des données
    """
    info = {
        'n_rows': df.shape[0],
        'n_cols': df.shape[1],
        'dtypes': df.dtypes.value_counts().to_dict(),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }

    print(f"Dimensions: {info['n_rows']} lignes × {info['n_cols']} colonnes")
    print(f"\nTypes de variables:")
    for dtype, count in info['dtypes'].items():
        print(f"  - {dtype}: {count}")
    print(f"\nUtilisation mémoire: {info['memory_usage']}")

    return info


def get_dtypes_summary(df):
    """
    Retourne un résumé des types de données par colonne.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser

    Returns:
    --------
    pd.Series : Types de données par colonne
    """
    return df.dtypes


def plot_missing_values(df, figsize=(20, 10)):
    """
    Visualise les valeurs manquantes avec une heatmap.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    figsize : tuple
        Taille de la figure
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df.isna(), cbar=False, yticklabels=False)
    plt.title('Valeurs manquantes (en blanc)')
    plt.tight_layout()
    plt.show()


def get_missing_values_summary(df):
    """
    Retourne un résumé des valeurs manquantes par colonne.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser

    Returns:
    --------
    pd.Series : Pourcentage de valeurs manquantes par colonne (trié)
    """
    missing = (df.isna().sum() / df.shape[0]).sort_values(ascending=False)

    print('Colonnes et pourcentage de valeurs manquantes :\n')
    for col, pct in missing.items():
        if pct > 0:
            print(f"  {col}: {pct:.2%}")

    return missing


# =============================================================================
# 2. ANALYSE DU FOND
# =============================================================================

def plot_target_distribution(df, target=TARGET_FEATURE, figsize=(8, 5)):
    """
    Visualise la distribution de la variable cible.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    target : str
        Nom de la variable cible
    figsize : tuple
        Taille de la figure

    Returns:
    --------
    pd.Series : Répartition de la target
    """
    print(f'Répartition de {target} :')
    distribution = df[target].value_counts(normalize=True)
    print(distribution)

    plt.figure(figsize=figsize)
    df[target].value_counts().plot(kind='bar', color=['steelblue', 'coral'])
    plt.title(f'Distribution de la variable cible ({target})')
    plt.xlabel(target)
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Non (0)', 'Oui (1)'], rotation=0)
    plt.tight_layout()
    plt.show()

    return distribution


def plot_continuous_distributions(df, columns=None, figsize=(10, 4)):
    """
    Visualise la distribution des variables continues.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    columns : list, optional
        Liste des colonnes à visualiser (par défaut: toutes les float)
    figsize : tuple
        Taille de chaque figure
    """
    if columns is None:
        columns = df.select_dtypes('float').columns

    for col in columns:
        plt.figure(figsize=figsize)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution de {col}')
        plt.tight_layout()
        plt.show()


def plot_categorical_distributions(df, columns=None, top_n=10, figsize=(8, 8)):
    """
    Visualise la distribution des variables catégorielles (top N valeurs).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    columns : list, optional
        Liste des colonnes à visualiser (par défaut: toutes les object)
    top_n : int
        Nombre de valeurs à afficher
    figsize : tuple
        Taille de chaque figure
    """
    if columns is None:
        columns = df.select_dtypes('object').columns

    for col in columns:
        plt.figure(figsize=figsize)
        df[col].value_counts().sort_values(ascending=False)[:top_n].plot.pie(autopct='%1.1f%%')
        plt.title(f'Distribution de {col} (top {top_n})')
        plt.ylabel('')
        plt.tight_layout()
        plt.show()


def print_unique_values(df, columns=None, max_display=50):
    """
    Affiche les valeurs uniques pour les colonnes catégorielles.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    columns : list, optional
        Liste des colonnes (par défaut: toutes les object)
    max_display : int
        Nombre maximum de valeurs à afficher
    """
    if columns is None:
        columns = df.select_dtypes('object').columns

    for col in columns:
        unique_vals = df[col].unique()
        n_unique = len(unique_vals)
        print(f'{col:-<50} ({n_unique} valeurs uniques)')
        if n_unique <= max_display:
            print(f'  {unique_vals[:max_display]}')
        else:
            print(f'  {unique_vals[:max_display]}... (+{n_unique - max_display} autres)')
        print()


# =============================================================================
# 2.3 RELATIONS FEATURES-TARGET
# =============================================================================

def create_target_subsets(df, target=TARGET_FEATURE):
    """
    Crée les sous-ensembles positifs et négatifs selon la target.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    target : str
        Nom de la variable cible

    Returns:
    --------
    tuple : (positive_df, negative_df)
    """
    positive_df = df[df[target] == 1]
    negative_df = df[df[target] == 0]

    print(f"Subset positif (Depression=1): {len(positive_df)} lignes ({len(positive_df)/len(df):.1%})")
    print(f"Subset négatif (Depression=0): {len(negative_df)} lignes ({len(negative_df)/len(df):.1%})")

    return positive_df, negative_df


def plot_feature_by_target(df, column, target=TARGET_FEATURE, figsize=(10, 4)):
    """
    Compare la distribution d'une feature entre les groupes positif/négatif.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    column : str
        Nom de la colonne à visualiser
    target : str
        Nom de la variable cible
    figsize : tuple
        Taille de la figure
    """
    positive_df = df[df[target] == 1]
    negative_df = df[df[target] == 0]

    plt.figure(figsize=figsize)
    sns.histplot(positive_df[column].dropna(), kde=True, label='Déprimés (1)', alpha=0.6)
    sns.histplot(negative_df[column].dropna(), kde=True, label='Non-déprimés (0)', alpha=0.6)
    plt.title(f'Distribution de {column} par {target}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multiple_features_by_target(df, columns, target=TARGET_FEATURE, figsize=(10, 4)):
    """
    Compare la distribution de plusieurs features entre les groupes.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    columns : list
        Liste des colonnes à visualiser
    target : str
        Nom de la variable cible
    figsize : tuple
        Taille de chaque figure
    """
    for col in columns:
        plot_feature_by_target(df, col, target, figsize)


def plot_categorical_by_target(df, column, target=TARGET_FEATURE, top_n=10, figsize=(14, 6)):
    """
    Compare la distribution d'une variable catégorielle entre les groupes.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    column : str
        Nom de la colonne catégorielle
    target : str
        Nom de la variable cible
    top_n : int
        Nombre de valeurs à afficher
    figsize : tuple
        Taille de la figure
    """
    positive_df = df[df[target] == 1]
    negative_df = df[df[target] == 0]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    positive_df[column].value_counts().sort_values(ascending=False)[:top_n].plot.pie(
        ax=axes[0], autopct='%1.1f%%')
    axes[0].set_title(f'{column} - Déprimés (1)')
    axes[0].set_ylabel('')

    negative_df[column].value_counts().sort_values(ascending=False)[:top_n].plot.pie(
        ax=axes[1], autopct='%1.1f%%')
    axes[1].set_title(f'{column} - Non-déprimés (0)')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.show()


def plot_crosstab_heatmap(df, column, target=TARGET_FEATURE, figsize=(8, 5)):
    """
    Affiche une heatmap de la table de contingence entre une variable et la target.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    column : str
        Nom de la colonne catégorielle
    target : str
        Nom de la variable cible
    figsize : tuple
        Taille de la figure
    """
    plt.figure(figsize=figsize)
    ct = pd.crosstab(df[target], df[column])
    sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Table de contingence: {target} vs {column}')
    plt.tight_layout()
    plt.show()


def plot_multiple_crosstabs(df, columns, target=TARGET_FEATURE, figsize=(8, 5)):
    """
    Affiche les heatmaps de contingence pour plusieurs variables.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    columns : list
        Liste des colonnes catégorielles
    target : str
        Nom de la variable cible
    figsize : tuple
        Taille de chaque figure
    """
    for col in columns:
        plot_crosstab_heatmap(df, col, target, figsize)


# =============================================================================
# 3. ANALYSE DU FOND COMPLÉMENTAIRE
# =============================================================================

def plot_correlation_matrix(df, columns=None, figsize=(12, 10)):
    """
    Affiche la matrice de corrélation des variables numériques.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    columns : list, optional
        Liste des colonnes (par défaut: toutes les numériques)
    figsize : tuple
        Taille de la figure

    Returns:
    --------
    pd.DataFrame : Matrice de corrélation
    """
    if columns is None:
        columns = df.select_dtypes(include=['float', 'int']).columns

    corr_matrix = df[columns].corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Matrice de corrélation')
    plt.tight_layout()
    plt.show()

    return corr_matrix


def plot_scatter_by_target(df, x_col, y_col, target=TARGET_FEATURE, figsize=(10, 6)):
    """
    Affiche un scatter plot coloré par la target.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    x_col : str
        Colonne pour l'axe X
    y_col : str
        Colonne pour l'axe Y
    target : str
        Variable cible pour la couleur
    figsize : tuple
        Taille de la figure
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=target, alpha=0.5)
    plt.title(f'{y_col} vs {x_col} par {target}')
    plt.tight_layout()
    plt.show()


def get_correlation_with_column(df, column, n_top=10):
    """
    Retourne les corrélations d'une colonne avec les autres variables numériques.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    column : str
        Colonne de référence
    n_top : int
        Nombre de corrélations à afficher

    Returns:
    --------
    pd.Series : Corrélations triées
    """
    correlations = df.select_dtypes(include=['number']).corr()[column].sort_values()

    print(f'Corrélations avec {column}:\n')
    print(correlations)

    return correlations


# =============================================================================
# TRANSFORMATIONS POUR L'ANALYSE
# =============================================================================

def categorize_sleep_duration(df):
    """
    Catégorise la durée de sommeil en groupes.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame avec colonne 'Sleep Duration'

    Returns:
    --------
    pd.DataFrame : DataFrame avec colonne 'sommeil' ajoutée
    """
    def classify_sleep(row):
        sleep = row['Sleep Duration']
        if sleep in ['Less than 5 hours', '1-2 hours', '2-3 hours', '3-4 hours',
                     '4-5 hours', '1-3 hours', '3-6 hours']:
            return '<5 heures'
        elif sleep in ['5-6 hours', '4-6 hours']:
            return '5-6 heures'
        elif sleep in ['7-8 hours', '6-8 hours', '6-7 hours', 'Moderate', '9-6 hours', '8 hours']:
            return '6-8 heures'
        elif sleep in ['More than 8 hours', '10-11 hours', '8-9 hours', '9-11 hours']:
            return '>8 heures'
        else:
            return 'inconnu'

    df = df.copy()
    df['sommeil'] = df.apply(classify_sleep, axis=1)
    return df


def categorize_dietary_habits(df):
    """
    Convertit les habitudes alimentaires en valeurs numériques.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame avec colonne 'Dietary Habits'

    Returns:
    --------
    pd.DataFrame : DataFrame avec colonne 'diet_float' ajoutée
    """
    def classify_diet(row):
        diet = row['Dietary Habits']
        if diet in ['Healthy', 'Moderate', 'More Healthy']:
            return 1  # Sain
        elif diet in ['Unhealthy', 'Less than Healthy', 'No Healthy', 'Less Healthy']:
            return 2  # Non sain
        else:
            return np.nan

    df = df.copy()
    df['diet_float'] = df.apply(classify_diet, axis=1)
    return df


def categorize_sleep_numeric(df):
    """
    Convertit la catégorie de sommeil en valeurs numériques.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame avec colonne 'sommeil'

    Returns:
    --------
    pd.DataFrame : DataFrame avec colonne 'sommeil_float' ajoutée
    """
    sleep_map = {
        '<5 heures': 1,
        '5-6 heures': 2,
        '6-8 heures': 3,
        '>8 heures': 4,
        'inconnu': np.nan
    }

    df = df.copy()
    df['sommeil_float'] = df['sommeil'].map(sleep_map)
    return df


def plot_feature_by_category(df, feature, category, figsize=(10, 4)):
    """
    Visualise la distribution d'une feature par catégorie.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    feature : str
        Variable continue à visualiser
    category : str
        Variable catégorielle pour le groupement
    figsize : tuple
        Taille de la figure
    """
    plt.figure(figsize=figsize)
    for cat in df[category].unique():
        subset = df[df[category] == cat][feature].dropna()
        if len(subset) > 0:
            sns.histplot(subset, kde=True, label=cat, alpha=0.5)
    plt.title(f'Distribution de {feature} par {category}')
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# ANALYSE DES NaN
# =============================================================================

def analyze_nan_impact(df, columns, target=TARGET_FEATURE):
    """
    Analyse l'impact des valeurs manquantes sur la distribution de la target.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    columns : list
        Colonnes à analyser
    target : str
        Variable cible

    Returns:
    --------
    dict : Proportions de la target pour chaque sous-ensemble
    """
    results = {}

    for col_group_name, col_group in [('all', columns)]:
        subset = df[col_group + [target]].dropna()
        if len(subset) > 0:
            dist = subset[target].value_counts(normalize=True)
            results[col_group_name] = dist
            print(f'Avec colonnes {col_group}:')
            print(f'  {len(subset)} lignes restantes')
            print(f'  Distribution de {target}:')
            print(dist)
            print()

    return results


def count_remaining_after_dropna(df):
    """
    Compte le nombre de lignes restantes après suppression des NaN.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser

    Returns:
    --------
    int : Nombre de lignes sans NaN
    """
    n_remaining = len(df.dropna())
    print(f'Lignes restantes après dropna(): {n_remaining} / {len(df)} ({n_remaining/len(df):.1%})')
    return n_remaining


# =============================================================================
# TESTS STATISTIQUES
# =============================================================================

def perform_ttest(df, column, target=TARGET_FEATURE):
    """
    Effectue un test t de Student pour comparer les moyennes entre les groupes.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    column : str
        Variable numérique à tester
    target : str
        Variable cible pour le groupement

    Returns:
    --------
    dict : Résultats du test (t-statistic, p-value)
    """
    group_0 = df[df[target] == 0][column].dropna()
    group_1 = df[df[target] == 1][column].dropna()

    if len(group_0) == 0 or len(group_1) == 0:
        print(f"Pas assez de données pour {column}")
        return None

    t_stat, p_value = stats.ttest_ind(group_0, group_1)

    print(f"Test t de Student pour {column}:")
    print(f"  Moyenne groupe 0 (non-déprimés): {group_0.mean():.3f}")
    print(f"  Moyenne groupe 1 (déprimés): {group_1.mean():.3f}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  → Différence significative (p < 0.05)")
    else:
        print(f"  → Pas de différence significative (p >= 0.05)")

    return {'t_statistic': t_stat, 'p_value': p_value}
