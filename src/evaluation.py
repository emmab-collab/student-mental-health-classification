"""
Module d'évaluation pour la prédiction de dépression
Contient les fonctions d'évaluation, courbes d'apprentissage et precision-recall
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)
from sklearn.model_selection import learning_curve


def evaluation(model, X_train, y_train, X_test, y_test):
    """
    Évalue un modèle avec matrice de confusion, rapport de classification
    et courbes d'apprentissage.

    Parameters:
    -----------
    model : sklearn estimator
        Modèle à évaluer
    X_train : pd.DataFrame
        Features d'entraînement
    y_train : pd.Series
        Target d'entraînement
    X_test : pd.DataFrame
        Features de test
    y_test : pd.Series
        Target de test

    Returns:
    --------
    model : sklearn estimator
        Modèle entraîné
    """
    # Entraînement
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Affichage des métriques
    print("=" * 50)
    print("MATRICE DE CONFUSION")
    print("=" * 50)
    print(confusion_matrix(y_test, y_pred))
    print()

    print("=" * 50)
    print("RAPPORT DE CLASSIFICATION")
    print("=" * 50)
    print(classification_report(y_test, y_pred))

    # Courbes d'apprentissage
    print("Génération des courbes d'apprentissage...")
    N, train_score, val_score = learning_curve(
        model, X_train, y_train,
        cv=4,
        scoring='f1',
        train_sizes=np.linspace(0.1, 1, 5)
    )

    plt.figure(figsize=(12, 6))
    plt.plot(N, train_score.mean(axis=1), label='Train Score', marker='o')
    plt.plot(N, val_score.mean(axis=1), label='Validation Score', marker='s')
    plt.xlabel('Nombre d\'échantillons d\'entraînement')
    plt.ylabel('F1 Score')
    plt.title('Courbes d\'apprentissage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return model


def evaluate_multiple_models(models_dict, X_train, y_train, X_test, y_test):
    """
    Évalue plusieurs modèles et affiche leurs performances.

    Parameters:
    -----------
    models_dict : dict
        Dictionnaire {nom: modèle}
    X_train : pd.DataFrame
        Features d'entraînement
    y_train : pd.Series
        Target d'entraînement
    X_test : pd.DataFrame
        Features de test
    y_test : pd.Series
        Target de test

    Returns:
    --------
    dict
        Dictionnaire {nom: modèle_entraîné}
    """
    trained_models = {}

    for name, model in models_dict.items():
        print("\n" + "=" * 60)
        print(f"  MODÈLE: {name}")
        print("=" * 60)

        trained_models[name] = evaluation(model, X_train, y_train, X_test, y_test)

    return trained_models


def plot_precision_recall_curve(model, X_test, y_test):
    """
    Affiche la courbe Precision-Recall.

    Parameters:
    -----------
    model : sklearn estimator
        Modèle entraîné avec decision_function
    X_test : pd.DataFrame
        Features de test
    y_test : pd.Series
        Target de test

    Returns:
    --------
    tuple (precision, recall, threshold)
        Valeurs de la courbe PR
    """
    # Calcul des scores de décision
    decision_scores = model.decision_function(X_test)

    # Calcul de la courbe PR
    precision, recall, threshold = precision_recall_curve(y_test, decision_scores)

    # Affichage
    plt.figure(figsize=(12, 6))
    plt.plot(threshold, precision[:-1], label='Precision', linewidth=2)
    plt.plot(threshold, recall[:-1], label='Recall', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Courbe Precision-Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return precision, recall, threshold


def model_final(model, X, threshold=0):
    """
    Applique un seuil personnalisé pour les prédictions.

    Parameters:
    -----------
    model : sklearn estimator
        Modèle entraîné avec decision_function
    X : pd.DataFrame
        Features à prédire
    threshold : float, default=0
        Seuil de décision

    Returns:
    --------
    np.ndarray
        Prédictions binaires
    """
    return model.decision_function(X) > threshold


def evaluate_with_threshold(model, X_test, y_test, threshold=0):
    """
    Évalue les performances avec un seuil personnalisé.

    Parameters:
    -----------
    model : sklearn estimator
        Modèle entraîné
    X_test : pd.DataFrame
        Features de test
    y_test : pd.Series
        Target de test
    threshold : float, default=0
        Seuil de décision

    Returns:
    --------
    dict
        Dictionnaire avec F1 et Recall
    """
    y_pred = model_final(model, X_test, threshold=threshold)

    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Threshold: {threshold}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall Score: {recall:.4f}")

    return {'f1': f1, 'recall': recall}


def find_optimal_threshold(model, X_test, y_test, target_recall=0.9):
    """
    Trouve le seuil optimal pour atteindre un recall cible.

    Parameters:
    -----------
    model : sklearn estimator
        Modèle entraîné
    X_test : pd.DataFrame
        Features de test
    y_test : pd.Series
        Target de test
    target_recall : float, default=0.9
        Recall cible

    Returns:
    --------
    float
        Seuil optimal
    """
    decision_scores = model.decision_function(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, decision_scores)

    # Trouver le seuil pour le recall cible
    for i, r in enumerate(recall[:-1]):
        if r >= target_recall:
            optimal_threshold = thresholds[i]
            print(f"Seuil optimal pour recall >= {target_recall}: {optimal_threshold:.4f}")
            print(f"Recall obtenu: {r:.4f}")
            print(f"Precision correspondante: {precision[i]:.4f}")
            return optimal_threshold

    print(f"Impossible d'atteindre le recall cible de {target_recall}")
    return 0
