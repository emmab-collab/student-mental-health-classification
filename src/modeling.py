"""
Module de modélisation pour la prédiction de dépression
Contient les modèles et les fonctions d'optimisation
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV


def create_preprocessor():
    """
    Crée le preprocessor pour les pipelines de modèles.
    Utilise PolynomialFeatures de degré 2.

    Returns:
    --------
    sklearn.pipeline.Pipeline
        Preprocessor pipeline
    """
    return make_pipeline(PolynomialFeatures(2, include_bias=False))


def create_models():
    """
    Crée un dictionnaire de modèles à évaluer.

    Modèles inclus:
    - RandomForest: avec PolynomialFeatures
    - AdaBoost: avec PolynomialFeatures
    - SVM: avec PolynomialFeatures et StandardScaler
    - KNN: avec PolynomialFeatures et StandardScaler

    Returns:
    --------
    dict
        Dictionnaire {nom: modèle}
    """
    preprocessor = create_preprocessor()

    models = {
        'RandomForest': make_pipeline(
            preprocessor,
            RandomForestClassifier(random_state=42)
        ),
        'AdaBoost': make_pipeline(
            preprocessor,
            AdaBoostClassifier(random_state=42)
        ),
        'SVM': make_pipeline(
            preprocessor,
            StandardScaler(),
            SVC(random_state=42)
        ),
        'KNN': make_pipeline(
            preprocessor,
            StandardScaler(),
            KNeighborsClassifier()
        )
    }

    return models


def get_adaboost_hyperparams():
    """
    Retourne les hyperparamètres pour GridSearchCV sur AdaBoost.

    Returns:
    --------
    dict
        Grille d'hyperparamètres
    """
    return {
        'adaboostclassifier__n_estimators': [100],
        'adaboostclassifier__learning_rate': [1.0],
        'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R'],
        'adaboostclassifier__estimator': [
            DecisionTreeClassifier(max_depth=1),
            DecisionTreeClassifier(max_depth=3)
        ]
    }


def optimize_model(model, X_train, y_train, hyper_params, scoring='recall', cv=4):
    """
    Optimise un modèle avec GridSearchCV.

    Parameters:
    -----------
    model : sklearn estimator
        Modèle à optimiser
    X_train : pd.DataFrame
        Features d'entraînement
    y_train : pd.Series
        Target d'entraînement
    hyper_params : dict
        Grille d'hyperparamètres
    scoring : str, default='recall'
        Métrique d'optimisation
    cv : int, default=4
        Nombre de folds pour la validation croisée

    Returns:
    --------
    GridSearchCV
        Objet GridSearchCV fitted
    """
    grid = GridSearchCV(model, hyper_params, scoring=scoring, cv=cv)
    grid.fit(X_train, y_train)

    print(f"Meilleurs paramètres: {grid.best_params_}")
    print(f"Meilleur score ({scoring}): {grid.best_score_:.4f}")

    return grid


def get_best_model():
    """
    Retourne le meilleur modèle avec les hyperparamètres optimaux trouvés.

    Configuration optimale:
    - AdaBoostClassifier avec DecisionTreeClassifier(max_depth=1)
    - algorithm='SAMME.R'
    - learning_rate=1.0
    - n_estimators=100

    Returns:
    --------
    AdaBoostClassifier
        Modèle configuré avec les meilleurs hyperparamètres
    """
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        algorithm='SAMME.R',
        learning_rate=1.0,
        n_estimators=100,
        random_state=42
    )
