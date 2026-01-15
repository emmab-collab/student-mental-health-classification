# Student Mental Health - Depression Prediction

Projet de Machine Learning pour prédire la dépression chez les étudiants et professionnels à partir de données cliniques et socio-professionnelles.

## Objectif

Prédire si une personne est en dépression (variable binaire) avec les métriques cibles :
- **F1 Score** ≥ 0.5
- **Recall** ≥ 0.7 (priorité : ne pas manquer de cas positifs)

### Résultats obtenus
- **F1 Score : 0.86**
- **Recall : 0.94**

## Structure du projet

```
STUDENT MENTAL HEALTH/
├── data/
│   ├── train.csv          # Données d'entraînement
│   ├── test.csv           # Données de test
│   └── submission.csv     # Prédictions pour Kaggle
├── notebooks/
│   └── main_pipeline.ipynb    # Notebook principal
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   # Encodage et imputation
│   ├── modeling.py        # Modèles ML
│   └── evaluation.py      # Métriques et courbes
├── EDA_Student_Mental_Health(1).ipynb  # Analyse exploratoire
├── pre-processing-mental-health.ipynb  # Notebook original
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Notebook principal

Ouvrir `notebooks/main_pipeline.ipynb` et exécuter les cellules dans l'ordre.

### 2. Utilisation des modules

```python
from src.preprocessing import set_trainset, preprocessing
from src.modeling import create_models, get_best_model
from src.evaluation import evaluation, plot_precision_recall_curve

# Charger et préparer les données
set_trainset(trainset)
X_train, y_train = preprocessing(trainset)

# Évaluer les modèles
models = create_models()
for name, model in models.items():
    evaluation(model, X_train, y_train, X_test, y_test)
```

## Preprocessing

### Encodage
- **Sleep Duration** : conversion en heures numériques
- **Dietary Habits** : encodage ordinal (0-4)
- **City, Name, Profession, Degree** : target encoding
- **Variables binaires** : 0/1

### Imputation
- Combinaison Academic Pressure / Work Pressure → `Pressure`
- Combinaison Study Satisfaction / Job Satisfaction → `Satisfaction`
- Valeurs manquantes : médiane du trainset

## Modèles

| Modèle | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| RandomForest | 0.86 | 0.87 | 0.86 |
| AdaBoost | 0.87 | 0.86 | 0.86 |
| SVM | 0.86 | 0.87 | 0.86 |
| KNN | 0.85 | 0.85 | 0.85 |

### Modèle final
**AdaBoostClassifier** avec :
- `estimator` : DecisionTreeClassifier(max_depth=1)
- `algorithm` : SAMME.R
- `learning_rate` : 1.0
- `n_estimators` : 100

## Insights EDA

- Classes déséquilibrées : 18% dépressifs / 82% non-dépressifs
- Étudiants plus touchés que les travailleurs
- Tranche 20-30 ans particulièrement affectée
- Pression académique/professionnelle corrélée à la dépression
- Lien fort entre idées suicidaires et dépression
- Hygiène de vie (sommeil, alimentation) influence la dépression

## Données

Source : [Kaggle Playground Series S4E11](https://www.kaggle.com/competitions/playground-series-s4e11)

- 140,700 observations
- 20 variables
- Target : `Depression` (0/1)
