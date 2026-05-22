#  Ames Housing — Prédiction de Prix Avancée

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-purple)
![Status](https://img.shields.io/badge/Status-Terminé-success)

---

##  Description

Analyse avancée du marché immobilier d'Ames, Iowa.
Prédiction du prix de vente à partir de 79 features
couvrant tous les aspects d'une maison résidentielle —
surface, qualité, équipements, localisation, année de construction.

> 💼 Projet portfolio — démonstration de compétences en
> nettoyage avancé, Feature Engineering et Machine Learning.

---

##  Objectifs

- Nettoyer et imputer intelligemment 19 colonnes avec valeurs manquantes
- Créer de nouvelles features plus informatives (Feature Engineering)
- Sélectionner automatiquement les 50 meilleures features (SelectKBest)
- Comparer 5 modèles ML et atteindre un **R² > 0.85**
- Estimer le prix d'une maison avec intervalle de confiance

---

##  Dataset

| Indicateur | Valeur |
|---|---|
|  Maisons | 1 460 |
|  Features originales | 79 |
|  Features après Feature Engineering | 83 |
|  Features après One-Hot Encoding | 252 |
|  Features sélectionnées (SelectKBest) | 50 |
|  Prix moyen | ~180 000 $ |
|  Prix médian | ~163 000 $ |

---

##  Pipeline complet

```
Données brutes (79 features)
        ↓
① Nettoyage & imputation intelligente
  • Suppression colonnes > 50% manquants (PoolQC, Alley...)
  • NaN Garage/Basement → 'None' (= pas de garage)
  • LotFrontage → médiane par quartier (Neighborhood)
        ↓
② Feature Engineering — 8 nouvelles variables
  • qualite_x_surface  = OverallQual × surface_totale
  • surface_totale     = Bsmt + 1st + 2nd Floor
  • age_maison         = YrSold - YearBuilt
  • total_bains, age_renovation, maison_neuve...
        ↓
③ One-Hot Encoding (38 variables catégorielles → 169 colonnes)
        ↓
④ Log transformation de SalePrice
  • Skewness brut : 1.883 → après log : 0.121
        ↓
⑤ SelectKBest (f_regression) — 50 meilleures features
        ↓
⑥ Pipeline StandardScaler + Modèle ML
        ↓
⑦ Cross-validation 5 folds + évaluation finale
```

---

##  Performances des modèles ML

| Modèle | R² Test | R² CV | MAE | RMSE |
|---|---|---|---|---|
| Régression Linéaire | 0.858 | 0.840 ± 0.051 | 18 979 $ | 32 997 $ |
| Ridge | 0.874 | 0.835 ± 0.063 | 18 603 $ | 31 141 $ |
| Lasso | 0.871 | 0.833 ± 0.068 | 18 619 $ | 31 431 $ |
| Random Forest | 0.889 | 0.858 ± 0.018 | 17 643 $ | 29 207 $ |
| **Gradient Boosting** | **0.902** | **0.873 ± 0.021** | **17 579 $** | **27 422 $** |

>  **Meilleur modèle : Gradient Boosting**
> R² = 0.902 — objectif 0.85 dépassé 
> Erreur moyenne : 17 579 $ sur un prix moyen de 180 000 $ (~9.8%)

---

##  Top features identifiées

| Rang | Feature | Score F | Origine |
|---|---|---|---|
| 1 | qualite_x_surface | 4 002 | 🆕 Feature Engineering |
| 2 | OverallQual | 2 437 | Originale |
| 3 | Surface_totale | 2 299 | 🆕 Feature Engineering |
| 4 | GrLivArea | 1 471 | Originale |
| 5 | GarageCars | 1 014 | Originale |
| 6 | total_bains | 968 | 🆕 Feature Engineering |
| 7 | GarageArea | 927 | Originale |
| 8 | TotalBsmtSF | 880 | Originale |

>  3 des 6 meilleures features ont été **créées** par
> Feature Engineering — preuve de son impact sur le R²

---

##  Exemple de prédiction

```
Maison estimée :
  Qualité générale  : 8/10
  Surface totale    : 2 500 sq ft
  Garage            : 2 voitures
  Age               : 10 ans
  Salles de bain    : 3

Estimations par modèle :
  Régression Linéaire  → 190 650 $
  Ridge                → 193 705 $
  Lasso                → 203 091 $
  Random Forest        → 193 943 $
  Gradient Boosting    → 199 162 $
  ─────────────────────────────────
  Prix moyen estimé    → 196 110 $
  Intervalle (± MAE)   → 178 531 $ — 213 689 $
```

---

##  Visualisations

### Valeurs manquantes par colonne
![Manquants](outputs/01_valeurs_manquantes.png)

### Top 20 features — SelectKBest
![Features](outputs/02_top20_features.png)

### Distribution SalePrice — brut vs log
![Distribution](outputs/03_distribution_saleprice.png)

### Corrélation entre top features
![Corrélation](outputs/04_heatmap_correlation.png)

### Comparaison des modèles ML
![Modèles](outputs/05_comparaison_modeles.png)

### Importance des features — Gradient Boosting
![Importance](outputs/06_importance_features.png)

---

##  Leçons clés

**1 — Feature Engineering > plus de données**
Les 2 meilleures features (`qualite_x_surface`, `surface_totale`)
ont été créées, pas trouvées dans le dataset original.

**2 — Log transformation indispensable**
Skewness 1.883 → 0.121 après log — amélioration significative du R².

**3 — Gradient Boosting séquentiel > Random Forest parallèle**
GB corrige ses erreurs à chaque étape → R² 0.902 vs 0.889 pour RF.

**4 — Pipeline = bonne pratique absolue**
Évite le data leakage lors de la cross-validation.

---

##  Statut du projet

- [x] Chargement et exploration
- [x] Nettoyage et imputation intelligente
- [x] Feature Engineering (8 nouvelles variables)
- [x] One-Hot Encoding
- [x] Log transformation SalePrice
- [x] Sélection de features (SelectKBest k=50)
- [x] Entraînement 5 modèles ML
- [x] Cross-validation avec Pipeline
- [x] Comparaison et évaluation
- [x] Prédiction sur nouvelle maison
- [ ] Optimisation hyperparamètres (GridSearchCV)
- [ ] XGBoost et LightGBM

---

##  Stack technique

| Librairie | Usage |
|---|---|
| `pandas` | Manipulation et nettoyage |
| `numpy` | Calculs & log transformation |
| `matplotlib` / `seaborn` | Visualisations statiques |
| `plotly` | Graphiques interactifs |
| `scikit-learn` | Pipeline, SelectKBest, ML complet |

---

##  Lancer le projet

```bash
git clone https://github.com/VOTRE_USERNAME/ames-housing-analysis.git
cd ames-housing-analysis
pip install -r requirements.txt
jupyter notebook notebooks/analyse_ames.ipynb
```

---

## Structure

```
ames-housing-analysis/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── analyse_ames.ipynb
├── outputs/
│   ├── 01_valeurs_manquantes.png
│   ├── 02_top20_features.png
│   ├── 03_distribution_saleprice.png
│   ├── 04_heatmap_correlation.png
│   ├── 05_comparaison_modeles.png
│   └── 06_importance_features.png
├── README.md
└── requirements.txt
```

---

## 👤 Auteur

**Mouad AOUS** — Ingénieur Qualité de l'Air | Data Analyst
🔗 [LinkedIn](https://www.linkedin.com/in/mouad-aous/)
📧 mouad.aous97@gmail.com