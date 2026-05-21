#  Ames Housing — Prédiction de Prix Avancée

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-En%20cours-yellow)

##  Description

Analyse avancée du marché immobilier d'Ames, Iowa.
Prédiction du prix de vente à partir de 79 features
couvrant tous les aspects d'une maison résidentielle.

> Projet portfolio — démonstration de compétences
> en Feature Engineering avancé et Machine Learning.

---

##  Dataset

| Indicateur | Valeur |
|---|---|
| Maisons | 1 460 |
| Features originales | 79 |
| Features après encodage | 252 |
| Features sélectionnées | 50 |
| Prix moyen | ~180 000 $ |

---

##  Pipeline complet

```
Données brutes (79 features)
       ↓
Nettoyage & imputation intelligente
(médiane par quartier pour LotFrontage)
       ↓
Feature Engineering (8 nouvelles variables)
- age_maison, surface_totale, total_bains
- qualite_x_surface, maison_neuve...
       ↓
One-Hot Encoding (252 features)
       ↓
SelectKBest — 50 meilleures features
       ↓
Log transformation de SalePrice
       ↓
Modèles ML + Cross-validation
```

---

##  Top features identifiées

| Rang | Feature | Score F | Origine |
|---|---|---|---|
| 1 | qualite_x_surface | 4 002 | 🆕 Créée |
| 2 | OverallQual | 2 437 | Originale |
| 3 | Surface_totale | 2 299 | 🆕 Créée |
| 4 | GrLivArea | 1 471 | Originale |
| 5 | GarageCars | 1 014 | Originale |

> 🆕 = features créées par Feature Engineering

---

##  Visualisations

### Valeurs manquantes
![Manquants](outputs/01_valeurs_manquantes.png)

### Top 20 features
![Features](outputs/02_top20_features.png)

### Distribution SalePrice
![Distribution](outputs/03_distribution_saleprice.png)

### Corrélation entre features
![Corrélation](outputs/04_heatmap_correlation.png)

---

## 🛠️ Stack technique

| Librairie | Usage |
|---|---|
| `pandas` | Manipulation des données |
| `numpy` | Calculs & log transformation |
| `matplotlib` / `seaborn` | Visualisations |
| `plotly` | Graphiques interactifs |
| `scikit-learn` | ML, Pipeline, SelectKBest |

---

##  Lancer le projet

```bash
git clone https://github.com/VOTRE_USERNAME/ames-housing-analysis.git
cd ames-housing-analysis
pip install -r requirements.txt
jupyter notebook notebooks/analyse_ames.ipynb
```

---

##  Structure

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
│   └── 04_heatmap_correlation.png
├── README.md
└── requirements.txt
```

---

## 🔄 Statut du projet

- [x] Chargement et exploration
- [x] Nettoyage et imputation
- [x] Feature Engineering
- [x] Encodage One-Hot
- [x] Sélection de features (SelectKBest)
- [x] Log transformation SalePrice
- [ ] Entraînement des modèles ML
- [ ] Comparaison et évaluation
- [ ] Optimisation hyperparamètres
- [ ] README final

---

## 👤 Auteur

**Mouad AOUS** — Ingénieur Qualité de l'Air | Data Analyst
🔗 [LinkedIn](https://www.linkedin.com/in/mouad-aous/)
📧 mouad.aous97@gmail.com