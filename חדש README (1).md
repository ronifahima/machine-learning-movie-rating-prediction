# IMDb Movie Rating Prediction — Part 2

## Project Overview
This project builds a machine learning model to predict the average IMDb rating (`averageRating`) of a movie **before its release**, using only features available prior to release.

---

## Models
- **Elastic Net** (mandatory): Linear regression with L1+L2 regularization. Hyperparameter tuning of `alpha` and `l1_ratio` via 10-Fold Cross Validation with GridSearchCV.
- **Random Forest** (chosen model): Ensemble of decision trees. Selected for its ability to capture non-linear relationships and provide built-in feature importance. Hyperparameter tuning via GridSearchCV.

**Best model: Random Forest** (RMSE=1.1090, MAE=0.8444, R²=0.2641)

---

## File Structure
```
├── notebook.ipynb        # Full code: prepare_data, training, evaluation, error analysis, fairness analysis
├── model.pkl             # Best trained model (Random Forest pipeline), serialized with joblib
├── report.pdf            # Project report (up to 8 pages)
├── README.md             # This file
├── requirements.txt      # Required libraries with versions
└── dataset.csv           # Input dataset (IMDb movies)
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the notebook
```bash
jupyter notebook notebook.ipynb
```
Run all cells top to bottom. The notebook will:
- Load and explore the dataset
- Run `prepare_data()` to engineer features
- Train both models with 10-Fold CV
- Evaluate and compare models
- Run Error Analysis and Fairness Analysis
- Compute Feature Importance

### 3. Test set inference (competition)
```python
import pandas as pd
import joblib

model = joblib.load('model.pkl')
df_2025 = pd.read_csv('test_set.csv')
X = prepare_data(df_2025)
y_pred = model.predict(X)
```

---

## prepare_data Function
The `prepare_data(df)` function:
- Accepts a raw DataFrame in the same format as `dataset.csv`
- Cleans bad values (`\N`, `Not Found`, `[citation needed]`, etc.)
- Engineers 16 features (see table below)
- Returns a processed DataFrame ready for `model.predict()`
- Is **deterministic** — same output for same input
- Works on a single row or hundreds of rows

### Features
| Feature | Type | Description |
|---------|------|-------------|
| `startYear` | Numeric | Release year (invalid values → NaN) |
| `runtimeMinutes` | Numeric | Movie runtime in minutes |
| `num_genres` | Engineered | Number of genres per movie |
| `runtime_per_genre` | Engineered | runtimeMinutes / num_genres (interaction feature) |
| `is_drama` | Binary | 1 if Drama genre |
| `is_comedy` | Binary | 1 if Comedy genre |
| `is_action` | Binary | 1 if Action genre |
| `is_thriller` | Binary | 1 if Thriller genre |
| `is_romance` | Binary | 1 if Romance genre |
| `is_horror` | Binary | 1 if Horror genre |
| `is_documentary` | Binary | 1 if Documentary genre |
| `is_crime` | Binary | 1 if Crime genre |
| `is_adventure` | Binary | 1 if Adventure genre |
| `is_animation` | Binary | 1 if Animation genre |
| `is_english` | Binary | 1 if primary language is English (0.5 if unknown) |
| `is_us` | Binary | 1 if US production (0.5 if unknown) |

---

## Data Leakage Prevention
The following fields are **strictly excluded** from all features:
- `averageRating` — target variable
- `numVotes` — accumulates after release
- `BoxOffice` — accumulates after release

All preprocessing (scaling, imputation) occurs **inside the CV pipeline** on training folds only.

---

## Requirements
- Python 3.10+
- See `requirements.txt` for full library list
- `random_state=42` used throughout for reproducibility
