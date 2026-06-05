# Movie Rating Prediction Model

**Course:** Machine Learning - Summary Assignment Part 2

**Submitted by:**
Zohar Colp - 322435918
Roni Fahima - 212009260

---

## Project Goal

Build a model that predicts a movie's average IMDb rating (`averageRating`) before its release, using only information available at production time.

---

## Repository Structure

```
├── notebook.ipynb        # Full pipeline: EDA, feature engineering, training, evaluation
├── model.pkl             # Trained Elastic Net model, serialized with joblib
├── report.pdf            # Written report (up to 8 pages)
├── README.md             # This file
└── requirements.txt      # Python dependencies with versions
```

---

## Dataset

The dataset is a CSV file containing movie-level data collected from IMDb and additional external sources. After cleaning invalid values, handling missing data, and engineering new features via the `prepare_data()` function, the dataset was prepared for analysis and used for training and evaluating the prediction models.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. File placement

Make sure all project files are in the same directory:

```
notebook.ipynb
model.pkl
requirements.txt
dataset.csv
```

### 3. Run the notebook

```bash
jupyter notebook notebook.ipynb
```

Run all cells top to bottom:
`Kernel → Restart & Run All`

---

## Inference on New Data

The trained model is saved in `model.pkl`.

Before making predictions, run the `prepare_data()` function defined in `notebook.ipynb` to apply the same preprocessing pipeline used during training.

Example:

```python
import pandas as pd
import joblib

df = pd.read_csv("dataset.csv")
X = prepare_data(df)

model = joblib.load("model.pkl")
predictions = model.predict(X)
```

---

## Models

### Elastic Net
Linear regression model combining L1 and L2 regularization.
`alpha` and `l1_ratio` were tuned via GridSearchCV with 10-Fold Cross Validation inside a Pipeline.

### Random Forest
Ensemble tree-based model capable of capturing non-linear relationships between features and movie ratings.
`n_estimators`, `max_depth`, and `min_samples_split` were tuned via GridSearchCV with 10-Fold Cross Validation inside a Pipeline.

### Submission Model
The file `model.pkl` contains the **Elastic Net** model, selected for the competition per the instructor's guidelines, after full hyperparameter tuning via GridSearchCV and 10-Fold Cross Validation.

---

## Evaluation Metrics

Both models were evaluated using:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

All metrics were computed using 10-Fold Cross Validation.

---

## Features

16 features are used, engineered inside `prepare_data()`:

| Feature | Type | Description |
|---------|------|-------------|
| `startYear` | Numeric | Release year |
| `runtimeMinutes` | Numeric | Film duration in minutes |
| `num_genres` | Numeric (engineered) | Number of genres listed |
| `runtime_per_genre` | Numeric (engineered) | runtimeMinutes / num_genres (ratio feature) |
| `is_drama` | Binary | 1 if genre includes Drama |
| `is_comedy` | Binary | 1 if genre includes Comedy |
| `is_action` | Binary | 1 if genre includes Action |
| `is_thriller` | Binary | 1 if genre includes Thriller |
| `is_romance` | Binary | 1 if genre includes Romance |
| `is_horror` | Binary | 1 if genre includes Horror |
| `is_documentary` | Binary | 1 if genre includes Documentary |
| `is_crime` | Binary | 1 if genre includes Crime |
| `is_adventure` | Binary | 1 if genre includes Adventure |
| `is_animation` | Binary | 1 if genre includes Animation |
| `is_english` | Binary | 1 if language is English, 0 otherwise, 0.5 if unknown |
| `is_us` | Binary | 1 if country is United States, 0 otherwise, 0.5 if unknown |

---

## Data Leakage Prevention

The following fields are strictly excluded from all features:
- `averageRating` - target variable
- `numVotes` - accumulates after release
- `BoxOffice` - accumulates after release

All preprocessing (imputation, scaling, encoding) is performed inside the Pipeline, fitted on training folds only. No data leakage.

---

## Feature Importance

Feature importance was analyzed using:
- Standardized coefficients for Elastic Net
- Permutation Importance for Random Forest
- SHAP values for Random Forest

---

## Reproducibility

`random_state=42` is set in all relevant steps:
- `KFold`
- `GridSearchCV`
- `RandomForestRegressor`

---
