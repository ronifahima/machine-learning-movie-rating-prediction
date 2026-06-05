# Movie Rating Prediction Model

**Course:** Machine Learning — Summary Assignment Part 2

**Submitted by:**
Zohar Colp — 322435918
Roni Fahima — 212009260

---

## Project Goal

Build a model that predicts a movie's average IMDb rating (`averageRating`) before its release, using only information available at production time.

---

## Repository Structure

```
├── notebook.ipynb        # Full pipeline: EDA, feature engineering, training, evaluation
├── model.pkl             # Trained best model (Random Forest), serialized with joblib
├── report.pdf            # Written report (up to 8 pages)
├── README.md             # This file
└── requirements.txt      # Python dependencies with versions
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the dataset

Make sure `dataset.csv` is in the same directory as `notebook.ipynb`.

### 3. Run the notebook

```bash
jupyter notebook notebook.ipynb
```

Run all cells top to bottom (`Kernel → Restart & Run All`).

---

## Inference on New Data

```python
import pandas as pd
import joblib

model = joblib.load("model.pkl")

df_new = pd.read_csv("new_movies.csv")
X = prepare_data(df_new)       # defined in notebook.ipynb
y_pred = model.predict(X)
```

---

## Models

- **Elastic Net** — linear regression with combined L1+L2 regularization. Alpha and l1_ratio tuned via 10-fold cross-validation inside a Pipeline.
- **Random Forest** — ensemble tree model. `n_estimators`, `max_depth`, and `min_samples_split` tuned via 10-fold cross-validation inside a Pipeline.

All preprocessing (imputation, scaling, encoding) is performed inside the Pipeline, fitted only on training folds. No data leakage.

---

## Features

16 features are used, engineered inside `prepare_data()`:

| Feature | Type | Description |
|---|---|---|
| `startYear` | Numeric | Release year |
| `runtimeMinutes` | Numeric | Film duration in minutes |
| `num_genres` | Numeric (engineered) | Number of genres listed |
| `runtime_per_genre` | Numeric (engineered) | runtimeMinutes / num_genres (ratio) |
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

## Reproducibility

`random_state=42` is set in all relevant steps (GridSearchCV, RandomForestRegressor, KFold).
