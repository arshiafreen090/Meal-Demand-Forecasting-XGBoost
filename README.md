# Meal Demand Forecasting with XGBoost

An end-to-end machine learning project for forecasting weekly meal demand across fulfillment centers using historical demand, meal characteristics, center information, pricing, promotions, and time-series features.

The model forecasts future weekly meal demand for operational planning, inventory preparation, and resource allocation.

> **Project Notebooks**
>
> - **`Meal-Demand-Forecasting-&-Inventory-Planning.ipynb`** → Complete experimentation and analysis notebook containing EDA, time-series analysis, feature engineering, baseline models, model comparisons, validation experiments, error analysis, and feature experiments.
> - **`meal-demand-forecasting-final-model.ipynb`** → Final model notebook containing the selected feature set, final XGBoost pipeline, time-aware validation, recursive forecasting, and final model training.
> - **`Final-model.pkl`** → Serialized final model used by the Streamlit application.

---

## Overview

Meal delivery companies need to estimate future demand across meals and fulfillment centers before orders arrive. The objective is to predict `num_orders` for future weeks using historical demand patterns and operational information.

**Workflow:**
Data → Inspection → EDA → Time-Series Analysis → Feature Engineering → Time-Aware Validation → Baseline Model → XGBoost Development → Feature Enrichment → Final Model → Recursive Forecasting → Streamlit App
 
---

## Business Problem

For each **fulfillment center × meal × future week** combination, the system estimates expected order volume.

The forecast can support:

- raw material planning
- meal preparation capacity
- fulfillment-center demand planning
- inventory planning
- understanding demand variation across meals and centers
- analyzing the relationship between promotions and demand

---

## Dataset

Source: [Meal Demand Forecasting — Kaggle](https://www.kaggle.com/datasets/sureshmecad/meal-demand-forecasting)

| Feature | Description |
|---|---|
| `id` | Unique record identifier |
| `week` | Week number |
| `center_id` | Fulfillment center identifier |
| `meal_id` | Meal identifier |
| `checkout_price` | Price at checkout |
| `base_price` | Base meal price |
| `emailer_for_promotion` | Whether email promotion was used |
| `homepage_featured` | Whether the meal was featured on homepage |
| `num_orders` | **Target variable** |

| Training | Forecast | Data | Quality |
|---|---|---|---|
| **Weeks 1–145** | **Weeks 146–155** | **456,548 rows · 145 weeks · 77 centers · 51 meals** | **0 missing values · 0 duplicates** |

The main dataset is enriched using:

- `meal_info.csv`
- `fulfilment_center_info.csv`

| Meal Information | Fulfillment Center Information |
|---|---|
| **`meal_info.csv`** provides:<br><br>• `meal_id`<br>• `category`<br>• `cuisine` | **`fulfilment_center_info.csv`** provides:<br><br>• `center_id`<br>• `city_code`<br>• `region_code`<br>• `center_type`<br>• `op_area` |

---

## Data Insights

### Target Distribution

`num_orders` is strongly right-skewed:

| Mean | Median | Std. Dev. | Min | Max |
|---:|---:|---:|---:|---:|
| 261.87 | 136 | 395.92 | 13 | 24,299 |

The final model therefore predicts `log1p(num_orders)` and converts predictions back using `np.expm1()`.

### Center × Meal Coverage

Of **3,927** possible Center × Meal combinations (`77 × 51`), **3,597** are observed. Missing weeks are **not treated as zero demand**, since they may indicate unavailable meals or missing observations.

### Autocorrelation

Historical demand shows positive temporal dependence:

| Lag | Correlation |
|---:|---:|
| 1 | **0.694** |
| 8 | **0.521** |

This indicates that recent historical demand contains useful information for forecasting future demand.

---

## Feature Engineering

The final model uses **23 features**.

| Feature Group | Features |
|---|---|
| **Static / Business** | `center_id`, `meal_id`, `category`, `cuisine`, `city_code`, `region_code`, `center_type`, `op_area`, `checkout_price`, `base_price`, `emailer_for_promotion`, `homepage_featured` |
| **Historical** | `lag_1`, `lag_2`, `lag_4`, `lag_8`, `rolling_mean_4`, `rolling_std_4` |
| **Long Lags** | `lag_10`, `lag_11`, `lag_12` |
| **Pair History** | `pair_mean`, `pair_std` |

### Historical Lags

The model uses:

```text
lag_1
lag_2
lag_4
lag_8
lag_10
lag_11
lag_12
```

Instead of relying on `shift()`, lag features are created using exact:

```text
center_id
+
meal_id
+
previous_week
```

matching.

This is important because some Center × Meal combinations contain missing weeks.

For example:

```text
Current Week = 100

lag_1 → demand from week 99
lag_2 → demand from week 98
lag_8 → demand from week 92
```

This ensures that a lag represents the correct calendar week rather than simply the previous available row.

### Rolling Features

The model uses:

```text
rolling_mean_4
rolling_std_4
```

These describe recent demand level and variability.

Only historical observations are used when calculating these features to prevent target leakage.

### Center × Meal Historical Statistics

The final model also includes:

```text
pair_mean
pair_std
```

These capture the historical demand level and variability for each specific Center × Meal combination.

---

## Validation Strategy

Random splitting can leak future information in time-series forecasting, so the project uses **expanding-window, time-aware validation**:

| Fold | Train | Validate |
|---|---|---|
| 1 | Weeks 1–80 | 81–90 |
| 2 | Weeks 1–90 | 91–100 |
| 3 | Weeks 1–100 | 101–110 |
| 4 | Weeks 1–110 | 111–120 |
| 5 | Weeks 1–120 | 121–130 |

This mirrors the real forecasting setup by always training on past data and validating on subsequent weeks.

**Metric:** RMSLE (Root Mean Squared Logarithmic Error), with the competition metric reported as `100 × RMSLE`. **Lower is better.**

---

## Model Development

All major experimentation is documented in:

```text
Meal-Demand-Forecasting-&-Inventory-Planning.ipynb
```

This notebook contains:

- data inspection
- EDA
- target analysis
- time-series analysis
- autocorrelation analysis
- lag experiments
- rolling statistics
- EWMA experiments
- baseline forecasting
- XGBoost development
- metadata enrichment
- feature importance
- feature reduction experiments
- LightGBM comparison
- CatBoost comparison
- price feature experiments
- error analysis
- Center × Meal historical statistics
- validation experiments

The final model is documented separately in:

```text
meal-demand-forecasting-final-model.ipynb
```

---

## Model Comparison

| Model / Experiment | Mean RMSLE |
|---|---:|
| Naive Baseline | 0.8252 |
| Original XGBoost | 0.5424 |
| + Metadata Enrichment | 0.5226 |
| + Center × Meal Statistics | 0.5076 |
| **+ Long Lags (Final)** | **0.5060** |
| LightGBM | 0.5326 |
| CatBoost | 0.5310 |
| Reduced-feature XGBoost | 0.5643 |

The final configuration was selected after comparing multiple feature and model experiments using time-aware recursive validation.

---

## Final Model

The final model is an **XGBoost Regressor** using 23 engineered features.

### Configuration

```text
n_estimators     = 500
learning_rate    = 0.05
max_depth        = 8
subsample        = 0.8
colsample_bytree = 0.8
random_state     = 42
```

The target is trained as:

```text
log1p(num_orders)
```

Categorical variables are encoded using:

```text
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing and model are combined into a Scikit-learn `Pipeline`.

### Final Validation

| Fold | RMSLE |
|---|---:|
| Fold 1 | 0.4877 |
| Fold 2 | 0.5080 |
| Fold 3 | 0.5336 |
| Fold 4 | 0.5140 |
| Fold 5 | 0.4865 |
| **Mean** | **0.50597** |
| Standard Deviation | 0.0176 |

**Final locked benchmark: Mean RMSLE = 0.50597**

The trained model is saved as:

```text
Final-model.pkl
```

---

## Feature Importance

The strongest model signals included:

```text
lag_1
category
emailer_for_promotion
rolling_mean_4
cuisine
homepage_featured
```

Feature importance represents model usage and should not be interpreted as causal evidence.

For example, a high importance for a promotion feature does not by itself prove that the promotion caused an increase in demand.

---

## Error Analysis

The model was evaluated across different demand ranges.

| Demand Range | RMSLE |
|---|---:|
| 0–50 | 0.6547 |
| 51–100 | 0.5398 |
| 101–250 | 0.4971 |
| 251–500 | 0.4197 |
| 501–1000 | 0.4217 |
| 1000+ | 0.4403 |

Low-demand observations remain the most difficult to forecast.

The model also shows some regression toward the middle:

```text
Low actual demand
        ↓
Predictions tend to be higher

High actual demand
        ↓
Predictions tend to be lower
```

This behavior is an important limitation of the current model.

---

## Recursive Forecasting

The final model is trained using:

```text
Weeks 1–145
```

and forecasts:

```text
Weeks 146–155
```

Forecasting is performed recursively.

```text
Week 146
   ↓
Prediction
   ↓
Prediction added to history
   ↓
Week 147
   ↓
Prediction
   ↓
Prediction added to history
   ↓
...
   ↓
Week 155
```

This prevents the forecasting process from using actual future demand that would not be available at prediction time.

---

## Streamlit Application

The trained XGBoost model is integrated into an interactive **Streamlit dashboard** where users can select a fulfillment center, meal, and forecast horizon.

| Dashboard | Forecasting |
|---|---|
| Selected configuration | Demand forecast |
| Forecast summary | Historical vs. forecast |
| Recent demand comparison | Planning insight |
| Historical demand analysis | Forecast details |
| Model information | Methodology, experiments & verification |
| Model limitations | — |

---

### Forecast Stabilization

The application blends raw model predictions with the **recent 4-week demand average** to reduce extreme presentation-level swings.

> **Note:** This stabilization is an application-layer forecasting heuristic and is **not included in the locked RMSLE benchmark**.

## Project Structure

```text
Meal Demand Forecasting/
│
├── app.py
├── Final-model.pkl
│
├── Meal-Demand-Forecasting-&-Inventory-Planning.ipynb
│   └── Complete experimentation & analysis
│
├── meal-demand-forecasting-final-model.ipynb
│   └── Final model & forecasting pipeline
│
├── test_model.py
│   └── Model verification
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── Datasets/
│   ├── fulfilment_center_info.csv
│   ├── meal_info.csv
│   ├── test.csv
│   └── train.csv
│
└── assets/
    └── Project diagrams & screenshots
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Feature Engineering | Pandas, NumPy |
| Model Persistence | Joblib |
| Dashboard | Streamlit |
| Development | Kaggle, VS Code, WSL |

### Key Dependencies

```text
numpy==2.5.3
scikit-learn==1.6.1
xgboost==3.0.2
```

Scikit-learn is pinned because serialized machine-learning pipelines can be sensitive to differences between training and deployment environments.

---

## Run Locally

### Clone the repository

```bash
git clone https://github.com/arshiafreen090/Meal-Demand-Forecasting-XGBoost.git
cd Meal-Demand-Forecasting-XGBoost
```

### Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

---

## Resources

- [GitHub Repository](https://github.com/arshiafreen090/Meal-Demand-Forecasting-XGBoost)
- [Meal Demand Forecasting Dataset — Kaggle](https://www.kaggle.com/datasets/sureshmecad/meal-demand-forecasting)
