import joblib
import pandas as pd
import numpy as np


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "meal_demand_forecasting_final.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(
    "Datasets/train.csv"
)

meal_info = pd.read_csv(
    "Datasets/meal_info.csv"
)

center_info = pd.read_csv(
    "Datasets/fulfilment_center_info.csv"
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

center_id = 10
meal_id = 1062
forecast_week = 146

checkout_price = 200.0
base_price = 138.0

emailer_for_promotion = 1
homepage_featured = 1


# ============================================================
# METADATA
# ============================================================

meal_row = meal_info[
    meal_info["meal_id"] == meal_id
].iloc[0]

center_row = center_info[
    center_info["center_id"] == center_id
].iloc[0]


# ============================================================
# CENTER × MEAL HISTORY
# ============================================================

pair_history = train[
    (train["center_id"] == center_id) &
    (train["meal_id"] == meal_id) &
    (train["week"] < forecast_week)
].copy()

pair_history = pair_history.sort_values("week")


# ============================================================
# LAGS
# ============================================================

demand_lookup = (
    pair_history
    .set_index("week")["num_orders"]
)

lag_1 = demand_lookup.get(
    forecast_week - 1,
    np.nan
)

lag_2 = demand_lookup.get(
    forecast_week - 2,
    np.nan
)

lag_4 = demand_lookup.get(
    forecast_week - 4,
    np.nan
)

lag_8 = demand_lookup.get(
    forecast_week - 8,
    np.nan
)


# ============================================================
# ROLLING FEATURES
# ============================================================

last_four = pair_history["num_orders"].tail(4)

rolling_mean_4 = last_four.mean()
rolling_std_4 = last_four.std()


# ============================================================
# PAIR STATISTICS
# ============================================================

pair_mean = pair_history["num_orders"].mean()
pair_std = pair_history["num_orders"].std()


# ============================================================
# BUILD ORIGINAL FEATURE ROW
# ============================================================

test_data = pd.DataFrame([{

    "center_id": center_id,
    "meal_id": meal_id,

    "category": meal_row["category"],
    "cuisine": meal_row["cuisine"],

    "city_code": center_row["city_code"],
    "region_code": center_row["region_code"],
    "center_type": center_row["center_type"],
    "op_area": center_row["op_area"],

    "checkout_price": checkout_price,
    "base_price": base_price,

    "emailer_for_promotion": emailer_for_promotion,
    "homepage_featured": homepage_featured,

    "lag_1": lag_1,
    "lag_2": lag_2,
    "lag_4": lag_4,
    "lag_8": lag_8,

    "rolling_mean_4": rolling_mean_4,
    "rolling_std_4": rolling_std_4,

    "pair_mean": pair_mean,
    "pair_std": pair_std
}])


# ============================================================
# ORIGINAL PREDICTION
# ============================================================

normal_prediction = np.expm1(
    model.predict(test_data)
)[0]


# ============================================================
# HIGH-DEMAND TEST
# ============================================================

test_high = test_data.copy()

test_high["lag_1"] = 1000
test_high["lag_2"] = 1000
test_high["lag_4"] = 1000
test_high["lag_8"] = 1000

test_high["rolling_mean_4"] = 1000
test_high["rolling_std_4"] = 50

test_high["pair_mean"] = 1000
test_high["pair_std"] = 100


high_prediction = np.expm1(
    model.predict(test_high)
)[0]


# ============================================================
# LOW-DEMAND TEST
# ============================================================

test_low = test_data.copy()

test_low["lag_1"] = 20
test_low["lag_2"] = 20
test_low["lag_4"] = 20
test_low["lag_8"] = 20

test_low["rolling_mean_4"] = 20
test_low["rolling_std_4"] = 5

test_low["pair_mean"] = 20
test_low["pair_std"] = 5


low_prediction = np.expm1(
    model.predict(test_low)
)[0]


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL SENSITIVITY TEST")
print("=" * 60)

print(f"\nCenter: {center_id}")
print(f"Meal: {meal_id}")
print(f"Forecast Week: {forecast_week}")

print("\nHistorical signals:")
print(f"Lag 1:             {lag_1:.2f}")
print(f"Lag 2:             {lag_2:.2f}")
print(f"Lag 4:             {lag_4:.2f}")
print(f"Lag 8:             {lag_8:.2f}")
print(f"Rolling mean 4:    {rolling_mean_4:.2f}")
print(f"Pair mean:         {pair_mean:.2f}")


print("\n" + "-" * 60)

print(f"Low-demand prediction:     {low_prediction:.2f}")
print(f"Original prediction:       {normal_prediction:.2f}")
print(f"High-demand prediction:    {high_prediction:.2f}")

print("-" * 60)

print(
    f"\nPrediction range: "
    f"{low_prediction:.2f} → "
    f"{normal_prediction:.2f} → "
    f"{high_prediction:.2f}"
)