import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Meal Demand Forecasting",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Final-model.pkl"

DATA_DIR = BASE_DIR / "Datasets"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MEAL_INFO_PATH = DATA_DIR / "meal_info.csv"
CENTER_INFO_PATH = DATA_DIR / "fulfilment_center_info.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       MAIN PAGE
       ====================================================== */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ======================================================
       SIDEBAR WIDTH
       ====================================================== */

    [data-testid="stSidebar"] {
        width: 370px !important;
        min-width: 370px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        width: 370px !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem !important;
        padding-left: 1.35rem !important;
        padding-right: 1.35rem !important;
        padding-bottom: 2rem !important;
    }


    /* ======================================================
       SIDEBAR HEADINGS
       ====================================================== */

    [data-testid="stSidebar"] h1 {
        font-size: 1.8rem !important;
    }

    [data-testid="stSidebar"] h2 {
        font-size: 1.55rem !important;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 1.3rem !important;
    }


    /* ======================================================
       SIDEBAR GENERAL TEXT
       ====================================================== */

    [data-testid="stSidebar"] p {
        font-size: 0.98rem !important;
        line-height: 1.5 !important;
    }


    /* ======================================================
       SIDEBAR LABELS
       ====================================================== */

    [data-testid="stSidebar"] label {
        font-size: 1rem !important;
        line-height: 1.4 !important;
        font-weight: 500 !important;
    }


    /* ======================================================
       SIDEBAR SELECT BOX
       ====================================================== */

    [data-testid="stSidebar"] [data-baseweb="select"] {
        font-size: 1rem !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        min-height: 44px !important;
        font-size: 1rem !important;
    }


    /* ======================================================
       SIDEBAR SELECTED VALUE
       ====================================================== */

    [data-testid="stSidebar"] [data-baseweb="select"] span {
        font-size: 1rem !important;
    }


    /* ======================================================
       SIDEBAR SLIDER
       ====================================================== */

    [data-testid="stSidebar"] [data-testid="stSlider"] {
        padding-top: 0.25rem;
        padding-bottom: 0.6rem;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] p {
        font-size: 1rem !important;
    }


    /* ======================================================
       SIDEBAR BUTTON
       ====================================================== */

    [data-testid="stSidebar"] button {
        font-size: 1rem !important;
        min-height: 44px !important;
    }


    /* ======================================================
       SIDEBAR MODEL INFORMATION
       ====================================================== */

    .sidebar-metric-label {
        font-size: 0.92rem;
        line-height: 1.3;
        color: #b8b8c0;
        margin-top: 15px;
        margin-bottom: 3px;
    }

    .sidebar-metric-value {
        font-size: 1.45rem;
        font-weight: 650;
        line-height: 1.25;
        color: #ffffff;
        margin-bottom: 9px;
    }


    /* ======================================================
       SIDEBAR DIVIDER
       ====================================================== */

    .sidebar-divider {
        border-top: 1px solid rgba(255,255,255,0.14);
        margin: 22px 0 16px 0;
    }


    /* ======================================================
       SELECTED CONFIGURATION
       ====================================================== */

    .config-item {
        font-size: 1.15rem;
        line-height: 1.55;
        padding-top: 4px;
        padding-bottom: 4px;
    }

    .config-item code {
        font-size: 1.05rem;
    }


    /* ======================================================
       MAIN METRICS
       ====================================================== */

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FINAL 23 FEATURES
# ============================================================

FINAL_FEATURES = [
    "center_id",
    "meal_id",
    "category",
    "cuisine",
    "city_code",
    "region_code",
    "center_type",
    "op_area",
    "checkout_price",
    "base_price",
    "emailer_for_promotion",
    "homepage_featured",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
    "rolling_std_4",
    "lag_10",
    "lag_11",
    "lag_12",
    "pair_mean",
    "pair_std"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Final-model.pkl was not found at:\n{MODEL_PATH}"
        )

    artifact = joblib.load(
        MODEL_PATH
    )

    if isinstance(artifact, dict):

        if "model" not in artifact:

            raise KeyError(
                "Final-model.pkl does not contain "
                "the key 'model'."
            )

        model = artifact["model"]

        saved_features = artifact.get(
            "features",
            FINAL_FEATURES
        )

        return model, saved_features

    return artifact, FINAL_FEATURES


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    train = pd.read_csv(
        TRAIN_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    meal_info = pd.read_csv(
        MEAL_INFO_PATH
    )

    center_info = pd.read_csv(
        CENTER_INFO_PATH
    )

    train_enriched = (
        train
        .merge(
            meal_info,
            on="meal_id",
            how="left"
        )
        .merge(
            center_info,
            on="center_id",
            how="left"
        )
    )

    test_enriched = (
        test
        .merge(
            meal_info,
            on="meal_id",
            how="left"
        )
        .merge(
            center_info,
            on="center_id",
            how="left"
        )
    )

    return (
        train_enriched,
        test_enriched,
        meal_info,
        center_info
    )


# ============================================================
# LOAD MODEL + DATA
# ============================================================

try:

    model, MODEL_FEATURES = load_model()

    (
        train_enriched,
        test_enriched,
        meal_info,
        center_info
    ) = load_data()

except Exception as e:

    st.error(
        f"Error loading project files:\n\n{e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Forecast Settings"
)


# ============================================================
# CENTER SELECTOR
# ============================================================

centers = sorted(
    train_enriched[
        "center_id"
    ].unique()
)

selected_center = st.sidebar.selectbox(
    "Select Fulfillment Center",
    centers
)


# ============================================================
# MEAL SELECTOR
# ============================================================

available_meals = sorted(
    train_enriched[
        train_enriched["center_id"] ==
        selected_center
    ][
        "meal_id"
    ].unique()
)

selected_meal = st.sidebar.selectbox(
    "Select Meal",
    available_meals
)


# ============================================================
# FORECAST HORIZON
# ============================================================

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon",
    min_value=1,
    max_value=10,
    value=2
)


# ============================================================
# GENERATE FORECAST BUTTON
# ============================================================

generate_forecast = st.sidebar.button(
    "Generate Forecast",
    use_container_width=True
)


# ============================================================
# SIDEBAR MODEL INFORMATION
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-divider"></div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="sidebar-metric-label">
        Validation RMSLE
    </div>

    <div class="sidebar-metric-value">
        0.506
    </div>

    <div class="sidebar-metric-label">
        Model
    </div>

    <div class="sidebar-metric-value">
        XGBoost
    </div>

    <div class="sidebar-metric-label">
        Features
    </div>

    <div class="sidebar-metric-value">
        23
    </div>

    <div class="sidebar-metric-label">
        Training Weeks
    </div>

    <div class="sidebar-metric-value">
        1–145
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_forecast_features(
    history_df,
    forecast_df
):

    result = forecast_df.copy()

    group_cols = [
        "center_id",
        "meal_id"
    ]

    # --------------------------------------------------------
    # Historical lags
    # --------------------------------------------------------

    for lag in [
        1,
        2,
        4,
        8,
        10,
        11,
        12
    ]:

        lookup = history_df[
            group_cols +
            [
                "week",
                "num_orders"
            ]
        ].copy()

        lookup = lookup.rename(
            columns={
                "week": "lookup_week",
                "num_orders": f"lag_{lag}"
            }
        )

        result["lookup_week"] = (
            result["week"] - lag
        )

        result = result.merge(
            lookup,
            on=[
                "center_id",
                "meal_id",
                "lookup_week"
            ],
            how="left"
        )

        result = result.drop(
            columns=["lookup_week"]
        )

    # --------------------------------------------------------
    # Rolling statistics
    # --------------------------------------------------------

    rolling_rows = []

    for _, row in result.iterrows():

        center = row[
            "center_id"
        ]

        meal = row[
            "meal_id"
        ]

        week = row[
            "week"
        ]

        pair_history = history_df[
            (history_df["center_id"] == center)
            & (history_df["meal_id"] == meal)
            & (history_df["week"] < week)
        ].sort_values(
            "week"
        )

        recent_orders = (
            pair_history[
                "num_orders"
            ].tail(4)
        )

        if len(recent_orders) > 0:

            rolling_mean = (
                recent_orders.mean()
            )

        else:

            rolling_mean = np.nan

        if len(recent_orders) > 1:

            rolling_std = (
                recent_orders.std()
            )

        else:

            rolling_std = np.nan

        rolling_rows.append(
            {
                "center_id": center,
                "meal_id": meal,
                "week": week,
                "rolling_mean_4": rolling_mean,
                "rolling_std_4": rolling_std
            }
        )

    rolling_df = pd.DataFrame(
        rolling_rows
    )

    result = result.merge(
        rolling_df,
        on=[
            "center_id",
            "meal_id",
            "week"
        ],
        how="left"
    )

    # --------------------------------------------------------
    # Center × Meal statistics
    # --------------------------------------------------------

    pair_stats = (
        history_df
        .groupby(
            group_cols
        )["num_orders"]
        .agg(
            pair_mean="mean",
            pair_std="std"
        )
        .reset_index()
    )

    result = result.merge(
        pair_stats,
        on=group_cols,
        how="left"
    )

    return result


# ============================================================
# FILL MISSING FEATURES
# ============================================================

def fill_missing_features(
    X,
    history_df
):

    X = X.copy()

    global_mean = (
        history_df[
            "num_orders"
        ].mean()
    )

    for idx in X.index:

        center = X.loc[
            idx,
            "center_id"
        ]

        meal = X.loc[
            idx,
            "meal_id"
        ]

        pair_history = history_df[
            (history_df["center_id"] == center)
            & (history_df["meal_id"] == meal)
        ]

        if len(pair_history) > 0:

            pair_mean = (
                pair_history[
                    "num_orders"
                ].mean()
            )

            pair_std = (
                pair_history[
                    "num_orders"
                ].std()
            )

        else:

            pair_mean = global_mean
            pair_std = 0.0

        # ----------------------------------------------------
        # Lags
        # ----------------------------------------------------

        for col in [
            "lag_1",
            "lag_2",
            "lag_4",
            "lag_8",
            "lag_10",
            "lag_11",
            "lag_12"
        ]:

            if col in X.columns:

                if pd.isna(
                    X.loc[idx, col]
                ):

                    X.loc[
                        idx,
                        col
                    ] = pair_mean

        # ----------------------------------------------------
        # Rolling mean
        # ----------------------------------------------------

        if (
            "rolling_mean_4" in X.columns
            and pd.isna(
                X.loc[
                    idx,
                    "rolling_mean_4"
                ]
            )
        ):

            X.loc[
                idx,
                "rolling_mean_4"
            ] = pair_mean

        # ----------------------------------------------------
        # Rolling std
        # ----------------------------------------------------

        if (
            "rolling_std_4" in X.columns
            and pd.isna(
                X.loc[
                    idx,
                    "rolling_std_4"
                ]
            )
        ):

            X.loc[
                idx,
                "rolling_std_4"
            ] = (
                0.0
                if pd.isna(pair_std)
                else pair_std
            )

        # ----------------------------------------------------
        # Pair mean
        # ----------------------------------------------------

        if (
            "pair_mean" in X.columns
            and pd.isna(
                X.loc[
                    idx,
                    "pair_mean"
                ]
            )
        ):

            X.loc[
                idx,
                "pair_mean"
            ] = pair_mean

        # ----------------------------------------------------
        # Pair std
        # ----------------------------------------------------

        if (
            "pair_std" in X.columns
            and pd.isna(
                X.loc[
                    idx,
                    "pair_std"
                ]
            )
        ):

            X.loc[
                idx,
                "pair_std"
            ] = (
                0.0
                if pd.isna(pair_std)
                else pair_std
            )

    return X


# ============================================================
# RECURSIVE FORECAST
# ============================================================

def recursive_forecast(
    model,
    history_df,
    future_df
):

    history = history_df.copy()

    future = (
        future_df
        .sort_values("week")
        .copy()
    )

    all_predictions = []

    future_weeks = sorted(
        future["week"].unique()
    )

    if not future_weeks:

        return pd.DataFrame()

    first_week = future_weeks[0]

    # ========================================================
    # WEEK-BY-WEEK
    # ========================================================

    for week in future_weeks:

        current_week = future[
            future["week"] == week
        ].copy()

        # ====================================================
        # FIRST FUTURE WEEK
        # ====================================================

        if week == first_week:

            anchor_predictions = []

            for _, row in current_week.iterrows():

                center = row[
                    "center_id"
                ]

                meal = row[
                    "meal_id"
                ]

                pair_history = history[
                    (history["center_id"] == center)
                    & (history["meal_id"] == meal)
                ].sort_values(
                    "week"
                )

                if pair_history.empty:

                    meal_history = history[
                        history["meal_id"] == meal
                    ]

                    if not meal_history.empty:

                        anchor = (
                            meal_history[
                                "num_orders"
                            ].mean()
                        )

                    else:

                        anchor = (
                            history[
                                "num_orders"
                            ].mean()
                        )

                else:

                    recent_orders = (
                        pair_history[
                            "num_orders"
                        ].tail(4)
                    )

                    recent_mean = (
                        recent_orders.mean()
                    )

                    latest_actual = (
                        pair_history[
                            "num_orders"
                        ].iloc[-1]
                    )

                    pair_mean = (
                        pair_history[
                            "num_orders"
                        ].mean()
                    )

                    anchor = (
                        0.40 * recent_mean
                        + 0.40 * latest_actual
                        + 0.20 * pair_mean
                    )

                anchor_predictions.append(
                    max(
                        0,
                        float(anchor)
                    )
                )

            current_week[
                "predicted_orders"
            ] = anchor_predictions

        # ====================================================
        # FOLLOWING FUTURE WEEKS
        # ====================================================

        else:

            feature_df = (
                create_forecast_features(
                    history,
                    current_week
                )
            )

            X = feature_df[
                FINAL_FEATURES
            ].copy()

            X = fill_missing_features(
                X,
                history
            )

            # ------------------------------------------------
            # RAW MODEL PREDICTION
            # ------------------------------------------------

            pred_log = model.predict(
                X
            )

            raw_predictions = np.expm1(
                pred_log
            )

            raw_predictions = np.clip(
                raw_predictions,
                0,
                None
            )

            stabilized_predictions = []

            # ------------------------------------------------
            # STABILIZATION
            # ------------------------------------------------

            for i, (_, row) in enumerate(
                current_week.iterrows()
            ):

                center = row[
                    "center_id"
                ]

                meal = row[
                    "meal_id"
                ]

                pair_history = history[
                    (history["center_id"] == center)
                    & (history["meal_id"] == meal)
                ].sort_values(
                    "week"
                )

                if pair_history.empty:

                    stabilized_predictions.append(
                        float(
                            raw_predictions[i]
                        )
                    )

                    continue

                recent_orders = (
                    pair_history[
                        "num_orders"
                    ].tail(4)
                )

                recent_mean = (
                    recent_orders.mean()
                )

                blended_prediction = (
                    0.60 * raw_predictions[i]
                    + 0.40 * recent_mean
                )

                # Dynamic bounds
                if recent_mean > 0:

                    lower_bound = (
                        recent_mean * 0.50
                    )

                    upper_bound = (
                        recent_mean * 1.50
                    )

                    stabilized_prediction = (
                        np.clip(
                            blended_prediction,
                            lower_bound,
                            upper_bound
                        )
                    )

                else:

                    stabilized_prediction = (
                        blended_prediction
                    )

                stabilized_predictions.append(
                    float(
                        stabilized_prediction
                    )
                )

            current_week[
                "predicted_orders"
            ] = stabilized_predictions

        # ====================================================
        # SAVE PREDICTION
        # ====================================================

        all_predictions.append(
            current_week.copy()
        )

        # ====================================================
        # ADD PREDICTION TO HISTORY
        # ====================================================

        predicted_history = (
            current_week.copy()
        )

        predicted_history[
            "num_orders"
        ] = predicted_history[
            "predicted_orders"
        ]

        predicted_history = (
            predicted_history[
                history.columns
            ]
        )

        history = pd.concat(
            [
                history,
                predicted_history
            ],
            ignore_index=True
        )

    return pd.concat(
        all_predictions,
        ignore_index=True
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "Meal Demand Forecasting"
)

st.write(
    "Machine learning based demand forecasting "
    "for meal delivery and inventory planning."
)


# ============================================================
# SELECTED CONFIGURATION
# ============================================================

meal_row = meal_info[
    meal_info["meal_id"] == selected_meal
]

if not meal_row.empty:

    selected_category = (
        meal_row.iloc[0]["category"]
    )

    selected_cuisine = (
        meal_row.iloc[0]["cuisine"]
    )

else:

    selected_category = "Unknown"
    selected_cuisine = "Unknown"


st.markdown("---")

st.subheader(
    "Selected Configuration"
)

col1, col2, col3, col4 = st.columns(4)

col1.markdown(
    f"""
    <div class="config-item">
        <strong>Center:</strong>
        <code>{selected_center}</code>
    </div>
    """,
    unsafe_allow_html=True
)

col2.markdown(
    f"""
    <div class="config-item">
        <strong>Meal:</strong>
        <code>{selected_meal}</code>
    </div>
    """,
    unsafe_allow_html=True
)

col3.markdown(
    f"""
    <div class="config-item">
        <strong>Category:</strong>
        {selected_category}
    </div>
    """,
    unsafe_allow_html=True
)

col4.markdown(
    f"""
    <div class="config-item">
        <strong>Cuisine:</strong>
        {selected_cuisine}
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# GENERATE FORECAST
# ============================================================

if generate_forecast:

    selected_history = (
        train_enriched[
            (train_enriched["center_id"] ==
             selected_center)
            &
            (train_enriched["meal_id"] ==
             selected_meal)
        ]
        .sort_values("week")
        .copy()
    )

    if selected_history.empty:

        st.error(
            "No historical data found for "
            "this Center × Meal combination."
        )

        st.stop()

    # --------------------------------------------------------
    # FUTURE WEEKS
    # --------------------------------------------------------

    last_training_week = int(
        train_enriched[
            "week"
        ].max()
    )

    future_weeks = list(
        range(
            last_training_week + 1,
            last_training_week +
            forecast_horizon + 1
        )
    )

    future_df = test_enriched[
        (test_enriched["center_id"] ==
         selected_center)
        &
        (test_enriched["meal_id"] ==
         selected_meal)
        &
        (
            test_enriched["week"].isin(
                future_weeks
            )
        )
    ].copy()

    if future_df.empty:

        st.error(
            "No matching future weeks were "
            "found in test.csv."
        )

        st.stop()

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    with st.spinner(
        "Generating forecast..."
    ):

        forecast_df = recursive_forecast(
            model=model,
            history_df=train_enriched,
            future_df=future_df
        )

    # ========================================================
    # FORECAST SUMMARY
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Forecast Summary"
    )

    total_forecast = (
        forecast_df[
            "predicted_orders"
        ].sum()
    )

    average_forecast = (
        forecast_df[
            "predicted_orders"
        ].mean()
    )

    peak_demand = (
        forecast_df[
            "predicted_orders"
        ].max()
    )

    peak_week = int(
        forecast_df.loc[
            forecast_df[
                "predicted_orders"
            ].idxmax(),
            "week"
        ]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Forecast",
        f"{total_forecast:.0f}"
    )

    col2.metric(
        "Average / Week",
        f"{average_forecast:.0f}"
    )

    col3.metric(
        "Peak Demand",
        f"{peak_demand:.0f}"
    )

    col4.metric(
        "Peak Week",
        f"Week {peak_week}"
    )


    # ========================================================
    # DEMAND FORECAST
    # ========================================================

    st.subheader(
        "Demand Forecast"
    )

    forecast_chart = (
        forecast_df[
            [
                "week",
                "predicted_orders"
            ]
        ]
        .set_index("week")
    )

    forecast_chart.columns = [
        "Forecasted Orders"
    ]

    st.line_chart(
        forecast_chart
    )


    # ========================================================
    # HISTORICAL VS FORECAST
    # ========================================================

    st.subheader(
        "Historical vs Forecast"
    )

    historical_part = (
        selected_history
        .tail(20)[
            [
                "week",
                "num_orders"
            ]
        ]
        .rename(
            columns={
                "num_orders":
                "Historical Orders"
            }
        )
        .set_index("week")
    )

    forecast_part = (
        forecast_df[
            [
                "week",
                "predicted_orders"
            ]
        ]
        .rename(
            columns={
                "predicted_orders":
                "Forecasted Orders"
            }
        )
        .set_index("week")
    )

    comparison_chart = pd.concat(
        [
            historical_part,
            forecast_part
        ],
        axis=1
    )

    st.line_chart(
        comparison_chart
    )


    # ========================================================
    # RECENT DEMAND COMPARISON
    # ========================================================

    st.subheader(
        "Forecast vs Recent Demand"
    )

    recent_4_avg = (
        selected_history
        .tail(4)[
            "num_orders"
        ]
        .mean()
    )

    recent_12_avg = (
        selected_history
        .tail(12)[
            "num_orders"
        ]
        .mean()
    )

    percentage_change = (
        (
            average_forecast -
            recent_4_avg
        )
        /
        recent_4_avg
        * 100
        if recent_4_avg != 0
        else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Recent 4-Week Avg",
        f"{recent_4_avg:.0f}"
    )

    col2.metric(
        "Recent 12-Week Avg",
        f"{recent_12_avg:.0f}"
    )

    col3.metric(
        "Expected Change",
        f"{percentage_change:+.1f}%"
    )


    # ========================================================
    # PLANNING INSIGHT
    # ========================================================

    st.subheader(
        "Planning Insight"
    )

    if percentage_change > 10:

        st.info(
            f"Forecast demand is approximately "
            f"{percentage_change:.1f}% higher than "
            "the recent 4-week average. Higher "
            "preparation and staffing capacity "
            "may be required."
        )

    elif percentage_change < -10:

        st.info(
            f"Forecast demand is approximately "
            f"{abs(percentage_change):.1f}% lower than "
            "the recent 4-week average. Lower "
            "preparation volume may help reduce "
            "excess inventory."
        )

    else:

        st.info(
            "Forecast demand is relatively close "
            "to the recent 4-week average, "
            "indicating stable expected demand."
        )


    # ========================================================
    # HISTORICAL DEMAND ANALYSIS
    # ========================================================

    st.subheader(
        "Historical Demand Analysis"
    )

    historical_average = (
        selected_history[
            "num_orders"
        ]
        .mean()
    )

    historical_peak = (
        selected_history[
            "num_orders"
        ]
        .max()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Historical Average",
        f"{historical_average:.0f}"
    )

    col2.metric(
        "Recent 4 Weeks",
        f"{recent_4_avg:.0f}"
    )

    col3.metric(
        "Recent 12 Weeks",
        f"{recent_12_avg:.0f}"
    )

    col4.metric(
        "Historical Peak",
        f"{historical_peak:.0f}"
    )

    historical_chart = (
        selected_history[
            [
                "week",
                "num_orders"
            ]
        ]
        .set_index("week")
    )

    historical_chart.columns = [
        "Historical Orders"
    ]

    st.line_chart(
        historical_chart
    )


    # ========================================================
    # FORECAST DETAILS
    # ========================================================

    st.subheader(
        "Forecast Details"
    )

    forecast_table = (
        forecast_df[
            [
                "week",
                "predicted_orders"
            ]
        ]
        .copy()
    )

    forecast_table[
        "predicted_orders"
    ] = (
        forecast_table[
            "predicted_orders"
        ]
        .round(0)
        .astype(int)
    )

    forecast_table.columns = [
        "Week",
        "Forecasted Orders"
    ]

    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABOUT THE MODEL
# ============================================================

st.markdown("---")

with st.expander(
    "About the Model"
):

    st.markdown(
        """
### Final XGBoost Model

The final model uses XGBoost regression with the target
transformed using `log1p(num_orders)`.

Predictions are converted back to the original demand
scale using `expm1()`.

### Feature Groups

**Meal and Center Information**

- Center ID
- Meal ID
- Category
- Cuisine
- City code
- Region code
- Center type
- Operational area

**Pricing and Promotion**

- Checkout price
- Base price
- Email promotion
- Homepage promotion

**Historical Demand**

- Lag 1
- Lag 2
- Lag 4
- Lag 8
- Lag 10
- Lag 11
- Lag 12

**Rolling Statistics**

- 4-week rolling mean
- 4-week rolling standard deviation

**Center × Meal History**

- Historical pair mean
- Historical pair standard deviation

The model was evaluated using five expanding-window
recursive validation folds.

**Mean validation RMSLE: 0.505973**
"""
    )


# ============================================================
# MODEL DEVELOPMENT
# ============================================================

with st.expander(
    "Model Development & Experiments"
):

    experiments = pd.DataFrame(
        [
            {
                "Experiment": "Naive Baseline",
                "Mean RMSLE": 0.825242
            },
            {
                "Experiment": "Basic XGBoost",
                "Mean RMSLE": 0.542365
            },
            {
                "Experiment":
                    "Metadata-Enriched XGBoost",
                "Mean RMSLE": 0.522641
            },
            {
                "Experiment":
                    "XGBoost + Pair History",
                "Mean RMSLE": 0.507643
            },
            {
                "Experiment":
                    "XGBoost + Pair History + Long Lags",
                "Mean RMSLE": 0.505973
            },
            {
                "Experiment":
                    "Time / Trend Features",
                "Mean RMSLE": 0.527701
            },
            {
                "Experiment":
                    "Price Change Feature",
                "Mean RMSLE": 0.521148
            },
            {
                "Experiment": "LightGBM",
                "Mean RMSLE": 0.532606
            },
            {
                "Experiment": "CatBoost",
                "Mean RMSLE": 0.530950
            },
            {
                "Experiment":
                    "Reduced Feature XGBoost",
                "Mean RMSLE": 0.564261
            }
        ]
    )

    st.dataframe(
        experiments,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FORECASTING METHODOLOGY
# ============================================================

with st.expander(
    "Forecasting Methodology"
):

    st.markdown(
        """
### Forecasting Approach

The application uses a recursive forecasting strategy.

**First forecast week**

The first future week is anchored using the historical
Center × Meal demand pattern:

- 40% recent 4-week average
- 40% latest observed demand
- 20% historical Center × Meal mean

**Following weeks**

The XGBoost model generates the next forecast.

The forecast is then stabilized using the recent demand
level before being added back to the historical sequence.

This prevents an extreme prediction from immediately
propagating through every subsequent recursive week.
"""
    )


# ============================================================
# MODEL VERIFICATION
# ============================================================

with st.expander(
    "Model Verification",
    expanded=False
):

    st.write(
        "Loaded model type:",
        type(model).__name__
    )

    st.write(
        "Model feature count:",
        len(MODEL_FEATURES)
    )

    expected = set(
        FINAL_FEATURES
    )

    actual = set(
        MODEL_FEATURES
    )

    if expected == actual:

        st.success(
            "Correct 23-feature final model loaded."
        )

    else:

        missing = sorted(
            expected - actual
        )

        extra = sorted(
            actual - expected
        )

        st.warning(
            "Model feature set differs from "
            "the expected 23-feature model."
        )

        if missing:

            st.write(
                "Missing features:",
                missing
            )

        if extra:

            st.write(
                "Unexpected features:",
                extra
            )

    st.write(
        "Model features:"
    )

    st.code(
        "\n".join(MODEL_FEATURES)
    )


# ============================================================
# LIMITATIONS
# ============================================================

with st.expander(
    "Model Limitations"
):

    st.markdown(
        """
### Important Limitations

- Forecast accuracy varies across Center × Meal combinations.
- Low-demand meals are harder to predict.
- Recursive forecasting can propagate prediction errors.
- Missing historical weeks are not automatically treated as
  zero demand.
- Validation RMSLE represents historical validation performance
  and does not guarantee identical performance on future data.
- The dashboard should be treated as a demand-planning aid,
  not an exact order-count prediction system.
"""
    )