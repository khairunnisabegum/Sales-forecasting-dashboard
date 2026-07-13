"""
Streamlit Dashboard — End-to-End Sales Forecasting & Demand Intelligence System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence",
                    layout="wide")

# ============================================================
# SHARED DATA LOADING (cached — this is the one thing we don't
# want to re-read from disk on every widget interaction)
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df

df = load_data()

# ============================================================
# SHARED HELPER FUNCTIONS (used across multiple pages)
# ============================================================
def monthly_series_for(subset_df):
    """Aggregate a filtered slice of row-level data into monthly totals."""
    return subset_df.set_index("Order Date").sort_index()["Sales"].resample("ME").sum()


def build_lag_features(series):
    """Lag-feature engineering — identical logic to the notebook's Task 3/4
    XGBoost pipeline, so dashboard forecasts stay consistent with the report."""
    ml = series.reset_index()
    ml.columns = ["Month", "Sales"]
    ml["Lag1"] = ml["Sales"].shift(1)
    ml["Lag2"] = ml["Sales"].shift(2)
    ml["Lag3"] = ml["Sales"].shift(3)
    ml["RollingMean3"] = ml["Sales"].shift(1).rolling(window=3).mean()
    ml["MonthNum"] = ml["Month"].dt.month
    ml["Quarter"] = ml["Month"].dt.quarter
    ml["SeasonNum"] = ml["MonthNum"].apply(lambda m: (m % 12) // 3)
    return ml.dropna().reset_index(drop=True)


FEATURE_COLS = ["Lag1", "Lag2", "Lag3", "RollingMean3", "MonthNum", "Quarter", "SeasonNum"]


def evaluate_holdout(series, holdout_months=3):
    """Train on all but the last `holdout_months`, predict those known
    months, and return MAE/RMSE — this is what Page 2 shows below the chart."""
    ml = build_lag_features(series)
    if len(ml) <= holdout_months + 5:
        return None, None  # not enough data for a meaningful holdout
    train_ml, test_ml = ml.iloc[:-holdout_months], ml.iloc[-holdout_months:]
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(train_ml[FEATURE_COLS], train_ml["Sales"])
    preds = model.predict(test_ml[FEATURE_COLS])
    mae = mean_absolute_error(test_ml["Sales"], preds)
    rmse = np.sqrt(mean_squared_error(test_ml["Sales"], preds))
    return mae, rmse


def recursive_forecast(series, periods=3):
    """Train on the FULL series, then forecast `periods` months beyond the
    end of the data, feeding each prediction back in as the next Lag1
    (real future values don't exist yet, so this is required for genuine
    future forecasting, unlike the known-holdout evaluation above)."""
    ml = build_lag_features(series)
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(ml[FEATURE_COLS], ml["Sales"])

    history = list(series.values)
    last_date = series.index[-1]
    forecast_values, forecast_dates = [], []

    for _ in range(periods):
        next_date = last_date + pd.offsets.MonthEnd(1)
        lag1, lag2, lag3 = history[-1], history[-2], history[-3]
        rolling_mean3 = np.mean(history[-3:])
        row = pd.DataFrame([[lag1, lag2, lag3, rolling_mean3,
                              next_date.month, next_date.quarter, (next_date.month % 12) // 3]],
                            columns=FEATURE_COLS)
        pred = model.predict(row)[0]
        forecast_values.append(pred)
        forecast_dates.append(next_date)
        history.append(pred)
        last_date = next_date

    return pd.Series(forecast_values, index=pd.DatetimeIndex(forecast_dates))


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("📊 Sales Intelligence Dashboard")
page = st.sidebar.radio(
    "Navigate to:",
    ["1. Sales Overview", "2. Forecast Explorer", "3. Anomaly Report", "4. Product Demand Segments"]
)
st.sidebar.markdown("---")
# ============================================================
# PAGE 1 — SALES OVERVIEW DASHBOARD
# ============================================================
if page == "1. Sales Overview":
    st.title("Sales Overview Dashboard")

    # --- Total sales by year (bar chart) ---
    st.subheader("Total Sales by Year")
    yearly_sales = df.groupby("Year")["Sales"].sum().reset_index()
    fig_yearly = px.bar(yearly_sales, x="Year", y="Sales", text_auto=".2s",
                         color="Sales", color_continuous_scale="Blues")
    fig_yearly.update_layout(yaxis_title="Total Sales ($)", showlegend=False)
    st.plotly_chart(fig_yearly, use_container_width=True)

    # --- Monthly sales trend line chart ---
    st.subheader("Monthly Sales Trend")
    monthly_all = monthly_series_for(df).reset_index()
    monthly_all.columns = ["Month", "Sales"]
    fig_monthly = px.line(monthly_all, x="Month", y="Sales", markers=True)
    fig_monthly.update_layout(yaxis_title="Total Sales ($)")
    st.plotly_chart(fig_monthly, use_container_width=True)

    # --- Sales by region and category, with interactive filters ---
    st.subheader("Sales by Region & Category")
    col1, col2 = st.columns(2)
    with col1:
        selected_regions = st.multiselect("Filter by Region", options=sorted(df["Region"].unique()),
                                           default=sorted(df["Region"].unique()))
    with col2:
        selected_categories = st.multiselect("Filter by Category", options=sorted(df["Category"].unique()),
                                              default=sorted(df["Category"].unique()))

    filtered = df[df["Region"].isin(selected_regions) & df["Category"].isin(selected_categories)]

    if filtered.empty:
        st.warning("No data for the selected filters — pick at least one Region and Category.")
    else:
        region_cat_sales = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
        fig_region_cat = px.bar(region_cat_sales, x="Region", y="Sales", color="Category",
                                 barmode="group", text_auto=".2s")
        fig_region_cat.update_layout(yaxis_title="Total Sales ($)")
        st.plotly_chart(fig_region_cat, use_container_width=True)

        st.metric("Total Sales (filtered selection)", f"${filtered['Sales'].sum():,.0f}")

# ============================================================
# PAGE 2 — FORECAST EXPLORER
# ============================================================
elif page == "2. Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Forecasts are generated live with XGBoost — the model that won "
               "the Task 3 comparison (lowest MAE/RMSE/MAPE) — trained fresh "
               "for whichever segment you pick below.")

    col1, col2 = st.columns(2)
    with col1:
        dimension = st.selectbox("Select dimension", ["Category", "Region"])
    with col2:
        options = sorted(df[dimension].unique())
        selection = st.selectbox(f"Select {dimension}", options)

    horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3)

    segment_df = df[df[dimension] == selection]
    segment_monthly = monthly_series_for(segment_df)

    if len(segment_monthly) < 10:
        st.error(f"Not enough monthly history for '{selection}' to build a reliable forecast "
                 f"(only {len(segment_monthly)} months available).")
    else:
        with st.spinner(f"Training XGBoost on {selection} history..."):
            forecast = recursive_forecast(segment_monthly, periods=horizon)
            mae, rmse = evaluate_holdout(segment_monthly, holdout_months=min(3, len(segment_monthly) // 4))

        # --- Chart: history + forecast ---
        fig = go.Figure()
        recent_history = segment_monthly.tail(18)
        fig.add_trace(go.Scatter(x=recent_history.index, y=recent_history.values,
                                  mode="lines+markers", name="Actual (recent history)",
                                  line=dict(color="steelblue")))
        connector_x = [recent_history.index[-1]] + list(forecast.index)
        connector_y = [recent_history.values[-1]] + list(forecast.values)
        fig.add_trace(go.Scatter(x=connector_x, y=connector_y,
                                  mode="lines+markers", name=f"XGBoost Forecast ({horizon}mo)",
                                  line=dict(color="firebrick", dash="dash")))
        fig.update_layout(title=f"{dimension}: {selection} — Sales Forecast",
                           yaxis_title="Sales ($)", xaxis_title="Month")
        st.plotly_chart(fig, use_container_width=True)

        # --- Forecast values table ---
        forecast_table = pd.DataFrame({
            "Month": forecast.index.strftime("%b %Y"),
            "Forecasted Sales": forecast.values.round(2)
        })
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)

        # --- MAE / RMSE below the chart, as required ---
        st.subheader("Model Accuracy (holdout evaluation)")
        if mae is not None:
            m1, m2 = st.columns(2)
            m1.metric("MAE", f"${mae:,.2f}")
            m2.metric("RMSE", f"${rmse:,.2f}")
            st.caption("Computed by holding out the last few known months, training on everything "
                       "before them, and comparing predictions against the real values — the same "
                       "method used for the Task 3 model comparison table.")
        else:
            st.info("Not enough history in this segment to compute a reliable holdout accuracy score.")

# ============================================================
# PAGE 3 — ANOMALY REPORT
# ============================================================
elif page == "3. Anomaly Report":
    st.title("Anomaly Report")
    st.caption("Weekly sales checked against two independent methods: Isolation Forest "
               "(global outlier detection) and Z-Score (local, rolling-baseline detection).")

    weekly_series = df.set_index("Order Date").sort_index()["Sales"].resample("W").sum()

    # --- Isolation Forest ---
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_preds = iso_forest.fit_predict(weekly_series.values.reshape(-1, 1))

    weekly_df = weekly_series.reset_index()
    weekly_df.columns = ["Week", "Sales"]
    weekly_df["IF_Anomaly"] = (iso_preds == -1)

    # --- Z-Score (current point excluded from its own baseline — see notebook note) ---
    window = 8
    rolling_mean = weekly_series.shift(1).rolling(window, min_periods=4).mean()
    rolling_std = weekly_series.shift(1).rolling(window, min_periods=4).std()
    z_scores = (weekly_series - rolling_mean) / rolling_std
    weekly_df["Z_Score"] = z_scores.values
    weekly_df["Z_Anomaly"] = weekly_df["Z_Score"].abs() > 2

    method = st.radio("Show anomalies detected by:", ["Isolation Forest", "Z-Score", "Both (flagged by either)"],
                       horizontal=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_df["Week"], y=weekly_df["Sales"],
                              mode="lines", name="Weekly Sales", line=dict(color="steelblue")))

    if method == "Isolation Forest":
        anomalies = weekly_df[weekly_df["IF_Anomaly"]]
        marker_color = "red"
    elif method == "Z-Score":
        anomalies = weekly_df[weekly_df["Z_Anomaly"]]
        marker_color = "orange"
    else:
        anomalies = weekly_df[weekly_df["IF_Anomaly"] | weekly_df["Z_Anomaly"]]
        marker_color = "purple"

    fig.add_trace(go.Scatter(x=anomalies["Week"], y=anomalies["Sales"], mode="markers",
                              name="Anomaly", marker=dict(color=marker_color, size=12, symbol="x")))
    fig.update_layout(title="Weekly Sales with Detected Anomalies", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomaly Weeks")
    st.dataframe(
        anomalies[["Week", "Sales"]].assign(Sales=lambda d: d["Sales"].round(2)),
        use_container_width=True, hide_index=True
    )

    both_count = (weekly_df["IF_Anomaly"] & weekly_df["Z_Anomaly"]).sum()
    st.caption(f"{both_count} week(s) were flagged by BOTH methods — these are the "
               f"highest-confidence anomalies.")

# ============================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ============================================================
elif page == "4. Product Demand Segments":
    st.title("Product Demand Segments")
    st.caption("Sub-categories clustered by total volume, average order value, "
               "volatility, and YoY growth rate — same methodology as the notebook's Task 6.")

    # --- Rebuild features (identical logic to the notebook) ---
    subcategories = df["Sub-Category"].unique()
    rows = []
    for sc in subcategories:
        sub = df[df["Sub-Category"] == sc]
        total_sales = sub["Sales"].sum()
        avg_order_value = sub["Sales"].mean()
        monthly = sub.set_index("Order Date").resample("ME")["Sales"].sum()
        volatility = monthly.std()
        yearly = sub.groupby(sub["Order Date"].dt.year)["Sales"].sum().sort_index()
        yoy_growth = yearly.pct_change().dropna()
        growth_rate = yoy_growth.mean() if len(yoy_growth) > 0 else 0
        rows.append({"Sub-Category": sc, "Total Sales": total_sales,
                      "Avg Order Value": avg_order_value, "Volatility": volatility,
                      "Growth Rate": growth_rate})

    seg_df = pd.DataFrame(rows)
    feature_cols = ["Total Sales", "Avg Order Value", "Volatility", "Growth Rate"]
    X_scaled = StandardScaler().fit_transform(seg_df[feature_cols])

    n_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=6, value=4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    seg_df["Cluster"] = kmeans.fit_predict(X_scaled)

    centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)

    def label_cluster(row):
        if row["Growth Rate"] > 1.0 and row["Growth Rate"] == centroids_scaled["Growth Rate"].max():
            return "Growing Demand"
        if row["Growth Rate"] < -1.0 and row["Growth Rate"] == centroids_scaled["Growth Rate"].min():
            return "Declining Demand"
        if row["Total Sales"] > 0 and row["Volatility"] <= 0:
            return "High Volume, Stable Demand"
        if row["Total Sales"] <= 0 and row["Volatility"] > 0:
            return "Low Volume, High Volatility"
        if row["Total Sales"] > 0 and row["Volatility"] > 0:
            return "High Volume, Volatile Demand"
        return "Low Volume, Stable Demand"

    cluster_labels = {i: label_cluster(centroids_scaled.loc[i]) for i in centroids_scaled.index}
    seg_df["Cluster Label"] = seg_df["Cluster"].map(cluster_labels)

    # --- PCA scatter plot ---
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    seg_df["PC1"], seg_df["PC2"] = coords[:, 0], coords[:, 1]

    fig = px.scatter(seg_df, x="PC1", y="PC2", color="Cluster Label", text="Sub-Category",
                      size=[20] * len(seg_df), size_max=15)
    fig.update_traces(textposition="top center")
    fig.update_layout(title="Product Sub-Category Demand Segments (PCA-reduced)")
    st.plotly_chart(fig, use_container_width=True)

    # --- Table: which sub-categories belong to which cluster ---
    st.subheader("Sub-Category → Cluster Assignment")
    display_table = seg_df[["Sub-Category", "Cluster Label", "Total Sales", "Avg Order Value",
                             "Volatility", "Growth Rate"]].copy()
    display_table[["Total Sales", "Avg Order Value", "Volatility"]] = \
        display_table[["Total Sales", "Avg Order Value", "Volatility"]].round(2)
    display_table["Growth Rate"] = (display_table["Growth Rate"] * 100).round(1).astype(str) + "%"
    st.dataframe(display_table.sort_values("Cluster Label"), use_container_width=True, hide_index=True)
