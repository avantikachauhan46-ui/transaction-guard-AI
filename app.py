import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import load_all_datasets
from src.feature_engineering import engineer_features
from src.preprocessing import prepare_and_scale_features
from src.models import run_dbscan, run_isolation_forest, run_lof
from src.visualization import plot_pca_anomalies, plot_hourly_distribution

st.set_page_config(page_title="TransactionGuard AI", layout="wide", page_icon="💳")

st.title("💳 TransactionGuard AI: Enterprise Anomaly Detection")
st.markdown("Multi-source transaction monitoring, feature engineering, and unsupervised ML anomaly detection engine.")

@st.cache_data
def get_processed_data():
    tx_df, cust_df, merch_df = load_all_datasets()
    merged_df = engineer_features(tx_df, cust_df, merch_df)
    processed_df, X_scaled = prepare_and_scale_features(merged_df)
    return processed_df, X_scaled

df, X_scaled = get_processed_data()

# Sidebar Controls
st.sidebar.header("⚙️ Algorithm Configuration")
algo_choice = st.sidebar.selectbox("Select Model", ["Isolation Forest", "DBSCAN", "Local Outlier Factor (LOF)"])

if algo_choice == "Isolation Forest":
    contamination = st.sidebar.slider("Contamination Rate", min_value=0.005, max_value=0.10, value=0.03, step=0.005)
    preds, scores = run_isolation_forest(X_scaled, contamination=contamination)
    df["anomaly_score"] = scores
elif algo_choice == "DBSCAN":
    eps = st.sidebar.slider("Epsilon (eps)", min_value=0.5, max_value=4.0, value=1.8, step=0.1)
    min_samples = st.sidebar.slider("Min Samples", min_value=3, max_value=25, value=8, step=1)
    preds = run_dbscan(X_scaled, eps=eps, min_samples=min_samples)
    df["anomaly_score"] = np.nan
elif algo_choice == "Local Outlier Factor (LOF)":
    n_neighbors = st.sidebar.slider("Number of Neighbors", min_value=5, max_value=60, value=20, step=1)
    contamination = st.sidebar.slider("Contamination Rate", min_value=0.005, max_value=0.10, value=0.03, step=0.005)
    preds, scores = run_lof(X_scaled, n_neighbors=n_neighbors, contamination=contamination)
    df["anomaly_score"] = scores

df["anomaly_pred"] = preds
df["selected_algorithm"] = algo_choice

# Metric Banner
total_tx = len(df)
total_anomalies = int(df["anomaly_pred"].sum())
anomaly_rate = (total_anomalies / total_tx) * 100
total_normal = total_tx - total_anomalies

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total_tx:,}")
col2.metric("Normal Transactions", f"{total_normal:,}")
col3.metric("Detected Anomalies", f"{total_anomalies:,}", delta=f"{anomaly_rate:.2f}%", delta_color="inverse")
col4.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "🧪 Algorithm Benchmark", "🔎 Transaction Explorer"])

with tab1:
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.plotly_chart(plot_pca_anomalies(df, "anomaly_pred", f"PCA Space Separation ({algo_choice})"), use_container_width=True)
    with vcol2:
        st.plotly_chart(plot_hourly_distribution(df, "anomaly_pred"), use_container_width=True)

with tab2:
    st.subheader("Cross-Model Comparison")
    dbscan_preds = run_dbscan(X_scaled, eps=1.8, min_samples=8)
    if_preds, _ = run_isolation_forest(X_scaled, contamination=0.03)
    lof_preds, _ = run_lof(X_scaled, n_neighbors=20, contamination=0.03)

    comp_data = {
        "Algorithm": ["DBSCAN", "Isolation Forest", "Local Outlier Factor (LOF)"],
        "Anomalies Flagged": [int(dbscan_preds.sum()), int(if_preds.sum()), int(lof_preds.sum())],
        "Anomaly Rate (%)": [
            round((dbscan_preds.sum() / total_tx) * 100, 2),
            round((if_preds.sum() / total_tx) * 100, 2),
            round((lof_preds.sum() / total_tx) * 100, 2)
        ],
        "Primary Trigger Focus": [
            "Dense clusters vs extreme isolated points",
            "Multi-dimensional global partition depth",
            "Local density deviation from nearest neighbors"
        ]
    }
    st.table(pd.DataFrame(comp_data))

with tab3:
    st.subheader("Filter & Export Results")
    search_id = st.text_input("Search by Transaction ID (e.g., TXN100001)")
    
    display_cols = [
        "transaction_id", "customer_id", "merchant_name", "amount",
        "avg_transaction_amount", "amount_to_avg_ratio", "transaction_hour",
        "location", "city", "merchant_risk_score", "anomaly_pred"
    ]
    
    filtered_df = df[display_cols].copy()
    if search_id:
        filtered_df = filtered_df[filtered_df["transaction_id"].str.contains(search_id.strip(), case=False)]

    st.dataframe(filtered_df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Prediction CSV",
        data=csv_data,
        file_name="anomaly_results.csv",
        mime="text/csv"
    )
