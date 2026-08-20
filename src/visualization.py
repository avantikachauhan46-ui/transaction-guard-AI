import plotly.express as px

def plot_pca_anomalies(df, pred_col="anomaly_pred", title="PCA Projection: Normal vs Anomaly"):
    plot_df = df.copy()
    plot_df["Status"] = plot_df[pred_col].map({0: "Normal", 1: "Anomaly"})
    
    fig = px.scatter(
        plot_df,
        x="pca_1",
        y="pca_2",
        color="Status",
        color_discrete_map={"Normal": "#2ecc71", "Anomaly": "#e74c3c"},
        hover_data=["transaction_id", "amount", "transaction_hour", "location"],
        title=title,
        opacity=0.75
    )
    fig.update_layout(template="plotly_white", legend_title_text="Classification")
    return fig

def plot_hourly_distribution(df, pred_col="anomaly_pred"):
    hourly_df = df.groupby(["transaction_hour", pred_col]).size().reset_index(name="count")
    hourly_df["Status"] = hourly_df[pred_col].map({0: "Normal", 1: "Anomaly"})
    
    fig = px.bar(
        hourly_df,
        x="transaction_hour",
        y="count",
        color="Status",
        barmode="group",
        color_discrete_map={"Normal": "#3498db", "Anomaly": "#e74c3c"},
        title="Transaction Volume by Hour (Normal vs Anomaly)",
        labels={"transaction_hour": "Hour of Day (0-23)", "count": "Volume"}
    )
    fig.update_layout(template="plotly_white")
    return fig
