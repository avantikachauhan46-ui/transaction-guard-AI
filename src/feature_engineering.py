import numpy as np
import pandas as pd

def engineer_features(tx_df, cust_df, merch_df):
    df = tx_df.merge(cust_df, on="customer_id", how="left", suffixes=("", "_cust"))
    df = df.merge(merch_df, on="merchant_id", how="left", suffixes=("", "_merch"))

    df["amount_to_avg_ratio"] = df["amount"] / (df["avg_transaction_amount"] + 1e-5)
    df["amount_diff_from_avg"] = df["amount"] - df["avg_transaction_amount"]
    df["is_night_transaction"] = df["transaction_hour"].apply(lambda h: 1 if h in [0, 1, 2, 3, 4, 5] else 0)
    df["is_foreign_location"] = (df["location"] != df["city"]).astype(int)
    
    risk_mapping = {"Low": 1, "Medium": 2, "High": 3}
    df["merchant_risk_score"] = df["risk_level"].map(risk_mapping).fillna(1)

    velocity_map = df.groupby("customer_id")["transaction_id"].transform("count")
    df["customer_tx_density"] = velocity_map / (df["transactions_per_month"] + 1e-5)

    return df
