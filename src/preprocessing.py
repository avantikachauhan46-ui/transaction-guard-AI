from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

FEATURE_COLUMNS = [
    "amount",
    "transaction_hour",
    "amount_to_avg_ratio",
    "amount_diff_from_avg",
    "is_night_transaction",
    "is_foreign_location",
    "merchant_risk_score",
    "customer_tx_density"
]

def prepare_and_scale_features(df):
    X = df[FEATURE_COLUMNS].copy()
    X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    df["pca_1"] = X_pca[:, 0]
    df["pca_2"] = X_pca[:, 1]
    
    return df, X_scaled
