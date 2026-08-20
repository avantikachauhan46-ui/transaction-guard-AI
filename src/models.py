import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def run_dbscan(X_scaled, eps=1.8, min_samples=8):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    preds = np.where(labels == -1, 1, 0)
    return preds

def run_isolation_forest(X_scaled, contamination=0.03, random_state=42):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X_scaled)
    preds = np.where(labels == -1, 1, 0)
    scores = -model.decision_function(X_scaled)
    return preds, scores

def run_lof(X_scaled, n_neighbors=20, contamination=0.03):
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = model.fit_predict(X_scaled)
    preds = np.where(labels == -1, 1, 0)
    scores = -model.negative_outlier_factor_
    return preds, scores
