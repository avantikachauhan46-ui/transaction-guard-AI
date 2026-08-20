# 💳 TransactionGuard AI: Intelligent Anomaly Detection System

TransactionGuard AI is an end-to-end unsupervised ML platform that integrates multiple raw data sources, performs domain-specific feature engineering, trains unsupervised anomaly detection models (DBSCAN, Isolation Forest, LOF), reduces dimensions using PCA, and serves an interactive Streamlit analytics dashboard.

---

## 📁 Repository Structure
```
TransactionGuard-AI/
├── data/
│   ├── raw/
│   │   ├── transactions.csv
│   │   ├── customers.csv
│   │   └── merchants.csv
│   ├── processed/
│   └── predictions/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── visualization.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Extract & Open in VS Code
Extract the downloaded zip file and open the `TransactionGuard-AI` folder in VS Code.

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Streamlit Dashboard
```bash
streamlit run app.py
```
