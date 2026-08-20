import os
import pandas as pd

def load_all_datasets(raw_dir="data/raw"):
    t_path = os.path.join(raw_dir, "transactions.csv")
    c_path = os.path.join(raw_dir, "customers.csv")
    m_path = os.path.join(raw_dir, "merchants.csv")

    tx_df = pd.read_csv(t_path)
    cust_df = pd.read_csv(c_path)
    merch_df = pd.read_csv(m_path)
    return tx_df, cust_df, merch_df
