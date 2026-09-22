"""
Customer Churn Intelligence Platform
Stage 1: Data cleaning, feature engineering, engagement/risk scoring
Dataset: IBM Telco Customer Churn (7,043 customers)
"""
import os
import urllib.request
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "telco.csv")
OUT = os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

if not os.path.exists(RAW):
    print(f"Dataset not found locally — downloading from {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, RAW)
    print(f"Saved -> {RAW}")

df = pd.read_csv(RAW)

# ---------- Cleaning ----------
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
df['ChurnFlag'] = (df['Churn'] == 'Yes').astype(int)

churn_rate = df['ChurnFlag'].mean()
print(f"Overall churn rate: {churn_rate:.1%}  (n={len(df)})")

# ---------- Service adoption count (drives 'Engagement') ----------
service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

def count_services(row):
    n = 0
    for c in service_cols:
        if row[c] == 'Yes':
            n += 1
    return n

df['ServicesAdopted'] = df.apply(count_services, axis=1)
df['HasInternet'] = (df['InternetService'] != 'No').astype(int)

# ---------- Engagement Score (0-100) ----------
# Weighted blend: service adoption breadth, tenure depth, autopay/paperless behavior,
# and contract commitment - all proxies for how "sticky"/engaged a customer is.
df['TenureNorm'] = (df['tenure'] / df['tenure'].max())
df['ServicesNorm'] = df['ServicesAdopted'] / len(service_cols)
df['ContractScore'] = df['Contract'].map({'Month-to-month': 0.0, 'One year': 0.5, 'Two year': 1.0})
df['AutoPay'] = df['PaymentMethod'].isin(['Bank transfer (automatic)', 'Credit card (automatic)']).astype(int)
df['PaperlessScore'] = (df['PaperlessBilling'] == 'Yes').astype(int)

df['EngagementScore'] = (
    0.35 * df['TenureNorm'] +
    0.30 * df['ServicesNorm'] +
    0.20 * df['ContractScore'] +
    0.10 * df['AutoPay'] +
    0.05 * df['PaperlessScore']
) * 100

# ---------- Risk Score (0-100) ----------
# Weighted blend of known churn drivers: month-to-month contract, low tenure,
# high monthly charge relative to services received, electronic check payment,
# fiber-only w/o security add-ons, no partner/dependents (less "anchored").
df['ChargeToServiceRatio'] = df['MonthlyCharges'] / (df['ServicesAdopted'] + 1)
df['ChargeRiskNorm'] = (df['ChargeToServiceRatio'] - df['ChargeToServiceRatio'].min()) / \
                        (df['ChargeToServiceRatio'].max() - df['ChargeToServiceRatio'].min())
df['LowTenureRisk'] = 1 - df['TenureNorm']
df['MonthToMonthRisk'] = (df['Contract'] == 'Month-to-month').astype(int)
df['EcheckRisk'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
df['FiberNoSecurityRisk'] = ((df['InternetService'] == 'Fiber optic') &
                              (df['OnlineSecurity'] != 'Yes') &
                              (df['TechSupport'] != 'Yes')).astype(int)
df['UnanchoredRisk'] = ((df['Partner'] == 'No') & (df['Dependents'] == 'No')).astype(int)

df['RiskScore'] = (
    0.25 * df['MonthToMonthRisk'] +
    0.20 * df['LowTenureRisk'] +
    0.20 * df['FiberNoSecurityRisk'] +
    0.15 * df['ChargeRiskNorm'] +
    0.10 * df['EcheckRisk'] +
    0.10 * df['UnanchoredRisk']
) * 100

print(f"\nEngagement Score stats:\n{df['EngagementScore'].describe()}")
print(f"\nRisk Score stats:\n{df['RiskScore'].describe()}")

# Sanity check: risk score should correlate positively with actual churn
corr = df[['RiskScore', 'ChurnFlag']].corr().iloc[0, 1]
print(f"\nRiskScore vs actual churn correlation: {corr:.3f}")
eng_corr = df[['EngagementScore', 'ChurnFlag']].corr().iloc[0, 1]
print(f"EngagementScore vs actual churn correlation: {eng_corr:.3f}")

df.to_csv(f"{OUT}/telco_features.csv", index=False)
print(f"\nSaved -> {OUT}/telco_features.csv  ({df.shape[0]} rows, {df.shape[1]} cols)")
