"""
Stage 2: K-Means segmentation into 5 behavioural cohorts using engineered
engagement + risk features, plus a supervised churn-prediction model for
risk scoring validation and playbook targeting.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.decomposition import PCA

import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data")
df = pd.read_csv(f"{OUT}/telco_features.csv")

# ---------------- Clustering ----------------
cluster_features = [
    'EngagementScore', 'RiskScore', 'tenure', 'MonthlyCharges',
    'ServicesAdopted', 'ContractScore'
]
X = df[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
df['Cohort'] = kmeans.fit_predict(X_scaled)

# Order cohorts by churn rate (so naming is consistent/interpretable) and label them
cohort_stats = df.groupby('Cohort').agg(
    Size=('Cohort', 'size'),
    ChurnRate=('ChurnFlag', 'mean'),
    AvgEngagement=('EngagementScore', 'mean'),
    AvgRisk=('RiskScore', 'mean'),
    AvgTenure=('tenure', 'mean'),
    AvgMonthlyCharges=('MonthlyCharges', 'mean'),
    AvgServices=('ServicesAdopted', 'mean'),
).sort_values('ChurnRate', ascending=False)

print("Raw cluster stats (sorted by churn rate):")
print(cohort_stats)

# Assign business-friendly names based on each cluster's actual profile
# (churn rate, engagement, tenure, spend, services) rather than a fixed rank order.
def label_cohort(row):
    if row['ChurnRate'] > 0.5:
        return "At-Risk Newcomers"          # very high churn, short tenure, M2M, low engagement
    if row['ChurnRate'] > 0.25 and row['AvgMonthlyCharges'] > 60:
        return "Wavering High-Spenders"      # decent tenure/services but high bills, still churning
    if row['ChurnRate'] > 0.25:
        return "Disengaged Low-Spenders"     # low engagement, low services, short tenure, low spend
    if row['AvgMonthlyCharges'] > 60:
        return "High-Value Champions"        # low churn, high engagement, high spend, long tenure
    return "Contract-Locked Loyalists"       # low churn, long tenure, low spend, low services

names = {cid: label_cohort(cohort_stats.loc[cid]) for cid in cohort_stats.index}

df['CohortName'] = df['Cohort'].map(names)

cohort_summary = df.groupby('CohortName').agg(
    CustomerCount=('CohortName', 'size'),
    ChurnRatePct=('ChurnFlag', lambda x: round(x.mean() * 100, 1)),
    AvgEngagementScore=('EngagementScore', lambda x: round(x.mean(), 1)),
    AvgRiskScore=('RiskScore', lambda x: round(x.mean(), 1)),
    AvgTenureMonths=('tenure', lambda x: round(x.mean(), 1)),
    AvgMonthlyCharges=('MonthlyCharges', lambda x: round(x.mean(), 2)),
    AvgServicesAdopted=('ServicesAdopted', lambda x: round(x.mean(), 1)),
    PctMonthToMonth=('Contract', lambda x: round((x == 'Month-to-month').mean() * 100, 1)),
).sort_values('ChurnRatePct', ascending=False)

cohort_summary['PctOfBase'] = round(cohort_summary['CustomerCount'] / len(df) * 100, 1)
print("\n=== 5 Behavioural Cohorts ===")
print(cohort_summary)

cohort_summary.to_csv(f"{OUT}/cohort_summary.csv")

# ---------------- PCA for 2D visualization ----------------
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df['PC1'] = coords[:, 0]
df['PC2'] = coords[:, 1]
print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

# ---------------- Supervised churn prediction model ----------------
model_features = [
    'EngagementScore', 'RiskScore', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'ServicesAdopted', 'ContractScore', 'AutoPay', 'PaperlessScore',
    'MonthToMonthRisk', 'EcheckRisk', 'FiberNoSecurityRisk', 'UnanchoredRisk', 'SeniorCitizen'
]
Xm = df[model_features]
ym = df['ChurnFlag']
X_train, X_test, y_train, y_test = train_test_split(Xm, ym, test_size=0.25, random_state=42, stratify=ym)

rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=15, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
proba = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print(f"\nRandomForest churn model AUC: {auc:.3f}")
print(classification_report(y_test, rf.predict(X_test)))

importances = pd.Series(rf.feature_importances_, index=model_features).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)
importances.to_csv(f"{OUT}/feature_importance.csv", header=['Importance'])

# Score entire base with the model (predicted churn probability) for the playbook
df['PredictedChurnProbability'] = rf.predict_proba(Xm)[:, 1]

df.to_csv(f"{OUT}/telco_scored.csv", index=False)
print(f"\nSaved -> {OUT}/telco_scored.csv, cohort_summary.csv, feature_importance.csv")
