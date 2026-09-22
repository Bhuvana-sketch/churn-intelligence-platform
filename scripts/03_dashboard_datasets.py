"""
Stage 3: Build the supporting cross-tab datasets that a 5-page Power BI
dashboard would be built on top of (one clean fact table + pre-aggregated
summary tables for fast page load / simple DAX).
"""
import pandas as pd

import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data")
df = pd.read_csv(f"{OUT}/telco_scored.csv")

# Tenure buckets for the Overview/Trends page
bins = [0, 6, 12, 24, 48, 72]
labels = ['0-6 mo', '7-12 mo', '13-24 mo', '25-48 mo', '49-72 mo']
df['TenureBucket'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)

# --- Page 1: Executive Overview KPIs ---
kpis = pd.DataFrame([{
    'TotalCustomers': len(df),
    'ChurnedCustomers': int(df['ChurnFlag'].sum()),
    'ChurnRate': round(df['ChurnFlag'].mean() * 100, 1),
    'AvgMonthlyRevenue': round(df['MonthlyCharges'].mean(), 2),
    'MonthlyRevenueAtRisk': round((df.loc[df['ChurnFlag'] == 1, 'MonthlyCharges']).sum(), 2),
    'AvgTenureMonths': round(df['tenure'].mean(), 1),
    'AvgEngagementScore': round(df['EngagementScore'].mean(), 1),
    'AvgRiskScore': round(df['RiskScore'].mean(), 1),
}])
kpis.to_csv(f"{OUT}/pg1_kpis.csv", index=False)

# --- Page 2: Cohort Explorer (already have cohort_summary.csv) ---
cohort_churn_by_tenure = df.groupby(['CohortName', 'TenureBucket'], observed=True)['ChurnFlag'].mean().reset_index()
cohort_churn_by_tenure.to_csv(f"{OUT}/pg2_cohort_by_tenure.csv", index=False)

# --- Page 3: Churn Drivers ---
by_contract = df.groupby('Contract')['ChurnFlag'].agg(['mean', 'count']).reset_index()
by_contract.columns = ['Contract', 'ChurnRate', 'Customers']
by_internet = df.groupby('InternetService')['ChurnFlag'].agg(['mean', 'count']).reset_index()
by_internet.columns = ['InternetService', 'ChurnRate', 'Customers']
by_payment = df.groupby('PaymentMethod')['ChurnFlag'].agg(['mean', 'count']).reset_index()
by_payment.columns = ['PaymentMethod', 'ChurnRate', 'Customers']
by_contract.to_csv(f"{OUT}/pg3_by_contract.csv", index=False)
by_internet.to_csv(f"{OUT}/pg3_by_internet.csv", index=False)
by_payment.to_csv(f"{OUT}/pg3_by_payment.csv", index=False)

feat_imp = pd.read_csv(f"{OUT}/feature_importance.csv")
feat_imp.to_csv(f"{OUT}/pg3_feature_importance.csv", index=False)

# --- Page 4: Revenue at Risk ---
revenue_at_risk_by_cohort = df.groupby('CohortName').apply(
    lambda g: pd.Series({
        'Customers': len(g),
        'ChurnedCustomers': int(g['ChurnFlag'].sum()),
        'MonthlyRevenue': round(g['MonthlyCharges'].sum(), 2),
        'MonthlyRevenueAtRisk': round(g.loc[g['ChurnFlag'] == 1, 'MonthlyCharges'].sum(), 2),
        'AnnualRevenueAtRisk': round(g.loc[g['ChurnFlag'] == 1, 'MonthlyCharges'].sum() * 12, 2),
    }), include_groups=False
).reset_index().sort_values('MonthlyRevenueAtRisk', ascending=False)
revenue_at_risk_by_cohort.to_csv(f"{OUT}/pg4_revenue_at_risk.csv", index=False)

# --- Page 5: High-Risk Customer Watchlist (top 100 by predicted probability, still active-looking) ---
watchlist_cols = ['customerID', 'CohortName', 'PredictedChurnProbability', 'RiskScore',
                   'EngagementScore', 'tenure', 'MonthlyCharges', 'Contract', 'InternetService',
                   'PaymentMethod']
watchlist = df[df['ChurnFlag'] == 0].sort_values('PredictedChurnProbability', ascending=False)[watchlist_cols].head(100)
watchlist['PredictedChurnProbability'] = (watchlist['PredictedChurnProbability'] * 100).round(1)
watchlist.to_csv(f"{OUT}/pg5_watchlist_top100.csv", index=False)

print("Dashboard datasets written:")
for f in ['pg1_kpis', 'pg2_cohort_by_tenure', 'pg3_by_contract', 'pg3_by_internet',
          'pg3_by_payment', 'pg3_feature_importance', 'pg4_revenue_at_risk', 'pg5_watchlist_top100']:
    d = pd.read_csv(f"{OUT}/{f}.csv")
    print(f"  {f}.csv -> {d.shape}")

print("\nPage 1 KPIs:")
print(kpis.T)
print("\nPage 4 Revenue at risk by cohort:")
print(revenue_at_risk_by_cohort)
