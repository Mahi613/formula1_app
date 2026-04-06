import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import os

# ── Load data ──────────────────────────────────────────
if not os.path.exists("train.csv") or not os.path.exists("test.csv"):
    print("❌ train.csv or test.csv missing. Please run test.py first.")
    exit()

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
full_data = pd.concat([train, test], ignore_index=True)

target_col = 'Target' 
feature_cols = [c for c in full_data.columns if c not in ['Year', 'Round', 'Abbreviation', 'Position', 'Target', 'GridPosition', 'driver_recent_avg_pos']]

def get_stack_model():
    # Try to load tuned params
    if os.path.exists('best_params.json'):
        with open('best_params.json', 'r') as f:
            import json
            best = json.load(f)
        
        return StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(**best['rf'], random_state=42)),
                ('xgb', XGBRegressor(**best['xgb'], random_state=42)),
                ('lgbm', LGBMRegressor(**best['lgbm'], random_state=42, verbose=-1))
            ],
            final_estimator=RidgeCV()
        )
    
    # Fallback to defaults
    return StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)),
            ('lgbm', LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1))
        ],
        final_estimator=RidgeCV()
    )

def evaluate_year(test_year):
    print(f"\n🏎️  Testing on Season: {test_year}")
    
    train_data = full_data[full_data['Year'] < test_year].copy()
    test_data  = full_data[full_data['Year'] == test_year].copy()
    
    if test_data.empty:
        print(f"⚠️  No data for {test_year}. Skipping.")
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data[feature_cols])
    X_train = pd.DataFrame(X_train_scaled, columns=feature_cols)
    y_train = train_data[target_col]
    X_test_scaled  = scaler.transform(test_data[feature_cols])
    X_test = pd.DataFrame(X_test_scaled, columns=feature_cols)

    model = get_stack_model()
    model.fit(X_train, y_train)

    y_pred_delta = model.predict(X_test)
    test_data['PredictedDelta'] = y_pred_delta
    test_data['RawPredictedPosition'] = test_data['GridPosition'] + test_data['PredictedDelta']

    results = []
    for rnd, group in test_data.groupby('Round'):
        # actual winners and podiums
        actual_winner_rows = group[group['Position'] == 1]
        if actual_winner_rows.empty:
            continue
        actual_winner = actual_winner_rows['Abbreviation'].values[0]
        actual_podium = set(group.loc[group['Position'] <= 3, 'Abbreviation'])

        # predicted winners and podiums
        pred_winner = group.loc[group['RawPredictedPosition'].idxmin(), 'Abbreviation']
        pred_podium = set(group.sort_values('RawPredictedPosition').head(3)['Abbreviation'])
        
        is_hit = 1 if actual_winner == pred_winner else 0
        podium_hit = len(actual_podium.intersection(pred_podium)) / 3.0
        
        corr, _ = spearmanr(group['Position'], group['RawPredictedPosition'])
        
        results.append({
            'Round': rnd,
            'Hit': is_hit,
            'PodiumHit': podium_hit,
            'Spearman': corr
        })
    
    res_df = pd.DataFrame(results)
    avg_spearman = res_df['Spearman'].mean()
    winner_acc = res_df['Hit'].mean()
    podium_acc = res_df['PodiumHit'].mean()
    
    # ROC AUC for winners
    test_data['is_winner'] = (test_data['Position'] == 1).astype(int)
    auc = roc_auc_score(test_data['is_winner'], -test_data['RawPredictedPosition'])
    
    print(f"✅ Summary for {test_year}:")
    print(f"   Avg Spearman: {avg_spearman:.4f}")
    print(f"   Winner Accuracy: {winner_acc:.1%}")
    print(f"   Podium Accuracy (Avg Top 3 Coverage): {podium_acc:.1%}")
    print(f"   ROC AUC: {auc:.4f}")
    
    return {
        'Year': test_year,
        'Spearman': avg_spearman,
        'Accuracy': winner_acc,
        'PodiumAcc': podium_acc,
        'AUC': auc
    }

# ── MISSION: Tested across 3+ seasons ───────────────────
test_years = [2022, 2023, 2024, 2025]
overall_results = []

for year in test_years:
    res = evaluate_year(year)
    if res:
        overall_results.append(res)

overall_df = pd.DataFrame(overall_results)
print("\n" + "═"*40)
print("🏆 OVERALL PERFORMANCE ACROSS SEASONS")
print("═"*40)
print(overall_df.to_string(index=False))
print("═"*40)

# ── Feature Importance (on last model) ──────────────────
# Re-train on all data before current year for feature importance visualization
latest_year = full_data['Year'].max()
train_data = full_data[full_data['Year'] < latest_year].copy()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data[feature_cols])
X_train = pd.DataFrame(X_train_scaled, columns=feature_cols)
y_train = train_data[target_col]
model = get_stack_model()
model.fit(X_train, y_train)

rf_base = model.named_estimators_['rf']
importances = pd.Series(rf_base.feature_importances_, index=feature_cols).sort_values(ascending=False)

print("\n── Top 5 Most Influential Features ──")
print(importances.head(5))

plt.figure(figsize=(10, 6))
importances.sort_values(ascending=True).plot(kind='barh', color='crimson')
plt.title(f'Feature Importance (Training up to {latest_year-1})', fontsize=14)
plt.tight_layout()
plt.savefig("feature_importance_refined.png")
print("\n✅ Feature importance saved to feature_importance_refined.png")
