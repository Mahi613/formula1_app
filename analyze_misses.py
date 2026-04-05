import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import os
import json

def get_stack_model():
    if os.path.exists('best_params.json'):
        with open('best_params.json', 'r') as f:
            best = json.load(f)
        return StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(**best['rf'], random_state=42)),
                ('xgb', XGBRegressor(**best['xgb'], random_state=42)),
                ('lgbm', LGBMRegressor(**best['lgbm'], random_state=42, verbose=-1))
            ],
            final_estimator=RidgeCV()
        )
    return None

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
full_data = pd.concat([train_df, test_df], ignore_index=True)

year = 2024
feature_cols = [c for c in full_data.columns if c not in ['Year', 'Round', 'Abbreviation', 'Position', 'Target', 'GridPosition']]
target_col = 'Target'

train_data = full_data[full_data['Year'] < year].copy()
test_data = full_data[full_data['Year'] == year].copy()

scaler = StandardScaler()
X_train = scaler.fit_transform(train_data[feature_cols])
y_train = train_data[target_col]
X_test = scaler.transform(test_data[feature_cols])

model = get_stack_model()
model.fit(X_train, y_train)

y_pred_delta = model.predict(X_test)
test_data['PredictedDelta'] = y_pred_delta
test_data['RawPredictedPosition'] = test_data['GridPosition'] + test_data['PredictedDelta']

print(f"--- 2024 Missed Winners Analysis ---")
for rnd, group in test_data.groupby('Round'):
    actual_winner = group.loc[group['Position'] == 1, 'Abbreviation'].values[0]
    pred_winner = group.loc[group['RawPredictedPosition'].idxmin(), 'Abbreviation']
    
    if actual_winner != pred_winner:
        actual_grid = group.loc[group['Position'] == 1, 'GridPosition'].values[0]
        pred_grid = group.loc[group['Abbreviation'] == pred_winner, 'GridPosition'].values[0]
        print(f"Round {rnd:2}: Miss! Actual: {actual_winner} (Grid {actual_grid}) | Predicted: {pred_winner} (Grid {pred_grid})")
