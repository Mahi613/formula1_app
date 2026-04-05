import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import json

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
full_data = pd.concat([train_df, test_df], ignore_index=True)

# We'll tune on 2021-2023 and validate on 2024
train_data = full_data[full_data['Year'] < 2024].copy()
val_data = full_data[full_data['Year'] == 2024].copy()

feature_cols = [c for c in full_data.columns if c not in ['Year', 'Round', 'Abbreviation', 'Position', 'Target', 'GridPosition']]
target_col = 'Target'

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data[feature_cols])
X_train = pd.DataFrame(X_train_scaled, columns=feature_cols)
y_train = train_data[target_col]
X_val_scaled = scaler.transform(val_data[feature_cols])
X_val = pd.DataFrame(X_val_scaled, columns=feature_cols)
y_val = val_data[target_col]

def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'random_state': 42
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds)

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds)

def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'verbose': -1,
        'random_state': 42
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds)

print("🚀 Starting Hyperparameter Tuning...")

study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=30)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=30)

study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(objective_lgbm, n_trials=30)

best_params = {
    'rf': study_rf.best_params,
    'xgb': study_xgb.best_params,
    'lgbm': study_lgbm.best_params
}

with open('best_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print("\n✅ Tuning Complete! Best parameters saved to best_params.json")
print(f"Best RF MSE: {study_rf.best_value:.4f}")
print(f"Best XGB MSE: {study_xgb.best_value:.4f}")
print(f"Best LGBM MSE: {study_lgbm.best_value:.4f}")
