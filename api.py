import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
import os

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for data
full_data = None
feature_cols = None
target_col = 'Target'

def get_stack_model():
    return StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)),
            ('lgbm', LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1))
        ],
        final_estimator=RidgeCV()
    )

@app.on_event("startup")
def load_data():
    global full_data, feature_cols
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    full_data = pd.concat([train, test], ignore_index=True)
    feature_cols = [c for c in full_data.columns if c not in ['Year', 'Round', 'Abbreviation', 'Position', 'Target', 'GridPosition']]

@app.get("/metadata")
def get_metadata():
    years = sorted(full_data['Year'].unique().tolist())
    metadata = {}
    for year in years:
        rounds = sorted(full_data[full_data['Year'] == year]['Round'].unique().tolist())
        metadata[year] = rounds
    return metadata

@app.get("/predictions/{year}/{round_num}")
def get_predictions(year: int, round_num: int):
    # Train on everything before this year
    train_data = full_data[full_data['Year'] < year].copy()
    test_data = full_data[(full_data['Year'] == year) & (full_data['Round'] == round_num)].copy()
    
    if test_data.empty:
        return {"error": "No data found for this year/round"}

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[feature_cols])
    y_train = train_data[target_col]
    X_test = scaler.transform(test_data[feature_cols])

    model = get_stack_model()
    model.fit(X_train, y_train)

    y_pred_delta = model.predict(X_test)
    test_data['PredictedDelta'] = y_pred_delta
    test_data['RawPredictedPosition'] = test_data['GridPosition'] + test_data['PredictedDelta']
    
    # Sort by predicted position
    test_data = test_data.sort_values('RawPredictedPosition')
    
    results = test_data[['Abbreviation', 'Position', 'GridPosition', 'RawPredictedPosition']].to_dict(orient='records')
    
    # Metrics for this specific race
    actual_winner = test_data.loc[test_data['Position'] == 1, 'Abbreviation'].values[0] if not test_data[test_data['Position'] == 1].empty else None
    pred_winner = test_data.iloc[0]['Abbreviation']
    
    actual_podium = set(test_data.loc[test_data['Position'] <= 3, 'Abbreviation'])
    pred_podium = set(test_data.iloc[:3]['Abbreviation'])
    podium_hit = len(actual_podium.intersection(pred_podium))
    
    corr, _ = spearmanr(test_data['Position'], test_data['RawPredictedPosition'])

    return {
        "race": f"{year} Round {round_num}",
        "predictions": results,
        "metrics": {
            "winner_hit": actual_winner == pred_winner,
            "podium_hits": podium_hit,
            "spearman": float(corr) if not np.isnan(corr) else 0
        }
    }

@app.get("/importance")
def get_importance():
    # Use latest available data for importance
    latest_year = full_data['Year'].max()
    train_data = full_data[full_data['Year'] < latest_year].copy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[feature_cols])
    y_train = train_data[target_col]
    
    model = get_stack_model()
    model.fit(X_train, y_train)
    
    rf_base = model.named_estimators_['rf']
    importances = pd.Series(rf_base.feature_importances_, index=feature_cols).sort_values(ascending=False)
    
    return importances.to_dict()

@app.get("/stats")
def get_stats():
    # Return pre-computed or on-the-fly stats for overview
    # For now, just return a summary of the 2022-2025 performance from REPORTS.md context
    return [
        {"year": 2022, "spearman": 0.5487, "accuracy": 36.4, "podium": 53.0},
        {"year": 2023, "spearman": 0.6361, "accuracy": 45.5, "podium": 62.1},
        {"year": 2024, "spearman": 0.7328, "accuracy": 45.8, "podium": 55.6},
        {"year": 2025, "spearman": 0.6501, "accuracy": 58.3, "podium": 69.4}
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
