import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Load enriched data
df = pd.read_csv("f1_data_enriched.csv")
df = df[df['Year'] >= 2019].copy() # Filter to start from 2019 (provides history for 2021 predictions)

# ── Clean & Basic Preprocessing ───────────────────────────
df['is_winner'] = df['is_winner'].fillna(0).astype(int)
df['is_top3'] = (pd.to_numeric(df['Position'], errors='coerce') <= 3).fillna(0).astype(int)
df['is_top5'] = (pd.to_numeric(df['Position'], errors='coerce') <= 5).fillna(0).astype(int)
df['Position'] = pd.to_numeric(df['Position'], errors='coerce').fillna(20).astype(int)
df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce').fillna(20).astype(int)
df['is_mechanical_failure'] = df['is_mechanical_failure'].fillna(0).astype(int)
df['pit_stops_count'] = pd.to_numeric(df['pit_stops_count'], errors='coerce').fillna(2).astype(int)
df['rain_probability'] = df['rain_probability'].fillna(0).astype(float)
df['avg_air_temp'] = df['avg_air_temp'].fillna(25.0).astype(float)
df['safety_car_laps'] = df['safety_car_laps'].fillna(0).astype(int)
df['event_name'] = df['event_name'].fillna('Unknown')

# Sort by time for temporal features
df = df.sort_values(['Year', 'Round'])

# ── Circuit Type Mapping ──────────────────────────────────
circuit_types = {
    'Bahrain Grand Prix': 'Permanent', 'Emilia Romagna Grand Prix': 'Permanent',
    'Portuguese Grand Prix': 'Permanent', 'Spanish Grand Prix': 'Permanent',
    'Monaco Grand Prix': 'Street', 'Azerbaijan Grand Prix': 'Street',
    'French Grand Prix': 'Permanent', 'Styrian Grand Prix': 'Permanent',
    'Austrian Grand Prix': 'Permanent', 'British Grand Prix': 'Permanent',
    'Hungarian Grand Prix': 'Permanent', 'Belgian Grand Prix': 'Permanent',
    'Dutch Grand Prix': 'Permanent', 'Italian Grand Prix': 'Permanent',
    'Russian Grand Prix': 'Permanent', 'Turkish Grand Prix': 'Permanent',
    'United States Grand Prix': 'Permanent', 'Mexico City Grand Prix': 'Permanent',
    'São Paulo Grand Prix': 'Permanent', 'Qatar Grand Prix': 'Permanent',
    'Saudi Arabian Grand Prix': 'Street', 'Abu Dhabi Grand Prix': 'Permanent',
    'Australian Grand Prix': 'Street', 'Miami Grand Prix': 'Street',
    'Canadian Grand Prix': 'Street', 'Singapore Grand Prix': 'Street',
    'Japanese Grand Prix': 'Permanent', 'Las Vegas Grand Prix': 'Street',
    'Chinese Grand Prix': 'Permanent'
}
df['circuit_type'] = df['event_name'].map(circuit_types).fillna('Permanent')

# ── Pre-Race Stat Verification (Addressing Leakage) ─────────

def compute_lagged_stats(df, group_cols, target_col, agg_func='mean'):
    race_stats = df.groupby(group_cols + ['Year', 'Round'])[target_col].agg(agg_func).reset_index()
    race_stats = race_stats.sort_values(['Year', 'Round'])
    shifted_name = f"lagged_{target_col}"
    race_stats[shifted_name] = race_stats.groupby(group_cols)[target_col].transform(
        lambda x: x.shift().expanding().mean()
    )
    return race_stats.drop(columns=[target_col])

# Driver Stats (Lagged)
driver_wins = compute_lagged_stats(df, ['Abbreviation'], 'is_winner')
driver_top3 = compute_lagged_stats(df, ['Abbreviation'], 'is_top3')
driver_top5 = compute_lagged_stats(df, ['Abbreviation'], 'is_top5')
driver_pos = compute_lagged_stats(df, ['Abbreviation'], 'Position')
driver_rel = compute_lagged_stats(df, ['Abbreviation'], 'is_mechanical_failure')

# Team Stats (Lagged)
team_wins = compute_lagged_stats(df, ['TeamName'], 'is_winner')
team_pos = compute_lagged_stats(df, ['TeamName'], 'Position')

# Track Stats (Lagged)
track_pos = compute_lagged_stats(df, ['Abbreviation', 'event_name'], 'Position')
track_sc = compute_lagged_stats(df, ['event_name'], 'safety_car_laps')
track_rain = compute_lagged_stats(df, ['event_name'], 'rain_probability')

# ── NEW: Wet Weather History ──
rainy_races = df[df['rain_probability'] > 0.1].copy()
driver_wet_stats = compute_lagged_stats(rainy_races, ['Abbreviation'], 'Position')
driver_wet_stats.rename(columns={'lagged_Position': 'driver_wet_avg_pos'}, inplace=True)

# Merge all back
df = df.merge(driver_wins, on=['Abbreviation', 'Year', 'Round'], how='left')
df = df.merge(driver_top3, on=['Abbreviation', 'Year', 'Round'], how='left')
df = df.merge(driver_top5, on=['Abbreviation', 'Year', 'Round'], how='left')
df = df.merge(driver_pos, on=['Abbreviation', 'Year', 'Round'], how='left')
df = df.merge(driver_rel, on=['Abbreviation', 'Year', 'Round'], how='left')
df = df.merge(team_wins, on=['TeamName', 'Year', 'Round'], how='left')
df = df.merge(team_pos, on=['TeamName', 'Year', 'Round'], how='left')
df = df.merge(track_pos, on=['Abbreviation', 'event_name', 'Year', 'Round'], how='left')
df = df.merge(track_sc, on=['event_name', 'Year', 'Round'], how='left')
df = df.merge(track_rain, on=['event_name', 'Year', 'Round'], how='left')
df = df.merge(driver_wet_stats, on=['Abbreviation', 'Year', 'Round'], how='left')

# Fill NaNs
df['lagged_is_winner_x'] = df['lagged_is_winner_x'].fillna(0)
df['lagged_is_top3'] = df['lagged_is_top3'].fillna(0)
df['lagged_is_top5'] = df['lagged_is_top5'].fillna(0)
df['lagged_Position_x'] = df['lagged_Position_x'].fillna(20)
df['lagged_is_mechanical_failure'] = df['lagged_is_mechanical_failure'].fillna(0.1)
df['lagged_is_winner_y'] = df['lagged_is_winner_y'].fillna(0)
df['lagged_Position_y'] = df['lagged_Position_y'].fillna(20)
df['lagged_Position'] = df['lagged_Position'].fillna(20)
df['lagged_safety_car_laps'] = df['lagged_safety_car_laps'].fillna(0)
df['lagged_rain_probability'] = df['lagged_rain_probability'].fillna(0)
df['driver_wet_avg_pos'] = df['driver_wet_avg_pos'].fillna(20)

# Recent Form (Moving average)
df['driver_recent_avg_pos'] = df.groupby('Abbreviation')['Position'].transform(
    lambda x: x.shift().rolling(5, min_periods=1).mean()
).fillna(20)

# Recent Form (Hot Streak - last 3 races)
df['driver_hot_form'] = df.groupby('Abbreviation')['Position'].transform(
    lambda x: x.shift().rolling(3, min_periods=1).mean()
).fillna(20)

# Team-mate Comparison (Position relative to teammate)
df['teammate_pos_diff'] = df.groupby(['Year', 'Round', 'TeamName'])['Position'].transform(
    lambda x: x - x.mean()
)
df['lagged_teammate_diff'] = df.groupby('Abbreviation')['teammate_pos_diff'].transform(
    lambda x: x.shift().expanding().mean()
).fillna(0)

# Grid Efficiency (How many positions gained/lost on average)
df['pos_gain'] = df['GridPosition'] - df['Position']
df['lagged_pos_gain'] = df.groupby('Abbreviation')['pos_gain'].transform(
    lambda x: x.shift().expanding().mean()
).fillna(0)

# ── LOG TRANSFORMATION & INTERACTIONS ────────────────────
# P1->P2 is bigger than P10->P11. Log scale captures this.
# Using log1p to handle any 0 values safely (though Grid is usually 1-20)
df['log_grid_pos'] = np.log1p(df['GridPosition'])

# Rebuild interaction with Log Grid
df['grid_team_interaction'] = df['log_grid_pos'] * df['lagged_Position_y']

# ── Target: Relative Finishing Position (Finish - Grid) ──
df['Target'] = df['Position'] - df['GridPosition']

# ── Encoding ────────────────────────────
le_event = LabelEncoder()
df['Event_encoded'] = le_event.fit_transform(df['event_name'])
le_circuit = LabelEncoder()
df['CircuitType_encoded'] = le_circuit.fit_transform(df['circuit_type'])

# ── Final Feature Selection ───────────────────────────
# GridPosition is kept for metadata/evaluation, but dropped from training in mainn.py
# Removed: driver_top3_rate, driver_top5_rate, team_win_rate (highly correlated)
# Removed: rain_wet_performance_interaction (too highly correlated with track_rain_prob)

data = df[[
    'Year', 'Round', 'Abbreviation',
    'Position', 'GridPosition', 'Target',
    'log_grid_pos',
    'lagged_is_winner_x', 'lagged_Position_x',
    'lagged_Position_y',
    'driver_recent_avg_pos', 'driver_wet_avg_pos',
    'lagged_Position', 'lagged_safety_car_laps', 'lagged_rain_probability',
    'grid_team_interaction',
    'driver_hot_form', 'lagged_teammate_diff', 'lagged_pos_gain',
    'Event_encoded', 'CircuitType_encoded'
]].copy()

data.columns = [
    'Year', 'Round', 'Abbreviation',
    'Position', 'GridPosition', 'Target',
    'log_grid_pos',
    'driver_win_rate', 'driver_avg_pos',
    'team_avg_pos',
    'driver_recent_avg_pos', 'driver_wet_avg_pos',
    'track_avg_pos', 'track_avg_sc_laps', 'track_rain_prob',
    'grid_team_interaction',
    'driver_hot_form', 'teammate_diff', 'pos_gain_avg',
    'Event_encoded', 'CircuitType_encoded'
]

train = data[data['Year'] < 2022].copy()
test  = data[data['Year'] >= 2022].copy()

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

print(f"✅ Refined features saved (Log Grid + Interactions).")
print(f"Dropped: GridPosition (raw), experience, avg_pit_stops")
