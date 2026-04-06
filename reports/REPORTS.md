# F1 Race Prediction Model Reports

## Current Status & Improvements
- **Redundant Features Removed**: Removed `driver_top3_rate`, `driver_top5_rate`, `team_win_rate`, and `rain_wet_performance_interaction` to simplify the model and reduce overfitting.
- **Tested Across Multiple Seasons**: The model is now evaluated on 4 separate seasons (2022-2025) using a temporal split (train on previous years, test on current).
- **Podium Accuracy Added**: Added Top 3 prediction accuracy (coverage).
- **Feature Leakage Verified**: Ensured all features use `shift()` or are known pre-race (Grid Position).
- **Clean Reproducible Pipeline**: Created `pipeline.py` to handle the end-to-end workflow.

## Overall Performance Across Seasons

| Year | Avg Spearman | Winner Accuracy | Podium Accuracy | ROC AUC |
| :--- | :--- | :--- | :--- | :--- |
| 2022 | 0.5487 | 36.4% | 53.0% | 0.8633 |
| 2023 | 0.6361 | 45.5% | 62.1% | 0.9358 |
| 2024 | 0.7328 | 45.8% | 55.6% | 0.9014 |
| 2025 | 0.6501 | 58.3% | 69.4% | 0.9598 |

**Average Winner Accuracy (Last 4 Years): 46.5%**
**Average Podium Accuracy (Last 4 Years): 60.0%**
**Average Spearman Correlation: 0.6419**

## Feature Importance (Top 5)
1. `log_grid_pos`: 37.6%
2. `grid_team_interaction`: 18.1%
3. `team_avg_pos`: 12.7%
4. `driver_avg_pos`: 6.5%
5. `driver_recent_avg_pos`: 6.4%

## Analysis
- **Grid Position is King**: As expected in F1, the starting position is the most influential factor.
- **Team Performance Matters**: The `grid_team_interaction` and `team_avg_pos` show that car performance is a critical predictor.
- **Improving Accuracy**: Winner accuracy and Podium accuracy (Top 3 coverage) have shown strong results, peaking in 2025 validation.
