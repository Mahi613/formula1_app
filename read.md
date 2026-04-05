# F1 Race Predictor

An advanced Formula 1 race outcome prediction model and interactive dashboard built with Machine Learning and Streamlit. The application predicts driver finishing positions using historical race data, track characteristics, and team performance metrics.

## Features

- **Interactive Dashboard**: A beautiful, broadcast-style Streamlit dashboard featuring driver win probabilities and predicted classifications.
- **Advanced Machine Learning**: Utilizes a `StackingRegressor` combining Random Forest, XGBoost, and LightGBM models for robust predictions.
- **Comprehensive Feature Engineering**: Incorporates:
  - Lagged driver and team performance (historical wins, average positions).
  - Track-specific characteristics (circuit type, safety car laps, rain probability).
  - Advanced feature interactions (Log Grid Position, Team/Grid performance interactions).
  - Driver form & teammate comparisons.
- **Dynamic Visualizations**: Including custom Plotly donut charts with CSS animations to display win probability distributions.
- **Backtesting & Evaluation**: Evaluated across multiple seasons (2022-2025) using temporal train/test splits to accurately assess model performance and prevent data leakage.

## Project Structure

- `app.py`: The main Streamlit dashboard application.
- `test.py`: Data preprocessing, temporal feature engineering, and train/test dataset generation.
- `pipeline.py` & `mainn.py`: End-to-end model training pipelines and evaluation scripts.
- `enrich_f1_data.py`: Script to enrich base F1 data with additional historical context.
- `REPORTS.md`: Detailed model performance reports and feature importance analysis.
- `train.csv` & `test.csv`: The processed datasets used for training and inference.
- `best_params.json`: Hyperparameter configurations for the ensemble models.

## Usage

### Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm plotly streamlit scipy
```

### Running the Dashboard
Start the local web application:
```bash
streamlit run app.py
```

### Data Pipeline & Training
To regenerate features and test the model from scratch:
```bash
python test.py
python mainn.py
```

## Performance Highlights

Based on validation testing from the 2022-2025 F1 seasons:
- **Average Winner Accuracy**: 46.5%
- **Average Podium Accuracy (Top 3)**: 60.0%
- **Average Spearman Correlation**: 0.6419
- **Most Influential Predictive Feature**: Starting Grid Position (`log_grid_pos`)
