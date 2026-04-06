import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
import os
import json
import plotly.express as px

# Set Page Config
st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")

st.markdown("""
    <style>
    .f1-header-clean {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(128,128,128,0.2);
    }
    .f1-logo-clean {
        width: 140px; 
        animation: fastSlideIn 0.5s cubic-bezier(0.1, 0.9, 0.2, 1) forwards;
    }
    .f1-title-clean {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
        animation: fastSlideIn 0.5s cubic-bezier(0.1, 0.9, 0.2, 1) forwards;
        animation-delay: 0.1s;
    }
    @keyframes fastSlideIn {
        0% { transform: translateX(-60px); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    </style>
    
    <div class="f1-header-clean">
        <img src="https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg" class="f1-logo-clean">
        <h1 class="f1-title-clean">Race Prediction Results</h1>
    </div>
""", unsafe_allow_html=True)

# ── LOAD DATA ──────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists("train.csv") or not os.path.exists("test.csv"):
        return None, None
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    # We need FullName which might be in enriched but not in train/test
    # Let's merge it back from enriched
    enriched = pd.read_csv("f1_data_enriched.csv")[['Year', 'Round', 'Abbreviation', 'FullName']].drop_duplicates()
    
    full_data = pd.concat([train, test], ignore_index=True)
    full_data = full_data.merge(enriched, on=['Year', 'Round', 'Abbreviation'], how='left')
    
    # Deduplicate
    full_data = full_data.drop_duplicates(subset=['Year', 'Round', 'Abbreviation'])
    
    if 'FullName' not in full_data.columns:
        full_data['FullName'] = full_data['Abbreviation']
        
    name_mapping = {
        'Sergio Pérez': 'Sergio Perez',
        'Andrea Kimi Antonelli': 'Kimi Antonelli'
    }
    full_data['FullName'] = full_data['FullName'].replace(name_mapping)
        
    return full_data, [c for c in full_data.columns if c not in ['Year', 'Round', 'Abbreviation', 'FullName', 'Position', 'Target', 'GridPosition', 'driver_recent_avg_pos']]

full_data, feature_cols = load_data()

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
    return StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)),
            ('lgbm', LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1))
        ],
        final_estimator=RidgeCV()
    )

@st.cache_resource
def train_and_predict(year, round_num):
    train_mask = (full_data['Year'] < year) | ((full_data['Year'] == year) & (full_data['Round'] < round_num))
    train_df = full_data[train_mask].copy()
    test_df = full_data[(full_data['Year'] == year) & (full_data['Round'] == round_num)].copy()
    
    if test_df.empty or train_df.empty:
        return None, None
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[feature_cols])
    X_train = pd.DataFrame(X_train_scaled, columns=feature_cols)
    y_train = train_df['Target']
    
    X_test_scaled = scaler.transform(test_df[feature_cols])
    X_test = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    model = get_stack_model()
    model.fit(X_train, y_train)
    
    # Use decision scores or raw predictions to estimate confidence
    preds = model.predict(X_test)
    test_df['RawPredictedPosition'] = test_df['GridPosition'] + preds
    
    # Simple probability estimation based on how close they are to the top rank
    best_score = test_df['RawPredictedPosition'].min()
    test_df['Win Prob %'] = np.exp(-(test_df['RawPredictedPosition'] - best_score) / 2.0)
    test_df['Win Prob %'] = (test_df['Win Prob %'] / test_df['Win Prob %'].sum() * 100).round(1)
    
    return test_df.sort_values('RawPredictedPosition'), model

# Sidebar
st.sidebar.header("Navigation")
years = sorted(full_data['Year'].unique(), reverse=True)
years = [y for y in years if y >= 2021]
selected_year = st.sidebar.selectbox("Season", years, index=0)

rounds = sorted(full_data[full_data['Year'] == selected_year]['Round'].unique())
selected_round = st.sidebar.selectbox("Round", rounds, index=0)
season_data = full_data[full_data['Year'] == selected_year]
winners_list = season_data[season_data['Position'] == 1][['Round', 'FullName']].drop_duplicates(['Round']).sort_values('Round')
winners_list.columns = ['Rd', 'Winner']

st.sidebar.markdown("### 🏆 Season Winners")
st.sidebar.dataframe(
    winners_list, 
    hide_index=True, 
    use_container_width=True,
    column_config={"Winner": st.column_config.TextColumn(width="medium")}
)

# ── MAIN CONTENT ─────────────────────────────────────
test_data, model = train_and_predict(selected_year, selected_round)

if test_data is not None:
    # Metrics
    actual_winner = test_data.loc[test_data['Position'] == 1, 'FullName'].values[0] if not test_data[test_data['Position'] == 1].empty else "N/A"
    pred_winner = test_data.iloc[0]['FullName']
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    col1.metric("Actual Winner", actual_winner)
    col2.metric("Predicted Winner", pred_winner, delta=f"{test_data.iloc[0]['Win Prob %']:.1f}% Conf", delta_color="normal")
    
    podium_actual = set(test_data.loc[test_data['Position'] <= 3, 'FullName'])
    podium_pred = set(test_data.iloc[:3]['FullName'])
    col3.metric("Podium Hits", f"{len(podium_actual.intersection(podium_pred))}/3", help="Number of drivers correctly predicted to finish in the top 3 (regardless of exact order).")

    st.markdown(f"### 🏁 {selected_year} Round {selected_round} Results")
    
    if actual_winner != "N/A" and actual_winner != pred_winner:
        actual_winner_rank = (test_data['FullName'] == actual_winner).values.argmax() + 1
        conf = test_data[test_data['FullName'] == actual_winner]['Win Prob %'].values[0]
        st.caption(f"**Note:** The actual winner ({actual_winner}) was ranked **#{actual_winner_rank}** by the model with **{conf:.1f}%** confidence.")
    elif actual_winner != "N/A" and actual_winner == pred_winner:
        conf = test_data[test_data['FullName'] == actual_winner]['Win Prob %'].values[0]
        st.caption(f"**Note:** The model correctly predicted the winner ({actual_winner}) with **{conf:.1f}%** confidence.")
    
    season_wins = season_data[season_data['Position'] == 1]['FullName'].value_counts().to_dict()
    
    display_df = test_data[['FullName', 'Position', 'Win Prob %']].copy()
    display_df['Season Wins'] = display_df['FullName'].map(lambda x: season_wins.get(x, 0))
    display_df['Model Rank'] = range(1, len(display_df) + 1)
    
    display_df = display_df[['FullName', 'Position', 'Season Wins', 'Win Prob %', 'Model Rank']]
    display_df.columns = ['Driver Name', 'Actual Finish', 'Season Wins', 'Win Confidence', 'Model Rank']
    
    def highlight_results(row):
        if row['Actual Finish'] == 1: return ['background-color: #ffd700; color: black'] * len(row)
        if row['Actual Finish'] <= 3: return ['background-color: #3d3d3d; color: white'] * len(row)
        return [''] * len(row)

    tab1, tab2 = st.tabs(["🏁 Predicted Classification", "🍩 Win Probability Chart"])
    
    with tab1:
        st.dataframe(
            display_df.style.apply(highlight_results, axis=1), 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Driver Name": st.column_config.TextColumn(width="medium"),
                "Win Confidence": st.column_config.ProgressColumn(
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                )
            }
        )
        
    with tab2:
        top_plot_df = display_df.head(8)
        fig = px.pie(top_plot_df, values='Win Confidence', names='Driver Name', 
                     title=f"Top 8 Win Probability - {selected_year} Round {selected_round}", 
                     hole=0.4,
                     color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_traces(textinfo='percent+label', textposition='inside')
        
        # Inject custom CSS entry animation for the Plotly Donut Chart
        st.markdown("""
        <style>
        /* Smooth radial expansion from the inside-out specifically on the pie slices */
        @keyframes donutBloom {
            0% { clip-path: circle(0% at 50% 50%); }
            100% { clip-path: circle(100% at 50% 50%); }
        }
        @keyframes textFade {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        /* Target the internal SVG pie layer directly */
        g.pielayer {
            animation: donutBloom 1.5s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
        }
        /* Delay the text label fade-in until the donut has expanded */
        g.slicetext {
            opacity: 0;
            animation: textFade 0.6s ease-out forwards;
            animation-delay: 1.0s;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Not enough historical data to generate predictions for this selection.")
