import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import pandas as pd 
import fastf1 
import os
import time

# ✅ Setup cache
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

# ✅ Years you want
years = [2021, 2022, 2023, 2024, 2025]

# ✅ Load existing dataset if present
if os.path.exists('f1_data.csv'):
    df_existing = pd.read_csv('f1_data.csv')
    processed = set(zip(df_existing['Year'], df_existing['Round']))
else:
    df_existing = pd.DataFrame()
    processed = set()

for year in years:
    schedule = fastf1.get_event_schedule(year)
    
    for rnd in schedule['RoundNumber']:
        race_id = (year, rnd)

        # ✅ Skip already done races
        if race_id in processed:
            print(f" Skipping {race_id}")
            continue

        try:
            session = fastf1.get_session(year, rnd, 'Race')

            # 🔥 Load ONLY results (fast + safe)
            session.load(laps=False, telemetry=False, weather=False, messages=False)

            results = session.results

            if results is not None and not results.empty:
                results['Year'] = year
                results['Round'] = rnd

                # ✅ Save immediately (no data loss)
                results.to_csv(
                    'f1_data.csv',
                    mode='a',
                    header=not os.path.exists('f1_data.csv'),
                    index=False
                )

                print(f" Saved {year} Round {rnd}")

            else:
                print(f" Empty {year} Round {rnd}")

            time.sleep(1)  # 🔥 prevents overload

        except Exception as e:
            print(f" Failed {year} Round {rnd}: {e}")

# ✅ Load final dataset
df = pd.read_csv('f1_data.csv')
print(df.head())