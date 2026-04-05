import pandas as pd
import fastf1
import os
import time

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV  = 'f1_data.csv'           # raw dataset
OUTPUT_CSV = 'f1_data_enriched.csv'  # enriched output
CACHE_DIR  = 'cache'
# ============================================================

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# ── Load existing data ───────────────────────────────────────
print(f"📂 Loading {INPUT_CSV} ...")
df_raw = pd.read_csv(INPUT_CSV)

if os.path.exists(OUTPUT_CSV):
    df_enriched = pd.read_csv(OUTPUT_CSV)
    processed_races = set(zip(df_enriched['Year'], df_enriched['Round']))
else:
    df_enriched = pd.DataFrame()
    processed_races = set()

# Initialize columns if they don't exist in df_raw
new_cols = ['is_winner', 'laps_led', 'pit_stops_count', 'tyre_compounds', 
            'is_mechanical_failure', 'rain_probability', 'avg_air_temp', 
            'safety_car_laps', 'event_name']
for col in new_cols:
    if col not in df_raw.columns:
        df_raw[col] = None

# ── Identify races to process ────────────────────────────────
unique_races = df_raw[['Year', 'Round']].drop_duplicates().sort_values(['Year', 'Round'])
races_to_process = unique_races.values # Process all found in raw

total = len(races_to_process)
print(f"   Total races to enrich: {total}\n")

if total == 0:
    print("✅ No new races to process.")
else:
    for i, (year, rnd) in enumerate(races_to_process, 1):
        year = int(year)
        rnd  = int(rnd)

        print(f"[{i}/{total}] Processing {year} Round {rnd} ...", end=' ')

        try:
            session = fastf1.get_session(year, rnd, 'Race')
            session.load(laps=True, telemetry=False, weather=True, messages=False)
            laps = session.laps
            weather_data = session.weather_data
            event_name = session.event['EventName']

            race_mask = (df_raw['Year'] == year) & (df_raw['Round'] == rnd)
            drivers   = df_raw.loc[race_mask, 'Abbreviation'].tolist()

            # ── Weather Stats (Race-level) ───────────────────────
            rain_prob = weather_data['Rainfall'].astype(float).mean() if not weather_data.empty else 0
            avg_temp  = weather_data['AirTemp'].astype(float).mean() if not weather_data.empty else 25
            
            # ── Safety Car Laps (Race-level) ────────────────────
            # TrackStatus 4 = SC, 6 = VSC
            sc_mask = laps['TrackStatus'].isin(['4', '6', '44', '64', '46'])
            total_sc_laps = laps[sc_mask]['LapNumber'].nunique()

            for abbr in drivers:
                driver_mask = race_mask & (df_raw['Abbreviation'] == abbr)
                driver_laps = laps[laps['Driver'] == abbr]
                
                df_raw.loc[driver_mask, 'event_name'] = event_name
                df_raw.loc[driver_mask, 'rain_probability'] = rain_prob
                df_raw.loc[driver_mask, 'avg_air_temp'] = avg_temp
                df_raw.loc[driver_mask, 'safety_car_laps'] = total_sc_laps

                # ── Winner flag ──────────────────────────────────
                try:
                    pos = df_raw.loc[driver_mask, 'Position'].values[0]
                    df_raw.loc[driver_mask, 'is_winner'] = 1 if float(pos) == 1.0 else 0
                except Exception:
                    df_raw.loc[driver_mask, 'is_winner'] = 0

                # ── Mechanical Failure ──────────────────────────
                try:
                    status = df_raw.loc[driver_mask, 'Status'].values[0]
                    failure_keywords = ['Engine', 'Gearbox', 'Transmission', 'Clutch', 'Hydraulics', 'Electrical', 'Spun off', 'Accident', 'Collision', 'Power Unit', 'Suspension', 'Brakes', 'Mechanical']
                    is_fail = 1 if any(kw.lower() in str(status).lower() for kw in failure_keywords) else 0
                    df_raw.loc[driver_mask, 'is_mechanical_failure'] = is_fail
                except Exception:
                    df_raw.loc[driver_mask, 'is_mechanical_failure'] = 0

                if driver_laps.empty:
                    continue

                # ── Laps Led ─────────────────────────────────────
                if 'Position' in driver_laps.columns:
                    df_raw.loc[driver_mask, 'laps_led'] = int(
                        (driver_laps['Position'] == 1).sum()
                    )

                # ── Pit Stops ─────────────────────────────────────
                if 'PitOutTime' in driver_laps.columns:
                    df_raw.loc[driver_mask, 'pit_stops_count'] = int(
                        driver_laps['PitOutTime'].notna().sum()
                    )

                # ── Tyre Compounds ────────────────────────────────
                if 'Compound' in driver_laps.columns:
                    compounds = driver_laps['Compound'].dropna().unique().tolist()
                    df_raw.loc[driver_mask, 'tyre_compounds'] = (
                        ', '.join(str(c) for c in compounds) if compounds else None
                    )

            print("done")
            time.sleep(0.5)

        except Exception as e:
            print(f"  Error — {e}")

    # Final Save
    if not df_enriched.empty:
        df_raw.set_index(['Year', 'Round', 'Abbreviation'], inplace=True)
        df_enriched.set_index(['Year', 'Round', 'Abbreviation'], inplace=True)
        df_raw.update(df_enriched)
        df_raw.reset_index(inplace=True)

    df_raw.to_csv(OUTPUT_CSV, index=False)
    print(f"\n Saved enriched dataset → {OUTPUT_CSV}")

# ── Quick sanity checks ──────────────────────────────────────
df = pd.read_csv(OUTPUT_CSV)
print("\n Sanity Checks:")
print(f"   Years: {df['Year'].unique()}")
print(f"   Winners per race count: {df[df['is_winner']==1].shape[0]}")
