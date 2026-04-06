import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n🚀 Running {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ {script_name} failed with exit code {result.returncode}")
        sys.exit(1)
    print(f"✅ {script_name} completed successfully.")

if __name__ == "__main__":
    print("🏁 F1 Race Prediction Pipeline")
    print("═" * 30)
    
    # 1. Enrich data (Mandatory for new years)
    run_script("enrich_f1_data.py")

    # 2. Process data and generate features
    run_script("test.py")

    # 3. Hyperparameter Tuning
    run_script("tune_model.py")

    # 4. Train and evaluate
    run_script("mainn.py")

    print("\n🎉 Pipeline execution finished!")
