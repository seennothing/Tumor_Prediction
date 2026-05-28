"""
Master Execution Script.
Executes preprocessing, model training, and prediction for all tasks sequentially.
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name: str, args: list = None):
    script_path = Path("src") / script_name
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
        print(f"Executing: {script_path} {' '.join(args)}")
    else:
        print(f"Executing: {script_path}")
        
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR executing {script_name}:\n{result.stderr}")
    else:
        for line in result.stdout.split('\n'):
            if line.startswith("[Task") or line.startswith("Preprocessing"):
                print(f"  -> {line.strip()}")
        print("  -> Success.\n")

if __name__ == "__main__":
    tasks = ["task1.py", "task2.py", "task3.py"]
    
    print("========================================")
    print("  PHASE 1: PREPROCESSING")
    print("========================================\n")
    run_script("preprocess.py")
    
    print("========================================")
    print("  PHASE 2: MODEL TRAINING & VALIDATION  ")
    print("========================================\n")
    for task in tasks:
        run_script(task, ["--mode", "train"])
        
    print("========================================")
    print("  PHASE 3: TEST SET PREDICTION          ")
    print("========================================\n")
    for task in tasks:
        run_script(task, ["--mode", "predict"])
        
    print("All tasks completed. Results are saved in the 'results/' directory.")