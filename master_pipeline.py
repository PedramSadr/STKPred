import subprocess
import sys
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PIPELINE] - %(message)s',
    handlers=[
        logging.FileHandler("daily_pipeline.log", encoding='utf-8')
    ]
)

# --- CONFIGURATION ---
BASE_LOGS_DIR = r"C:\My Documents\Mics\Logs"
SOURCE_DATA_DIR = os.path.join(BASE_LOGS_DIR, r"AlphaVantage\TSLA_Options_Chain_Historical")

# INTERMEDIATE FILE: The massive raw CSV containing all history
RAW_COMBINED_FILE = os.path.join(BASE_LOGS_DIR, "TSLA_Options_Chain_Historical_combined.csv")

# FINAL FILE: The cleaned/processed catalog that run_daily.py reads
FINAL_CATALOG_FILE = os.path.join(BASE_LOGS_DIR, "TSLA_Options_Contracts.csv")


def find_file_path(target_file):
    """Scans the project to find the script's location."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_file in filenames:
            return os.path.join(dirpath, target_file), dirpath
    return None, None


def run_step(script_name, description, args=None):
    """Runs a script with optional arguments and streams output."""
    print(f"\n--- [STEP START] {description} ---")
    logging.info(f"--- [STEP START] {description} ---")

    script_path, work_dir = find_file_path(script_name)

    if not script_path:
        msg = f"!!! MISSING FILE: {script_name} not found !!!"
        print(msg)
        logging.error(msg)
        return False

    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    try:
        process = subprocess.Popen(
            cmd,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace'
        )

        for line in process.stdout:
            print(line, end='')
            logging.info(line.strip())

        return_code = process.wait()

        if return_code == 0:
            print(f"--- [STEP SUCCESS] {description} ---")
            logging.info(f"--- [STEP SUCCESS] {description} ---")
            return True
        else:
            print(f"!!! FAILED: {description} (Code: {return_code}) !!!")
            return False

    except Exception as e:
        print(f"!!! ERROR: {e} !!!")
        logging.error(f"!!! ERROR: {e} !!!")
        return False


if __name__ == "__main__":
    print("Starting Daily Trading Pipeline...")

    # 1. Download Stock Data
    if not run_step("GemniDownloader.py", "1. Download Stock Data"):
        print("Warning: GemniDownloader failed. Prediction models may lack recent data.")

    # 2. Download Option Chains (AlphaVantage)
    if not run_step("AlphavantageDownloader.py", "2. Download Option Chains"): sys.exit(1)

    # FIX: Output to the RAW COMBINED file, not the final catalog
    appender_args = [
        "--input_dir", SOURCE_DATA_DIR,  # <--- Changed hyphen to underscore
        "--output_file", RAW_COMBINED_FILE,  # <--- Changed hyphen to underscore
        "--pattern", "*.csv"
    ]
    if not run_step("FileAppender.py", "3. Append to History", args=appender_args): sys.exit(1)

    # 4. Generate Vol Surface & Final Menu
    # FIX: Explicitly tell it where to read (Raw) and where to write (Final)
    surface_args = [
        "--options", RAW_COMBINED_FILE,
        "--out-menu", FINAL_CATALOG_FILE
    ]
    if not run_step("surface_generator.py", "4. Generate Vol Surface", args=surface_args): sys.exit(1)

    # 5. Execute Strategy
    if not run_step("run_daily.py", "5. Daily Strategy Execution"): sys.exit(1)

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY.")