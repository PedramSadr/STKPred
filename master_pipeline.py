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
# The file that run_daily.py actually reads
FINAL_CATALOG_FILE = r"C:\My Documents\Mics\Logs\TSLA_Options_Contracts.csv"

# The folder where AlphavantageDownloader saves daily files
SOURCE_DATA_DIR = r"C:\My Documents\Mics\Logs\AlphaVantage\TSLA_Options_Chain_Historical"


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

    # Construct command: [python, script_path, arg1, arg2...]
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


def clean_artifacts():
    """
    [DISABLED] Deleting the catalog caused the 'Stale Data' loop.
    We now APPEND to the catalog instead of deleting it.
    """
    print("🧹 [SKIP] Cleanup disabled to preserve historical data.")
    # if os.path.exists(FINAL_CATALOG_FILE):
    #     try:
    #         os.remove(FINAL_CATALOG_FILE)
    #         msg = f"Deleted old catalog: {FINAL_CATALOG_FILE}"
    #         print(f"🧹 {msg}")
    #         logging.info(msg)
    #     except Exception as e:
    #         print(f"⚠️ Could not delete catalog: {e}")


if __name__ == "__main__":
    print("Starting Daily Trading Pipeline...")

    # 1. Download Stock Data (Optional/Placeholder)
    # if not run_step("GemniDownloader.py", "1. Download Stock Data"):
    #     print("Warning: GemniDownloader failed or missing. Continuing...")

    # 2. Download Option Chains (AlphaVantage)
    if not run_step("AlphavantageDownloader.py", "2. Download Option Chains"): sys.exit(1)

    # 3. Consolidate Chains (FileAppender)
    # CRITICAL FIX: We explicit tell FileAppender where to output the file
    appender_args = [
        "--input-dir", SOURCE_DATA_DIR,
        "--output-file", FINAL_CATALOG_FILE,
        "--pattern", "*.csv"
    ]
    if not run_step("FileAppender.py", "3. Append to History", args=appender_args): sys.exit(1)

    # --- CLEANUP STEP (DISABLED) ---
    clean_artifacts()

    # 4. Generate Vol Surface (Assuming this augments the file, does not replace it)
    # if not run_step("surface_generator.py", "4. Generate Vol Surface"): sys.exit(1)

    # 5. Execute Strategy
    if not run_step("run_daily.py", "5. Daily Strategy Execution"): sys.exit(1)

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY.")