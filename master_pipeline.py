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


def find_file_path(target_file):
    """Scans the project to find the script's location."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_file in filenames:
            return os.path.join(dirpath, target_file), dirpath
    return None, None


def run_step(script_name, description):
    """Runs a script and streams its output in real-time."""
    print(f"\n--- [STEP START] {description} ---")
    logging.info(f"--- [STEP START] {description} ---")

    script_path, work_dir = find_file_path(script_name)

    if not script_path:
        msg = f"!!! MISSING FILE: {script_name} not found !!!"
        print(msg)
        logging.error(msg)
        return False

    try:
        # Run with real-time output streaming
        process = subprocess.Popen(
            [sys.executable, script_name],
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

    # 1. Download Stock Data (Yahoo)
    if not run_step("GemniDownloader.py", "1. Download Stock Data"): sys.exit(1)

    # 2. Download Option Chains (AlphaVantage)
    if not run_step("AlphavantageDownloader.py", "2. Download Option Chains"): sys.exit(1)

    # 3. Consolidate Chains (Appender)
    if not run_step("FileAppender.py", "3. Append to History"): sys.exit(1)

    # 4. Generate Volatility Surface (Physics)
    if not run_step("surface_generator.py", "4. Generate Vol Surface"): sys.exit(1)

    # 5. Execute Strategy (Gatekeeper)
    if not run_step("run_daily.py", "5. Daily Strategy Execution"): sys.exit(1)

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY.")