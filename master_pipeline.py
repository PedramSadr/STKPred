import subprocess
import sys
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PIPELINE] - %(message)s',
    handlers=[
        logging.FileHandler("daily_pipeline.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)


def find_file_path(target_file):
    """
    Scans the entire project tree to find the full path of a specific file.
    Returns: (full_path, directory_containing_file)
    """
    # Start searching from the project root (where this master script is)
    root_dir = os.path.dirname(os.path.abspath(__file__))

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_file in filenames:
            full_path = os.path.join(dirpath, target_file)
            logging.info(f"[FOUND] {target_file} at: {full_path}")
            return full_path, dirpath

    return None, None


def run_step(script_name, description):
    """
    Finds the script anywhere in the project and runs it from its own folder.
    """
    logging.info(f"--- [STEP START] {description} ---")

    # 1. Hunt down the file
    script_path, work_dir = find_file_path(script_name)

    if not script_path:
        logging.error(f"!!! MISSING FILE: Could not find '{script_name}' anywhere in this project !!!")
        return False

    try:
        # 2. Run the script from its OWN directory
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=work_dir,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.stdout:
            logging.info(f"OUTPUT:\n{result.stdout.strip()}")

        logging.info(f"--- [STEP SUCCESS] {description} ---\n")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"!!! FAILED: {description} !!!")
        logging.error(f"ERROR LOG:\n{e.stderr}")
        if e.stdout:
            logging.error(f"STANDARD OUTPUT:\n{e.stdout}")
        return False


def main():
    print("Starting Daily Trading Pipeline...")
    print("Scanning project for required scripts...\n")

    # ---------------------------------------------------------
    # STEP 1: DOWNLOAD RAW DATA
    # ---------------------------------------------------------
    if not run_step("GemniDownloader.py", "1. Download Fresh Data"):
        sys.exit(1)

    # ---------------------------------------------------------
    # STEP 2: ARCHIVE DATA
    # ---------------------------------------------------------
    if not run_step("FileAppender.py", "2. Append to History"):
        sys.exit(1)

    # ---------------------------------------------------------
    # STEP 3: GENERATE SURFACE
    # ---------------------------------------------------------
    if not run_step("surface_generator.py", "3. Generate Vol Surface"):
        sys.exit(1)

    # ---------------------------------------------------------
    # STEP 4: STRATEGY EXECUTION
    # ---------------------------------------------------------
    # FIXED: Changed from 'daily_run.py' to 'run_daily.py'
    if not run_step("run_daily.py", "4. Daily Strategy Execution"):
        sys.exit(1)

    print("\nPIPELINE COMPLETED SUCCESSFULLY.")
    logging.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()