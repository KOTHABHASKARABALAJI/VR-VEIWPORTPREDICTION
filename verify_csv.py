"""
Script: verify_csv.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Purpose:
    Performs an automated sanity check and structural validation of the consolidated CSV dataset
    ('combined_viewport_data.csv'). It checks file existence, loads the dataset, inspects the
    table dimensions (total rows and columns), checks column names, and displays the top 10 rows.

Role in Workflow:
    Stage 4 (Data Ingestion Verification): Ensures that the parsing of 48 subject JSON files
    into a unified tabular format succeeded without file truncation or missing column headers.
"""

# Standard library module for object-oriented filesystem paths with cross-platform compatibility
from pathlib import Path

# Third-party tabular data processing library used to load and inspect CSV files
import pandas as pd


def verify_csv(csv_path: str | Path):
    """
    Validates and summarizes the structural integrity of the specified CSV file.

    Parameters:
        csv_path (str | Path): Absolute or relative filesystem path to the CSV file to verify.

    Returns:
        None: Outputs inspection details directly to the console.
    """
    # Convert string or Path object into a standard pathlib.Path object
    path = Path(csv_path)

    # Sanity check: Ensure the target file exists before attempting to read it
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return

    # Inform the user which file is currently being loaded into memory
    print(f"Reading CSV file for verification: {path.name}...\n")

    # Load the CSV file into a pandas DataFrame (reads all rows and columns)
    df = pd.read_csv(path)

    # Print a structured diagnostic verification report
    print("=" * 60)
    print("                   VERIFICATION REPORT                   ")
    print("=" * 60)

    # Display DataFrame dimensions: Expected (284247, 6) for combined_viewport_data.csv
    # Rows represent individual 100ms head tracking samples; columns represent feature fields
    print(f"DataFrame Shape (Rows, Columns): {df.shape}")

    # Display all column header names to ensure schema consistency
    # Expected columns: ['video', 'hmd', 'timestamp', 'pitch', 'yaw', 'roll']
    print(f"Columns: {list(df.columns)}")
    print("-" * 60)

    # Preview the first 10 rows of data in a cleanly aligned tabular format without default row indices
    print("First 10 Rows:")
    print(df.head(10).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    # Locate the project root directory dynamically relative to this script's location
    project_dir = Path(__file__).parent.resolve()

    # Define the target path to combined_viewport_data.csv in the project directory
    csv_file = project_dir / "combined_viewport_data.csv"

    # Execute the CSV verification routine
    verify_csv(csv_file)
