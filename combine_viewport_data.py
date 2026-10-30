"""
Script: combine_viewport_data.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Purpose:
    Parses the 48 individual participant head tracking JSON files in the 'Head_rotation/' folder,
    extracts time-series orientation data points (video filename, HMD hardware type, timestamp in seconds,
    pitch, yaw, and roll angles in degrees), consolidates all sessions into a single master tabular dataset,
    and exports 'combined_viewport_data.csv'.

Role in Workflow:
    Stage 3 (Raw Viewport Trajectory Harvesting): Transforms raw nested JSON recordings from 48 human
    viewers across 20 videos into a standardized 284,247-row CSV format, serving as the raw data source
    for all subsequent trajectory segmentation, motion analysis, and LSTM/GRU sequence generation.
"""

# Standard library module to parse nested JSON files containing head rotation logs
import json

# Standard library module for object-oriented filesystem paths
from pathlib import Path

# Third-party tabular data manipulation library used to build DataFrames and calculate statistics
import pandas as pd


def combine_json_to_csv(input_folder: str | Path, output_csv_path: str | Path) -> pd.DataFrame:
    """
    Reads all JSON files in input_folder, extracts viewport trajectory data,
    and writes to output_csv_path.

    Parameters:
        input_folder (str | Path): Path to the folder containing the subject JSON files (Head_rotation/).
        output_csv_path (str | Path): Destination path for the consolidated CSV file (combined_viewport_data.csv).

    Returns:
        pd.DataFrame: The consolidated pandas DataFrame containing all 284,247 orientation records.
    """
    # Convert string paths into pathlib.Path objects for reliable path resolution
    folder = Path(input_folder)
    output_file = Path(output_csv_path)

    # Find all .json files in input folder and sort alphabetically
    json_files = sorted(list(folder.glob("*.json")))
    
    if not json_files:
        # Check subdirectories if no json files found directly
        json_files = sorted(list(folder.rglob("*.json")))

    print(f"Found {len(json_files)} JSON file(s) in '{folder}'. Starting conversion...\n")

    # In-memory accumulator for flattened row records
    records = []
    processed_files_count = 0
    failed_files = []

    # Process each subject's head rotation JSON file
    for json_path in json_files:
        try:
            # Open JSON with UTF-8 encoding to prevent character encoding issues
            with open(json_path, "r", encoding="utf-8") as f:
                content = json.load(f)

            # Structural validation: Ensure top-level element is a dict containing the 'data' key
            if not isinstance(content, dict) or "data" not in content:
                raise ValueError(f"Missing 'data' root key or invalid JSON structure in {json_path.name}")

            data_list = content.get("data", [])
            if not isinstance(data_list, list):
                raise ValueError(f"'data' field is not a list in {json_path.name}")

            # Iterate through each video viewing session inside the subject's JSON
            for data_item in data_list:
                if not isinstance(data_item, dict):
                    continue

                # Extract video identifier (e.g., '1.mp4') and headset hardware model (e.g., 'vive')
                video = data_item.get("filename")
                hmd = data_item.get("hmd")
                points = data_item.get("pitch_yaw_roll_data_hmd", [])

                if not isinstance(points, list):
                    continue

                # Iterate through individual 100ms head orientation samples within the session
                for point in points:
                    if not isinstance(point, dict):
                        continue

                    # Append flattened row dictionary
                    records.append({
                        "video": video,
                        "hmd": hmd,
                        "timestamp": point.get("sec"),
                        "pitch": point.get("pitch"),
                        "yaw": point.get("yaw"),
                        "roll": point.get("roll")
                    })

            processed_files_count += 1

        except Exception as e:
            # Isolate any malformed JSON file to prevent failure of the entire batch
            failed_files.append((json_path.name, str(e)))
            print(f"[WARNING] Error processing '{json_path.name}': {e}. Skipping file.")

    # Create DataFrame from accumulated dictionary records with explicit column ordering
    df = pd.DataFrame(records, columns=["video", "hmd", "timestamp", "pitch", "yaw", "roll"])

    # Save to CSV without modifying numerical values or dropping duplicates
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"[SUCCESS] CSV successfully saved to: {output_file.resolve()}\n")

    # Print useful conversion statistics to verify dataset health
    print("=" * 50)
    print("              CONVERSION STATISTICS             ")
    print("=" * 50)
    print(f"Number of JSON files processed : {processed_files_count}") # Expected: 48 subjects
    print(f"Number of JSON files that failed: {len(failed_files)}")      # Expected: 0 failures
    if failed_files:
        print("Failed files detail:")
        for name, err in failed_files:
            print(f"  - {name}: {err}")
    print(f"Total CSV rows                 : {len(df):,}")             # Expected: 284,247 rows
    
    # Compute descriptive dataset sanity metrics
    unique_videos = df["video"].nunique(dropna=True)                # Expected: 20 unique videos
    unique_hmds = df["hmd"].nunique(dropna=True)                    # Expected: 1 ('vive')
    min_ts = df["timestamp"].min()                                  # Minimum timestamp (0.0s)
    max_ts = df["timestamp"].max()                                  # Maximum timestamp (~30.096s)

    print(f"Number of unique videos        : {unique_videos}")
    print(f"Number of unique HMD values    : {unique_hmds} ({list(df['hmd'].dropna().unique())})")
    print(f"Minimum timestamp (sec)        : {min_ts}")
    print(f"Maximum timestamp (sec)        : {max_ts}")
    print("-" * 50)
    
    # Audit for missing / null values across each extracted column
    print("Number of missing values in each column:")
    missing_counts = df.isna().sum()
    for col, count in missing_counts.items():
        print(f"  - {col:<10}: {count}")                            # Expected: 0 across all columns
    print("=" * 50)

    return df


if __name__ == "__main__":
    # Determine input directory (defaults to Head_rotation directory)
    project_dir = Path(__file__).parent.resolve()
    input_dir = project_dir / "Head_rotation"
    
    # Fallback to project root if Head_rotation subfolder does not exist
    if not input_dir.exists():
        input_dir = project_dir

    # Output file path for consolidated CSV
    output_csv = project_dir / "combined_viewport_data.csv"

    # Execute data conversion
    df = combine_json_to_csv(input_dir, output_csv)
