import json

# Standard library module for object-oriented filesystem paths
from pathlib import Path
import pandas as pd

def combine_json_to_csv(input_folder: str | Path, output_csv_path: str | Path) -> pd.DataFrame:
    """
    Reads all JSON files in input_folder, extracts viewport trajectory data,
    and writes to output_csv_path.
    """
    folder = Path(input_folder)
    output_file = Path(output_csv_path)

    # Find all .json files in input folder
    json_files = sorted(list(folder.glob("*.json")))
    
    if not json_files:
        # Check subdirectories if no json files found directly
        json_files = sorted(list(folder.rglob("*.json")))

    print(f"Found {len(json_files)} JSON file(s) in '{folder}'. Starting conversion...\n")

    records = []
    processed_files_count = 0
    failed_files = []

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                content = json.load(f)

            if not isinstance(content, dict) or "data" not in content:
                raise ValueError(f"Missing 'data' root key or invalid JSON structure in {json_path.name}")

            data_list = content.get("data", [])
            if not isinstance(data_list, list):
                raise ValueError(f"'data' field is not a list in {json_path.name}")

            for data_item in data_list:
                if not isinstance(data_item, dict):
                    continue

                video = data_item.get("filename")
                hmd = data_item.get("hmd")
                points = data_item.get("pitch_yaw_roll_data_hmd", [])

                if not isinstance(points, list):
                    continue

                for point in points:
                    if not isinstance(point, dict):
                        continue

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
            failed_files.append((json_path.name, str(e)))
            print(f"[WARNING] Error processing '{json_path.name}': {e}. Skipping file.")

    # Create DataFrame
    df = pd.DataFrame(records, columns=["video", "hmd", "timestamp", "pitch", "yaw", "roll"])

    # Save to CSV without modifying numerical values or dropping duplicates
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"[SUCCESS] CSV successfully saved to: {output_file.resolve()}\n")

    # Print useful statistics
    print("=" * 50)
    print("              CONVERSION STATISTICS             ")
    print("=" * 50)
    print(f"Number of JSON files processed : {processed_files_count}")
    print(f"Number of JSON files that failed: {len(failed_files)}")
    if failed_files:
        print("Failed files detail:")
        for name, err in failed_files:
            print(f"  - {name}: {err}")
    print(f"Total CSV rows                 : {len(df):,}")
    
    unique_videos = df["video"].nunique(dropna=True)
    unique_hmds = df["hmd"].nunique(dropna=True)
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()

    print(f"Number of unique videos        : {unique_videos}")
    print(f"Number of unique HMD values    : {unique_hmds} ({list(df['hmd'].dropna().unique())})")
    print(f"Minimum timestamp (sec)        : {min_ts}")
    print(f"Maximum timestamp (sec)        : {max_ts}")
    print("-" * 50)
    print("Number of missing values in each column:")
    missing_counts = df.isna().sum()
    for col, count in missing_counts.items():
        print(f"  - {col:<10}: {count}")
    print("=" * 50)

    return df


if __name__ == "__main__":
    # Determine project root and input directory (defaults to data/raw/Head_rotation)
    project_root = Path(__file__).resolve().parents[2]
    input_dir = project_root / "data" / "raw" / "Head_rotation"
    
    if not input_dir.exists():
        input_dir = project_root / "Head_rotation"

    output_csv = project_root / "data" / "processed" / "combined_viewport_data.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = combine_json_to_csv(input_dir, output_csv)
