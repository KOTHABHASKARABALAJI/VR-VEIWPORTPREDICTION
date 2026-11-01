"""
Script: analyze_unique_videos.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Purpose:
    Analyzes the consolidated viewport dataset ('combined_viewport_data.csv') to identify all unique
    360° videos present, counts the total recorded orientation samples per video, sorts the videos
    using natural numerical ordering (e.g., '2.mp4' precedes '10.mp4'), exports the summary to
    'unique_videos.csv', and prints a formatted diagnostic report.

Role in Workflow:
    Stage 4 (Exploratory Data Analysis & Validation): Confirms that all 20 videos from the AVTrack360
    dataset are represented with balanced sample counts (~14,140 to 14,300 samples each), verifying that
    no video was truncated or omitted during data combination.
"""

# Standard library module for object-oriented filesystem paths
from pathlib import Path

# Third-party tabular data manipulation library used for aggregation, sorting, and CSV export
import pandas as pd


def analyze_unique_videos(csv_path: str | Path, output_csv_path: str | Path):
    """
    Computes sample distributions across unique videos in combined_viewport_data.csv,
    applies natural numerical sorting, saves results to unique_videos.csv, and prints a summary report.

    Parameters:
        csv_path (str | Path): Path to the input consolidated CSV file (combined_viewport_data.csv).
        output_csv_path (str | Path): Destination path for the exported summary CSV (unique_videos.csv).

    Returns:
        None: Exports 'unique_videos.csv' and displays diagnostic metrics in the console.
    """
    # Convert input string paths into pathlib.Path objects
    path = Path(csv_path)
    output_path = Path(output_csv_path)

    # Sanity check: Ensure input CSV exists before attempting analysis
    if not path.exists():
        print(f"[ERROR] Original CSV file not found: {path}")
        return

    # 1. Load combined_viewport_data.csv using pandas
    print(f"Loading {path.name}...")
    df = pd.read_csv(path)
    total_rows = len(df) # Expected: 284,247 rows

    # 2 & 5. Count samples per video
    # Group by 'video' column and count rows per video group, renaming count column to 'total_samples'
    video_counts = df.groupby("video").size().reset_index(name="total_samples")

    # Natural sorting helper: sort 2.mp4 before 10.mp4
    # In standard alphabetical sorting, '10.mp4' comes before '2.mp4' ('1' < '2').
    # By extracting the numeric stem (int('2') vs int('10')), natural numerical order is achieved.
    def sort_key(val):
        stem = str(val).split('.')[0]
        # Return tuple: (integer value for numerical order, original string fallback for non-digits)
        return (int(stem) if stem.isdigit() else float('inf'), str(val))

    # Apply the natural sort key function to create a temporary sorting column
    video_counts["sort_key"] = video_counts["video"].apply(sort_key)
    
    # Sort by the numeric key, drop the temporary helper column, and reset DataFrame index
    video_counts = video_counts.sort_values(by="sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

    # 3. Total unique videos: Count the number of rows in the aggregated table (Expected: 20)
    total_unique_videos = len(video_counts)

    # 4. Sorted list of unique video names
    unique_video_list = video_counts["video"].tolist()

    # 6. Save results to unique_videos.csv without row indices
    video_counts.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[SUCCESS] Saved unique videos summary to {output_path.resolve()}\n")

    # Display requested outputs in a clean, human-readable terminal report
    print("=" * 60)
    print("                 UNIQUE VIDEOS SUMMARY REPORT            ")
    print("=" * 60)
    print(f"Total rows in original CSV  : {total_rows:,}")
    print(f"Total unique videos         : {total_unique_videos}")
    print("-" * 60)
    print("Complete unique video list  :")
    print(", ".join(unique_video_list))
    print("-" * 60)
    print("First 10 rows of unique_videos.csv:")
    print(video_counts.head(10).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    # Resolve the project root directory dynamically (two levels up from src/preprocessing)
    project_root = Path(__file__).resolve().parents[2]
    
    # Define source CSV and target summary CSV paths
    csv_file = project_root / "data" / "processed" / "combined_viewport_data.csv"
    output_file = project_root / "data" / "processed" / "unique_videos.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Execute unique video analysis routine
    analyze_unique_videos(csv_file, output_file)
