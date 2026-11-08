from pathlib import Path
import pandas as pd

def analyze_unique_videos(csv_path: str | Path, output_csv_path: str | Path):
    path = Path(csv_path)
    output_path = Path(output_csv_path)

    if not path.exists():
        print(f"[ERROR] Original CSV file not found: {path}")
        return

    # 1. Load combined_viewport_data.csv using pandas
    print(f"Loading {path.name}...")
    df = pd.read_csv(path)
    total_rows = len(df)

    # 2 & 5. Count samples per video
    # Group by 'video' and count rows
    video_counts = df.groupby("video").size().reset_index(name="total_samples")

    # Natural sorting helper: sort 2.mp4 before 10.mp4
    def sort_key(val):
        stem = str(val).split('.')[0]
        return (int(stem) if stem.isdigit() else float('inf'), str(val))

    video_counts["sort_key"] = video_counts["video"].apply(sort_key)
    video_counts = video_counts.sort_values(by="sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

    # 3. Total unique videos
    total_unique_videos = len(video_counts)

    # 4. Sorted list of unique video names
    unique_video_list = video_counts["video"].tolist()

    # 6. Save results to unique_videos.csv
    video_counts.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[SUCCESS] Saved unique videos summary to {output_path.resolve()}\n")

    # Display requested outputs
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
    project_dir = Path(__file__).parent.resolve()
    csv_file = project_dir / "combined_viewport_data.csv"
    output_file = project_dir / "unique_videos.csv"

    analyze_unique_videos(csv_file, output_file)
