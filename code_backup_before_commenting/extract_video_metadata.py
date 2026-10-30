import json
from pathlib import Path
import pandas as pd

def extract_metadata():
    project_dir = Path(__file__).parent.resolve()
    src_info_dir = project_dir / "SRC_info"
    output_csv = project_dir / "video_metadata.csv"

    # Prepare files 1.json to 20.json in numerical order
    json_files = [src_info_dir / f"{i}.json" for i in range(1, 21)]

    records = []

    for json_path in json_files:
        video_id = f"{json_path.stem}.mp4"
        
        if not json_path.exists():
            print(f"[WARNING] File not found: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        streams = data.get("streams", [])
        fmt = data.get("format", {})

        # Find video stream and audio stream
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

        # Extract video stream metadata
        width = video_stream.get("width")
        height = video_stream.get("height")
        resolution = f"{width}x{height}" if width is not None and height is not None else None
        
        display_aspect_ratio = video_stream.get("display_aspect_ratio")
        r_frame_rate = video_stream.get("r_frame_rate")
        avg_frame_rate = video_stream.get("avg_frame_rate")
        duration = video_stream.get("duration")
        nb_frames = video_stream.get("nb_frames")
        video_codec = video_stream.get("codec_name")
        pixel_format = video_stream.get("pix_fmt")

        # Extract format / audio metadata if available
        container_format = fmt.get("format_name")
        file_size = fmt.get("size")
        overall_bitrate = fmt.get("bit_rate")
        
        audio_codec = audio_stream.get("codec_name") if audio_stream else None
        audio_bitrate = audio_stream.get("bit_rate") if audio_stream else None

        records.append({
            "video_id": video_id,
            "width": width,
            "height": height,
            "resolution": resolution,
            "display_aspect_ratio": display_aspect_ratio,
            "r_frame_rate": r_frame_rate,
            "avg_frame_rate": avg_frame_rate,
            "duration": duration,
            "nb_frames": nb_frames,
            "video_codec": video_codec,
            "pixel_format": pixel_format,
            "container_format": container_format,
            "file_size": file_size,
            "overall_bitrate": overall_bitrate,
            "audio_codec": audio_codec,
            "audio_bitrate": audio_bitrate
        })

    df = pd.DataFrame(records)
    
    # Save CSV
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[SUCCESS] Metadata extracted and saved to: {output_csv.resolve()}\n")

    # Print full 20-row table
    print("=" * 120)
    print("                                      VIDEO METADATA TABLE (20 ROWS)                                      ")
    print("=" * 120)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 1000)
    print(df.to_string(index=False))
    print("=" * 120)

    # Report missing fields
    print("\n" + "=" * 60)
    print("                      MISSING FIELDS REPORT                      ")
    print("=" * 60)
    missing_counts = df.isna().sum()
    total_rows = len(df)
    
    has_missing = False
    for col, count in missing_counts.items():
        if count > 0:
            has_missing = True
            missing_videos = df[df[col].isna()]["video_id"].tolist()
            print(f"  - Column '{col}': {count}/{total_rows} missing values (Missing in: {', '.join(missing_videos)})")
        else:
            print(f"  - Column '{col}': 0 missing values (100% complete)")

    if not has_missing:
        print("\n[INFO] All columns are 100% complete with 0 missing fields across all 20 videos.")
    else:
        print("\n[INFO] Note: Missing audio_codec or audio_bitrate indicates videos that do not contain an audio stream.")
    print("=" * 60)

if __name__ == "__main__":
    extract_metadata()
