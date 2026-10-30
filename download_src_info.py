"""
Script: download_src_info.py
Project: Machine Learning-Based Viewport Tile Prediction and Adaptive Quality Allocation for 360° Video Streaming

Purpose:
    Automates the download of technical source video metadata files from the public AVTrack360
    benchmark repository on GitHub. Specifically, it downloads 20 JSON files (1.json to 20.json)
    containing FFprobe stream and container metadata for each 360° video used in the study.

Role in Workflow:
    Stage 1 (Dataset Acquisition): Fetches video encoding information (resolution, duration,
    frame rates, bitrates) that are subsequently parsed by 'extract_video_metadata.py'.
"""

# Standard library module for HTTP requests; avoids external dependencies like 'requests'
import urllib.request

# Standard library module used to introduce polite pacing delays between network requests
import time

# Standard library module for resilient filesystem path management across operating systems
from pathlib import Path


def download_src_info():
    """
    Downloads 20 video source metadata JSON files (1.json to 20.json) from the AVTrack360 GitHub
    repository over HTTPS, saves them into the 'SRC_info' directory, and verifies download completeness.

    Parameters:
        None

    Returns:
        None: Files are written directly to disk and a summary report is printed to console.
    """
    # Dynamically determine the project root directory where this script resides
    project_dir = Path(__file__).parent.resolve()

    # Define the destination directory for downloaded video source metadata
    src_info_dir = project_dir / "SRC_info"
    
    # Create the 'SRC_info' folder if it does not already exist
    # 'parents=True' creates intermediate directories if needed; 'exist_ok=True' prevents errors if it exists
    src_info_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {src_info_dir.resolve()}\n")

    # Base URL pointing to the raw JSON metadata files in the official AVTrack360 GitHub repository
    base_url = "https://raw.githubusercontent.com/acmmmsys/2018-AVTrack360/master/AVTrack360_dataset/SRC_info/"

    # Generate the target file list: 1.json, 2.json, ..., 20.json (20 source videos in the dataset)
    files_to_download = [f"{i}.json" for i in range(1, 21)]

    # Tracking counters and error log for download session diagnostics
    downloaded = 0
    failed = []

    # Iterate through each filename and fetch it over the network
    for filename in files_to_download:
        url = base_url + filename
        dest_path = src_info_dir / filename
        
        try:
            print(f"Downloading {filename} from {url}...")

            # Construct an HTTP Request with a browser User-Agent header
            # This prevents raw GitHub servers from blocking Python's default urllib user-agent
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

            # Open HTTP connection, read incoming binary payload, and save to local disk
            with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
                out_file.write(response.read())

            downloaded += 1

            # Introduce a short 100ms pause to practice polite scraping and avoid network rate-limiting
            time.sleep(0.1) # polite delay

        except Exception as e:
            # Capture any network timeout, HTTP 404/500, or I/O errors without terminating the loop
            print(f"[ERROR] Failed to download {filename}: {e}")
            failed.append((filename, str(e)))

    # Display post-download diagnostic summary
    print("\n" + "=" * 60)
    print("                     VERIFICATION REPORT                  ")
    print("=" * 60)
    
    # Scan the destination directory to discover all present JSON files
    # Natural sort key ensures '2.json' appears before '10.json' based on numeric stem value
    existing_json_files = sorted(
        list(src_info_dir.glob("*.json")),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.name
    )
    total_found = len(existing_json_files)

    print(f"Total downloaded successfully : {downloaded}")
    print(f"Total failed downloads        : {len(failed)}")
    print(f"Total JSON files in SRC_info   : {total_found}")
    print("-" * 60)
    
    # Confirm that all 20 video metadata files exist
    if total_found == 20:
        print("[SUCCESS] Exactly 20 JSON files exist in SRC_info!")
    else:
        print(f"[WARNING] Expected 20 files, but found {total_found} files.")

    # List each file and its size on disk in formatted bytes
    print("\nExisting file list in SRC_info:")
    for f in existing_json_files:
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")
    print("=" * 60)


if __name__ == "__main__":
    # Execute the download routine when invoked as a standalone script
    download_src_info()
