import urllib.request
import time
from pathlib import Path

def download_src_info():
    project_dir = Path(__file__).parent.resolve()
    src_info_dir = project_dir / "SRC_info"
    
    # Create SRC_info folder if it does not exist
    src_info_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {src_info_dir.resolve()}\n")

    base_url = "https://raw.githubusercontent.com/acmmmsys/2018-AVTrack360/master/AVTrack360_dataset/SRC_info/"
    files_to_download = [f"{i}.json" for i in range(1, 21)]

    downloaded = 0
    failed = []

    for filename in files_to_download:
        url = base_url + filename
        dest_path = src_info_dir / filename
        
        try:
            print(f"Downloading {filename} from {url}...")
            # Set user-agent header to avoid potential rate-limiting/blocking
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
                out_file.write(response.read())
            downloaded += 1
            time.sleep(0.1) # polite delay
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            failed.append((filename, str(e)))

    print("\n" + "=" * 60)
    print("                     VERIFICATION REPORT                  ")
    print("=" * 60)
    
    # Verify that exactly 20 JSON files exist
    existing_json_files = sorted(list(src_info_dir.glob("*.json")), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
    total_found = len(existing_json_files)

    print(f"Total downloaded successfully : {downloaded}")
    print(f"Total failed downloads        : {len(failed)}")
    print(f"Total JSON files in SRC_info   : {total_found}")
    print("-" * 60)
    
    if total_found == 20:
        print("[SUCCESS] Exactly 20 JSON files exist in SRC_info!")
    else:
        print(f"[WARNING] Expected 20 files, but found {total_found} files.")

    print("\nExisting file list in SRC_info:")
    for f in existing_json_files:
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")
    print("=" * 60)

if __name__ == "__main__":
    download_src_info()
