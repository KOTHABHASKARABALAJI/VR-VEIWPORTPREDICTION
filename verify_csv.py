from pathlib import Path
import pandas as pd

def verify_csv(csv_path: str | Path):
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return

    print(f"Reading CSV file for verification: {path.name}...\n")
    df = pd.read_csv(path)

    print("=" * 60)
    print("                   VERIFICATION REPORT                   ")
    print("=" * 60)
    print(f"DataFrame Shape (Rows, Columns): {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("-" * 60)
    print("First 10 Rows:")
    print(df.head(10).to_string(index=False))
    print("=" * 60)

if __name__ == "__main__":
    project_dir = Path(__file__).parent.resolve()
    csv_file = project_dir / "combined_viewport_data.csv"
    verify_csv(csv_file)
