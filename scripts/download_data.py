# CS4241 Manual RAG Chatbot — Data Download Script
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141
#
# Downloads:
#   - Ghana_Election_Result.csv  from GitHub
#   - 2025-Budget-Statement.pdf  from mofep.gov.gh
#
# Usage: python scripts/download_data.py

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from tqdm import tqdm
from src.config import CSV_PATH, PDF_PATH, CSV_URL, PDF_URL
from src.logger import log_stage, banner


def download(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        log_stage("DOWNLOAD_SKIP", {"file": label, "reason": "already exists", "path": str(dest)})
        print(f"  [skip] {label} already at {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {label} …")
    log_stage("DOWNLOAD_START", {"url": url, "dest": str(dest)})

    headers = {"User-Agent": "Mozilla/5.0 (compatible; AcityRAG/1.0)"}
    resp = requests.get(url, headers=headers, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=label, ncols=70
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    size_kb = dest.stat().st_size / 1024
    log_stage("DOWNLOAD_DONE", {"file": label, "size_kb": round(size_kb, 1), "path": str(dest)})
    print(f"  Done — {size_kb:.0f} KB saved to {dest}")


def main() -> None:
    banner("DOWNLOAD DATA SOURCES")
    download(CSV_URL, CSV_PATH, "Ghana_Election_Result.csv")
    download(PDF_URL, PDF_PATH, "2025-Budget-Statement.pdf")
    print("\nAll data ready in data/raw/")


if __name__ == "__main__":
    main()
