"""
batch_process.py
================
Process a folder of check images and export all results to a single CSV / JSON.

Usage:
  python batch_process.py --folder ./checks --output results.csv
  python batch_process.py --folder ./checks --output results.json
"""

import argparse
import json
import csv
import os
from pathlib import Path
from check_extractor import extract_check_fields

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def process_folder(folder: str, output: str, preprocess: bool = True):
    folder_path = Path(folder)
    images = [
        f for f in folder_path.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not images:
        print(f"No supported images found in {folder}")
        return

    print(f"Found {len(images)} check image(s) to process...\n")
    results = []

    for img_path in sorted(images):
        print(f"  → {img_path.name} ...", end=" ")
        try:
            data = extract_check_fields(str(img_path), preprocess=preprocess)
            data["file"] = img_path.name
            data["error"] = ""
            print("✓")
        except Exception as e:
            data = {"file": img_path.name, "error": str(e)}
            print(f"✗ {e}")
        results.append(data)

    output_path = Path(output)

    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ JSON saved: {output_path}")

    else:  # CSV
        # Flatten nested owner dict
        flat_results = []
        for r in results:
            flat = {
                "file": r.get("file", ""),
                "error": r.get("error", ""),
                "owner_name": r.get("owner", {}).get("name", ""),
                "owner_address1": r.get("owner", {}).get("address_line1", ""),
                "owner_address2": r.get("owner", {}).get("address_line2", ""),
                "owner_phone": r.get("owner", {}).get("phone", ""),
                "date": r.get("date", ""),
                "payee": r.get("payee", ""),
                "amount_numeric": r.get("amount_numeric", ""),
                "amount_written": r.get("amount_written", ""),
                "memo": r.get("memo", ""),
                "routing_number": r.get("routing_number", ""),
                "account_number": r.get("account_number", ""),
                "check_number": r.get("check_number", ""),
            }
            flat_results.append(flat)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())
            writer.writeheader()
            writer.writerows(flat_results)
        print(f"\n✅ CSV saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch-process a folder of check images."
    )
    parser.add_argument("--folder", required=True,
                        help="Folder containing check images")
    parser.add_argument("--output", default="check_results.csv",
                        help="Output file: .csv or .json (default: check_results.csv)")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip image pre-processing")
    args = parser.parse_args()

    process_folder(args.folder, args.output, preprocess=not args.no_preprocess)


if __name__ == "__main__":
    main()
