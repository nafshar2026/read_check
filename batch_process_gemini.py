"""
batch_process_gemini.py
=======================
Batch-process a folder of check images using Google Gemini Vision API (free).

Usage:
  python batch_process_gemini.py --folder ./images --output results.csv
  python batch_process_gemini.py --folder ./images --output results.json
"""

import json
import csv
import time
import argparse
from pathlib import Path
from check_extractor_gemini import extract_with_gemini, validate_result

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Gemini free tier allows 15 requests/minute — stay safely under that
REQUEST_DELAY_SECONDS = 4.0


def process_folder(folder: str, output: str):
    folder_path = Path(folder)
    images = sorted([
        f for f in folder_path.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not images:
        print(f"No supported images found in {folder}")
        return

    print(f"Found {len(images)} check image(s)\n")
    results = []

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name} ...", end=" ", flush=True)
        try:
            data = extract_with_gemini(str(img_path))
            warnings = validate_result(data)
            data["file"]     = img_path.name
            data["error"]    = ""
            data["warnings"] = "; ".join(warnings)
            print("✓")
        except Exception as e:
            data = {"file": img_path.name, "error": str(e), "warnings": ""}
            print(f"✗  {e}")

        results.append(data)

        # Respect Gemini free tier rate limit (15 req/min)
        if i < len(images):
            time.sleep(REQUEST_DELAY_SECONDS)

    output_path = Path(output)

    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ JSON saved: {output_path}")
    else:
        flat = []
        for r in results:
            flat.append({
                "file":           r.get("file", ""),
                "error":          r.get("error", ""),
                "warnings":       r.get("warnings", ""),
                "owner_name":     r.get("owner_name", ""),
                "address_line1":  r.get("address_line1", ""),
                "address_line2":  r.get("address_line2", ""),
                "phone":          r.get("phone", ""),
                "bank_name":      r.get("bank_name", ""),
                "date":           r.get("date", ""),
                "payee":          r.get("payee", ""),
                "amount_numeric": r.get("amount_numeric", ""),
                "amount_written": r.get("amount_written", ""),
                "memo":           r.get("memo", ""),
                "routing_number": r.get("routing_number", ""),
                "account_number": r.get("account_number", ""),
                "check_number":   r.get("check_number", ""),
                "micr_raw":       r.get("micr_raw", ""),
            })

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat[0].keys())
            writer.writeheader()
            writer.writerows(flat)
        print(f"\n✅ CSV saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch-process checks using Google Gemini Vision (free)."
    )
    parser.add_argument("--folder",  required=True,
                        help="Folder containing check images")
    parser.add_argument("--output",  default="check_results.csv",
                        help="Output file (.csv or .json)")
    args = parser.parse_args()
    process_folder(args.folder, args.output)


if __name__ == "__main__":
    main()
