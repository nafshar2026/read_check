#!/usr/bin/env python3
"""
main.py – Command-line interface for the check-reader application.

Usage
-----
    python main.py <image_path> [--tesseract-cmd <path>] [--json]

Examples
--------
    python main.py check.jpg
    python main.py check.png --json
    python main.py check.tiff --tesseract-cmd /usr/local/bin/tesseract
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from check_reader import CheckReader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_reader",
        description="Extract information from a handwritten check image.",
    )
    parser.add_argument(
        "image",
        metavar="IMAGE_PATH",
        help="Path to the check image (JPEG, PNG, TIFF, etc.)",
    )
    parser.add_argument(
        "--tesseract-cmd",
        metavar="PATH",
        default=None,
        help="Path to the Tesseract executable (optional if Tesseract is on PATH).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON instead of human-readable text.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        return 1

    reader = CheckReader(image_path, tesseract_cmd=args.tesseract_cmd)
    data = reader.read()

    if args.output_json:
        output = data.to_dict()
        if data.errors:
            output["warnings"] = data.errors
        print(json.dumps(output, indent=2))
    else:
        print("=" * 50)
        print("CHECK INFORMATION")
        print("=" * 50)
        print(f"Name / Address   :\n{data.name_address or '(not found)'}")
        print("-" * 50)
        print(f"Amount           : {data.amount or '(not found)'}")
        print(f"Routing Number   : {data.routing_number or '(not found)'}")
        print(f"Account Number   : {data.account_number or '(not found)'}")
        print("=" * 50)
        if data.errors:
            print("\nWarnings / errors encountered:")
            for err in data.errors:
                print(f"  - {err}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
