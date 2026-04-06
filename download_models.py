"""
download_models.py
==================
Pre-downloads all PaddleOCR model weights into the local ./models/ directory.

Run this ONCE on a machine with internet access before deploying to Azure AI
Foundry or any other environment where outbound model downloads may be blocked.

The downloaded models are then committed alongside the code (or copied to your
Azure ML datastore / Docker image), and check_extractor.py will use them
automatically via the MODEL_DIR environment variable or --model-dir flag.

Usage:
  python download_models.py                    # downloads to ./models/
  python download_models.py --dir /path/to/models
  python download_models.py --list             # show what will be downloaded
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# ── Disable connectivity check & oneDNN before any paddle import ─────────────
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_DISABLE_ONEDNN", "1")

# ── Model catalogue ───────────────────────────────────────────────────────────
# These are the models used by PaddleOCR for English check OCR.
# det  = text detection (finds where text is)
# rec  = text recognition (reads what the text says)
# cls  = angle classifier (corrects upside-down text) — v2 only

MODELS_V2 = {
    "det": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
    "rec": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar",
    "cls": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
}

MODELS_V3 = {
    # v3 downloads models automatically to ~/.paddlex/official_models/
    # This script triggers that download by initialising the pipeline.
    "note": "PaddleOCR v3 manages its own model cache under ~/.paddlex/official_models/"
}


def _paddleocr_version():
    try:
        import paddleocr
        parts = getattr(paddleocr, "__version__", "0.0.0").split(".")
        return tuple(int(p) for p in parts if p.isdigit())
    except Exception:
        return (0, 0, 0)


def download_v2_models(dest_dir: Path):
    """Download PaddleOCR v2 model tarballs and extract them."""
    import requests, tarfile

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📦 Downloading PaddleOCR v2 models → {dest_dir}\n")

    for name, url in MODELS_V2.items():
        model_name = url.split("/")[-1].replace(".tar", "")
        model_path = dest_dir / model_name

        if model_path.exists():
            print(f"  ✓ {name:4s}  already cached: {model_path}")
            continue

        print(f"  ↓ {name:4s}  {url}")
        tar_path = dest_dir / url.split("/")[-1]

        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(tar_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r       {pct:5.1f}%  {downloaded//1024//1024} MB", end="")
            print()

            print(f"       Extracting...")
            with tarfile.open(tar_path) as tf:
                tf.extractall(dest_dir)
            tar_path.unlink()
            print(f"  ✓ {name:4s}  saved to {model_path}")

        except Exception as e:
            print(f"\n  ✗ {name:4s}  FAILED: {e}")
            if tar_path.exists():
                tar_path.unlink()

    print(f"\n✅ v2 models ready in: {dest_dir}")
    print("\nTo use these local models, set the environment variable:")
    print(f'  export PADDLE_MODEL_DIR="{dest_dir}"     # Linux/macOS')
    print(f'  set PADDLE_MODEL_DIR="{dest_dir}"        # Windows CMD')
    print(f'  $env:PADDLE_MODEL_DIR="{dest_dir}"       # Windows PowerShell')


def warm_up_v3_models(dest_dir: Path):
    """
    For PaddleOCR v3, trigger model download by initialising the OCR pipeline.
    v3 manages its own cache at ~/.paddlex/official_models/ but we also copy
    the cache to dest_dir so it can be bundled for offline deployment.
    """
    print(f"\n📦 Warming up PaddleOCR v3 models (will download if not cached)...\n")

    try:
        from paddleocr import PaddleOCR
        print("  Initialising PaddleOCR pipeline (this triggers model downloads)...")
        ocr = PaddleOCR(lang="en")
        print("  ✓ Models loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to initialise PaddleOCR: {e}")
        sys.exit(1)

    # Locate the paddlex cache
    paddlex_cache = Path.home() / ".paddlex" / "official_models"
    if paddlex_cache.exists():
        print(f"\n  Copying model cache → {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        for item in paddlex_cache.iterdir():
            dst = dest_dir / item.name
            if dst.exists():
                print(f"  ✓ {item.name}  already in dest, skipping")
                continue
            print(f"  ↓ Copying {item.name}...")
            shutil.copytree(str(item), str(dst))
        print(f"\n✅ v3 model cache copied to: {dest_dir}")
    else:
        print(f"\n  ⚠ paddlex cache not found at {paddlex_cache}")
        print("  Models are cached in a different location on this system.")

    print("\nTo use local models in Azure, set:")
    print(f'  export PADDLEX_MODEL_DIR="{dest_dir}"')


def show_model_list():
    version = _paddleocr_version()
    print(f"\nPaddleOCR version: {'.'.join(str(v) for v in version)}")
    if version >= (3, 0, 0):
        print("\nPaddleOCR v3 models (auto-managed by paddlex):")
        print("  • PP-LCNet_x1_0_doc_ori      — document orientation classifier")
        print("  • UVDoc                       — document unwarping")
        print("  • PP-LCNet_x1_0_textline_ori  — text line orientation")
        print("  • PP-OCRv5_server_det         — text detection")
        print("  • en_PP-OCRv5_mobile_rec      — English text recognition")
        print(f"\n  Cache location: {Path.home() / '.paddlex' / 'official_models'}")
    else:
        print("\nPaddleOCR v2 models:")
        for name, url in MODELS_V2.items():
            print(f"  • {name:4s}  {url.split('/')[-1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download PaddleOCR model weights for offline/Azure deployment."
    )
    parser.add_argument(
        "--dir", default="./models",
        help="Directory to save models (default: ./models)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List models that will be downloaded without downloading"
    )
    args = parser.parse_args()

    if args.list:
        show_model_list()
        return

    dest = Path(args.dir).resolve()
    version = _paddleocr_version()
    print(f"PaddleOCR version: {'.'.join(str(v) for v in version)}")

    if version >= (3, 0, 0):
        warm_up_v3_models(dest)
    else:
        download_v2_models(dest)


if __name__ == "__main__":
    main()
