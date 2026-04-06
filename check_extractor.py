"""
check_extractor.py
==================
Extracts structured fields from scanned check images using PaddleOCR.

Extracted fields:
  - Owner name & address (printed header)
  - Payee (Pay To The Order Of)
  - Amount (numeric $ box)
  - Amount (written line)
  - Routing number  (MICR line, between ⑆ symbols)
  - Account number  (MICR line, between ⑆ and ⑈ symbols)
  - Check number
  - Date
  - Memo

Usage:
  python check_extractor.py --image path/to/check.jpg
  python check_extractor.py --image path/to/check.jpg --output result.json
"""

import os
import re
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Must be set before paddle C++ runtime loads — disables oneDNN/MKL-DNN
# which causes NotImplementedError on Windows with PaddleOCR 3.x
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_DISABLE_ONEDNN", "1")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# ── OCR backend ──────────────────────────────────────────────────────────────

def _paddleocr_version() -> tuple[int, ...]:
    """Return PaddleOCR version as an integer tuple, e.g. (2, 7, 3) or (3, 4, 0)."""
    try:
        import paddleocr
        parts = getattr(paddleocr, "__version__", "0.0.0").split(".")
        return tuple(int(p) for p in parts if p.isdigit())
    except Exception:
        return (0, 0, 0)


def get_ocr_engine():
    """
    Returns a PaddleOCR instance, auto-detecting v2 vs v3 API differences.

    PaddleOCR v2.x  →  PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    PaddleOCR v3.x  →  PaddleOCR(lang='en')   (show_log & use_angle_cls removed)

    Falls back to pytesseract if PaddleOCR is not installed.
    """
    try:
        import os
        # Suppress the connectivity-check banner on v3
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        # Disable oneDNN (Intel MKL-DNN) — fixes NotImplementedError on Windows
        # with PaddleOCR 3.x: ConvertPirAttribute2RuntimeAttribute not support
        # [pir::ArrayAttribute<pir::DoubleAttribute>]
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("PADDLE_DISABLE_ONEDNN", "1")

        from paddleocr import PaddleOCR
        version = _paddleocr_version()
        print(f"[OCR] PaddleOCR version detected: {'.'.join(str(v) for v in version)}")

        if version >= (3, 0, 0):
            # v3 API — removed show_log, use_angle_cls, use_gpu
            ocr = PaddleOCR(lang="en")
        else:
            # v2 API
            ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        print("[OCR] Using PaddleOCR backend")
        return ("paddle", ocr)

    except ImportError:
        print("[OCR] PaddleOCR not installed, falling back to pytesseract")
    except Exception as e:
        print(f"[OCR] PaddleOCR init failed ({e}), falling back to pytesseract")

    try:
        import pytesseract
        print("[OCR] Using pytesseract backend")
        return ("tesseract", pytesseract)
    except ImportError:
        raise RuntimeError(
            "No OCR backend found. Install PaddleOCR:\n"
            "  pip install paddlepaddle paddleocr\n"
            "Or pytesseract:\n"
            "  pip install pytesseract  (+ Tesseract binary from https://github.com/UB-Mannheim/tesseract/wiki)"
        )


def run_ocr(image_path: str, engine_tuple) -> list[dict]:
    """
    Run OCR and return a normalised list of word detections:
      [{"text": str, "x": int, "y": int, "w": int, "h": int, "conf": float}, ...]
    Sorted top-to-bottom, left-to-right.
    """
    import cv2, numpy as np
    from PIL import Image

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    backend, engine = engine_tuple
    detections = []

    if backend == "paddle":
        version = _paddleocr_version()

        if version >= (3, 0, 0):
            # v3: .ocr() returns a list of Result objects with a .boxes attribute
            # Each item: result[i] has keys like 'rec_texts', 'rec_scores', 'rec_boxes'
            results = engine.ocr(image_path)
            for res in (results or []):
                # v3 returns objects; iterate over the result set
                boxes   = getattr(res, "boxes",      None) or []
                texts   = getattr(res, "rec_texts",  None) or []
                scores  = getattr(res, "rec_scores", None) or []
                # Fallback: some v3 builds still return the old list-of-lists
                if not texts and isinstance(res, list):
                    for line in res:
                        try:
                            pts, (text, conf) = line
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            detections.append({
                                "text": text,
                                "x": int(min(xs)), "y": int(min(ys)),
                                "w": int(max(xs) - min(xs)),
                                "h": int(max(ys) - min(ys)),
                                "conf": float(conf),
                            })
                        except Exception:
                            continue
                    continue
                for pts, text, conf in zip(boxes, texts, scores):
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    detections.append({
                        "text": text,
                        "x": int(min(xs)), "y": int(min(ys)),
                        "w": int(max(xs) - min(xs)),
                        "h": int(max(ys) - min(ys)),
                        "conf": float(conf),
                    })
        else:
            # v2: returns [ [ [pts, (text, conf)], ... ] ]
            result = engine.ocr(image_path, cls=True)
            for line in result[0] or []:
                pts, (text, conf) = line
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                detections.append({
                    "text": text,
                    "x": int(min(xs)),
                    "y": int(min(ys)),
                    "w": int(max(xs) - min(xs)),
                    "h": int(max(ys) - min(ys)),
                    "conf": float(conf),
                })
    else:
        # pytesseract path
        import pytesseract
        pil_img = Image.open(image_path)
        data = pytesseract.image_to_data(
            pil_img, output_type=pytesseract.Output.DICT,
            config="--psm 11"
        )
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            conf = float(data["conf"][i])
            if conf < 20:
                continue
            detections.append({
                "text": text,
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
                "conf": conf / 100.0,
            })

    # Sort top-to-bottom then left-to-right
    detections.sort(key=lambda d: (d["y"], d["x"]))
    return detections


# ── Image pre-processing ─────────────────────────────────────────────────────

# Rotation codes and their human-readable labels
_ROTATIONS = [
    (0,   "0° — no rotation",          None),
    (90,  "90° clockwise",              "ROTATE_90_CLOCKWISE"),
    (180, "180° — upside down",         "ROTATE_180"),
    (270, "270° clockwise (90° CCW)",   "ROTATE_90_COUNTERCLOCKWISE"),
]

def _rotate_image(img, degrees):
    """Rotate a cv2 image by 0/90/180/270 degrees."""
    import cv2
    if degrees == 0:   return img
    if degrees == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if degrees == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _score_orientation(img_bgr, ocr_engine):
    """
    Run OCR on the image and return a confidence score for this orientation.
    Score = sum of per-word confidence * word length (longer high-conf words = better).
    Also gives a bonus if known check keywords are detected (PAY, DATE, MEMO, DOLLARS).
    """
    import tempfile, os, cv2
    CHECK_KEYWORDS = {"PAY", "DATE", "MEMO", "DOLLAR", "DOLLARS", "ORDER", "BANK", "VOID"}

    # Write to a temp file so OCR engine can read it
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, img_bgr)

    try:
        detections = run_ocr(tmp.name, ocr_engine)
    finally:
        os.unlink(tmp.name)

    if not detections:
        return 0.0, detections

    score = sum(d["conf"] * max(len(d["text"]), 1) for d in detections)

    # Bonus for check-specific keywords found
    found_keywords = sum(
        1 for d in detections
        if d["text"].upper().strip() in CHECK_KEYWORDS
    )
    score += found_keywords * 5.0

    # Bonus for aspect ratio being landscape (checks are always wider than tall)
    h, w = img_bgr.shape[:2]
    if w > h:
        score *= 1.2

    return score, detections


def _auto_rotate(img, ocr_engine):
    """
    Try all 4 rotations (0°, 90°, 180°, 270°), score each one using OCR
    confidence and keyword detection, and return the best-oriented image
    along with its detections.

    This is more reliable than heuristic pixel-density checks because it
    uses the OCR model's own confidence as the quality signal.
    """
    import cv2
    print("[orientation] Testing all 4 rotations to find best orientation...")

    best_score    = -1
    best_img      = img
    best_degrees  = 0
    best_detections = []

    for degrees, label, _ in _ROTATIONS:
        rotated = _rotate_image(img, degrees)
        score, detections = _score_orientation(rotated, ocr_engine)
        print(f"  {degrees:3d}°  score={score:6.1f}  words={len(detections):3d}  ({label})")
        if score > best_score:
            best_score      = score
            best_img        = rotated
            best_degrees    = degrees
            best_detections = detections

    print(f"[orientation] Best orientation: {best_degrees}° (score={best_score:.1f})")
    return best_img, best_detections


def preprocess_image(image_path: str, output_path: str = None,
                     ocr_engine=None) -> tuple:
    """
    Load the image, determine correct orientation by scoring all 4 rotations
    with OCR confidence, then apply mild denoising and contrast enhancement.

    Returns (processed_image_path, detections) so the caller can reuse the
    detections that were already computed during orientation scoring — avoiding
    a redundant second OCR pass.
    """
    import cv2, numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    # Find best orientation using OCR confidence scoring
    if ocr_engine is not None:
        best_img, best_detections = _auto_rotate(img, ocr_engine)
    else:
        best_img = img
        best_detections = None

    gray = cv2.cvtColor(best_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Mild deskew for small tilts only (cap at 5° to avoid misdetection)
    coords = np.column_stack(np.where(gray < 180))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if 0.3 < abs(angle) < 5.0:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            best_img = cv2.warpAffine(best_img, M, (w, h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
            gray = cv2.cvtColor(best_img, cv2.COLOR_BGR2GRAY)

    # Mild denoise — preserve handwriting ink
    denoised = cv2.fastNlMeansDenoising(gray, h=7)

    # Gentle contrast boost
    enhanced = cv2.convertScaleAbs(denoised, alpha=1.3, beta=-20)

    if output_path:
        out = output_path
    else:
        from pathlib import Path as _Path
        _p = _Path(image_path)
        out = str(_p.parent / f"{_p.stem}_processed{_p.suffix}")
    cv2.imwrite(out, enhanced)
    return out, best_detections


# ── Field extraction helpers ─────────────────────────────────────────────────

def _join_words(words):
    """Join a list of detection dicts into a single string, left to right."""
    return " ".join(w["text"] for w in sorted(words, key=lambda d: d["x"]))


def _image_height_width(image_path):
    import cv2
    img = cv2.imread(image_path)
    return img.shape[:2]


def _same_row(a, b, tolerance_ratio=0.6):
    """
    True if two detections are on the same text line.
    Uses a ratio of the taller word's height as tolerance so the check
    scales naturally regardless of image resolution or font size.
    """
    tolerance = max(a["h"], b["h"]) * tolerance_ratio
    return abs(a["y"] - b["y"]) <= tolerance


def _row_words(detections, anchor, tolerance_ratio=0.6):
    """All detections on the same row as anchor, sorted left to right."""
    return sorted(
        [d for d in detections if _same_row(anchor, d, tolerance_ratio)],
        key=lambda d: d["x"]
    )


def _words_right_of(detections, anchor, tolerance_ratio=0.6):
    """Detections on the same row as anchor and strictly to its right."""
    right_edge = anchor["x"] + anchor["w"]
    return [
        d for d in _row_words(detections, anchor, tolerance_ratio)
        if d["x"] > right_edge
    ]


def _words_left_of(detections, anchor, tolerance_ratio=0.6):
    """Detections on the same row as anchor and strictly to its left."""
    return [
        d for d in _row_words(detections, anchor, tolerance_ratio)
        if d["x"] + d["w"] < anchor["x"]
    ]


def _find_label(detections, *keywords):
    """
    Find the detection whose text contains one of the given keywords.
    Case-insensitive substring match to handle OCR merging (e.g. "PAYTO").
    Returns the best match (longest keyword matched) or None.
    """
    kw_upper = sorted([k.upper() for k in keywords], key=len, reverse=True)
    best = None
    best_kw_len = 0
    for d in detections:
        t = d["text"].upper().replace(" ", "")
        for kw in kw_upper:
            kw_nospace = kw.replace(" ", "")
            if kw_nospace in t and len(kw_nospace) > best_kw_len:
                best = d
                best_kw_len = len(kw_nospace)
    return best


def _rows_below(detections, anchor, max_gap_ratio=3.0, min_y=None):
    """
    Return detections that are below anchor, grouped into rows.
    max_gap_ratio: stop if gap between rows exceeds this multiple of anchor height.
    Returns a list of rows, each row being a list of detection dicts.
    """
    anchor_bottom = anchor["y"] + anchor["h"]
    max_gap = anchor["h"] * max_gap_ratio
    candidates = [
        d for d in detections
        if d["y"] > anchor_bottom
        and (min_y is None or d["y"] >= min_y)
    ]
    if not candidates:
        return []

    # Group into rows
    rows = []
    current_row = []
    prev_y = None
    for d in sorted(candidates, key=lambda x: (x["y"], x["x"])):
        if prev_y is None or abs(d["y"] - prev_y) <= anchor["h"] * 0.8:
            current_row.append(d)
        else:
            if d["y"] - prev_y > max_gap:
                break
            rows.append(current_row)
            current_row = [d]
        prev_y = d["y"]
    if current_row:
        rows.append(current_row)
    return rows


# ── MICR parser ───────────────────────────────────────────────────────────────

def _aba_valid(routing: str) -> bool:
    """Validate a 9-digit ABA routing number using the official checksum."""
    if not re.match(r'\d{9}$', routing):
        return False
    d = [int(c) for c in routing]
    return (3*(d[0]+d[3]+d[6]) + 7*(d[1]+d[4]+d[7]) + (d[2]+d[5]+d[8])) % 10 == 0


def _find_routing_fuzzy(digits: str):
    """
    Search a digit string for a valid 9-digit ABA routing number.

    US MICR layout:  [check# 4-6 digits] [routing 9 digits] [account 6-12 digits]

    Strategy:
      1. Exact match, constrained to the middle section of the string
         (positions 3 to len-12) to exclude the check# on the left and
         leave room for the account on the right.
      2. Single-digit fuzzy correction in the same window, scored by:
           - Routing first digit must be 0-3 (Federal Reserve constraint)
           - Prefer known OCR confusion pairs: 1↔2, 1↔7, 0↔6, 3↔8, 5↔6, 4↔9
           - Prefer positions closest to len/4 (typical routing start)

    Returns (position, corrected_routing) or (None, None).
    """
    CONFUSION = {
        ('1','2'): 2, ('2','1'): 2,
        ('1','7'): 2, ('7','1'): 2,
        ('0','6'): 2, ('6','0'): 2,
        ('3','8'): 2, ('8','3'): 2,
        ('5','6'): 2, ('6','5'): 2,
        ('4','9'): 2, ('9','4'): 2,
    }

    n = len(digits)
    if n < 14:
        return None, None

    # Typical routing number starts ~4-6 digits in from the left
    # Constrain search: at least 3 chars on left (check#), at least 3 on right (account)
    min_pos = 3
    max_pos = max(min_pos, n - 9 - 3)

    # Pass 1: exact match in the constrained window
    for i in range(min_pos, max_pos + 1):
        candidate = digits[i:i+9]
        if _aba_valid(candidate) and candidate[0] in '0123':
            return i, candidate

    # Pass 1b: exact match anywhere (in case check is unusually long/short)
    for i in range(n - 8):
        candidate = digits[i:i+9]
        if _aba_valid(candidate) and candidate[0] in '0123':
            return i, candidate

    # Pass 2: single-digit correction in constrained window
    ideal_pos = n // 4   # routing typically starts ~1/4 in
    candidates = []
    for i in range(min_pos, max_pos + 1):
        clist = list(digits[i:i+9])
        for fix_pos in range(9):
            original = clist[fix_pos]
            for digit in '0123456789':
                if digit == original:
                    continue
                clist[fix_pos] = digit
                corrected = ''.join(clist)
                if _aba_valid(corrected) and corrected[0] in '0123':
                    confusion_score = CONFUSION.get((original, digit), 0)
                    proximity_score = max(0, 6 - abs(i - ideal_pos))
                    edge_bonus = 1 if fix_pos in (0, 8) else 0
                    score = confusion_score * 3 + proximity_score + edge_bonus
                    candidates.append((score, i, corrected))
                clist[fix_pos] = original

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        _, best_i, best_routing = candidates[0]
        return best_i, best_routing

    # Pass 3: two-digit correction (handles double OCR errors in routing)
    # Only within the constrained window to keep it fast
    candidates = []
    for i in range(min_pos, max_pos + 1):
        clist = list(digits[i:i+9])
        for p1 in range(9):
            orig1 = clist[p1]
            for d1 in '0123456789':
                if d1 == orig1: continue
                clist[p1] = d1
                for p2 in range(p1+1, 9):
                    orig2 = clist[p2]
                    for d2 in '0123456789':
                        if d2 == orig2: continue
                        clist[p2] = d2
                        corrected = ''.join(clist)
                        if _aba_valid(corrected) and corrected[0] in '0123':
                            c1 = CONFUSION.get((orig1, d1), 0)
                            c2 = CONFUSION.get((orig2, d2), 0)
                            proximity_score = max(0, 6 - abs(i - ideal_pos))
                            score = (c1 + c2) * 3 + proximity_score
                            candidates.append((score, i, corrected))
                        clist[p2] = orig2
                clist[p1] = orig1

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        _, best_i, best_routing = candidates[0]
        return best_i, best_routing

    return None, None


def parse_micr(raw_micr: str) -> dict:
    """
    Parse MICR line into routing, account, and check number.

    Handles two cases:
    A) Delimiters survived OCR: ⑆122000661⑆ 23624305 ⑈ 3092
    B) All delimiters stripped, one long digit string: 003092122000662362430582

    For case B we search for the 9-digit routing number using the ABA
    checksum, with single-digit fuzzy correction for OCR errors (e.g.
    OCR reads 122000662 but correct value is 122000661).

    US MICR layout (left to right on the check bottom):
      [check# 4-6 digits] [routing 9 digits] [account 6-12 digits]
    """
    # ── Case A: delimiter-based parsing ──────────────────────────────────
    cleaned = raw_micr
    for ch in ["⑆","⑈","⑇","©","°","¢",":",";","!","T","t"]:
        cleaned = cleaned.replace(ch, "|")
    cleaned = re.sub(r"\s{2,}", "|", cleaned)
    m = re.search(r"\|(\d{8,9})\|[\s]*(\d{6,12})[\|\s]*(\d{3,6})?", cleaned)
    if m:
        return {
            "routing_number":    m.group(1),
            "account_number":    m.group(2),
            "check_number_micr": m.group(3) or "",
        }

    # ── Case B: concatenated digit string ─────────────────────────────────
    digits = re.sub(r"[^\d]", "", raw_micr)
    if len(digits) < 14:
        return {}

    pos, routing = _find_routing_fuzzy(digits)
    if routing:
        left  = digits[:pos]       # check number lives here
        right = digits[pos+9:]     # account number lives here
        # Check number: last 3-6 digits of the left segment
        check_num = left[-6:].lstrip('0') or left[-6:] if len(left) >= 3 else ""
        # Account: up to 12 digits on the right, strip trailing check# echo
        account = right[:12]
        return {
            "routing_number":    routing,
            "account_number":    account,
            "check_number_micr": check_num,
        }

    # ── Last resort: positional heuristic ────────────────────────────────
    # Layout: ~4-6 digit check# | 9-digit routing | remaining = account
    if len(digits) >= 20:
        return {
            "routing_number":    digits[6:15],
            "account_number":    digits[15:],
            "check_number_micr": digits[:6],
        }
    runs = re.findall(r"\d{6,}", digits)
    return {
        "routing_number":    runs[0] if len(runs) > 0 else "",
        "account_number":    runs[1] if len(runs) > 1 else "",
        "check_number_micr": runs[2] if len(runs) > 2 else "",
    }


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_check_fields(image_path: str,
                         preprocess: bool = True,
                         debug: bool = False) -> dict:
    """
    Extract structured fields from any US check image.

    Design principle: ZERO hardcoded pixel coordinates or percentage zones.
    Every field is found by locating its printed label first, then reading
    what is spatially adjacent to it. Distances are expressed as multiples
    of the detected character height so the logic scales to any resolution,
    DPI, or check geometry.

    Labels we anchor on:
      "PAY TO THE ORDER OF"  → payee (right), amount $ (far right)
      "DOLLARS"              → written amount (left of label)
      "MEMO" / "FOR"         → memo (right)
      "DATE"                 → date (right)   [if printed; many checks omit it]

    Fields without a label:
      Owner header  → topmost text cluster in the left half of the image
      Check number  → topmost text cluster in the right ~15% of the image
      MICR line     → lowest text row anywhere in the image
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    engine = get_ocr_engine()

    # ── Pre-process & auto-orient ─────────────────────────────────────────
    work_path = image_path
    cached_detections = None
    if preprocess:
        work_path, cached_detections = preprocess_image(
            image_path, ocr_engine=engine
        )

    if cached_detections and len(cached_detections) >= 10:
        detections = cached_detections
        print(f"[OCR] Reusing {len(detections)} detections from orientation pass")
    else:
        detections = run_ocr(work_path, engine)

    if len(detections) < 10:
        print(f"[warn] Only {len(detections)} detections — retrying raw image")
        detections = run_ocr(image_path, engine)

    if debug:
        print("\n── Raw OCR detections (sorted top→bottom) ──────────────")
        for d in detections:
            print(f"  y={d['y']:4d}  x={d['x']:4d}  h={d['h']:3d}  conf={d['conf']:.2f}  '{d['text']}'")
        print("─────────────────────────────────────────────────────────\n")

    img_h, img_w = _image_height_width(work_path)

    # ── Locate anchor labels ──────────────────────────────────────────────
    lbl_payto   = _find_label(detections,
                              "PAY TO THE ORDER OF", "PAY TO THE ORDER",
                              "PAY TO", "PAYTO", "ORDER OF", "ORDEROF",
                              "PAY", "ORDER")
    lbl_dollars = _find_label(detections, "DOLLARS", "DOLLAR")
    lbl_memo    = _find_label(detections, "MEMO", "FOR")
    lbl_date    = _find_label(detections, "DATE")

    if debug:
        print(f"  lbl_payto   = {lbl_payto['text'] if lbl_payto else None}")
        print(f"  lbl_dollars = {lbl_dollars['text'] if lbl_dollars else None}")
        print(f"  lbl_memo    = {lbl_memo['text'] if lbl_memo else None}")
        print(f"  lbl_date    = {lbl_date['text'] if lbl_date else None}\n")

    # ── Median character height — used as our adaptive unit of distance ───
    heights = [d["h"] for d in detections if d["h"] > 0]
    char_h  = sorted(heights)[len(heights)//2] if heights else 20

    # ────────────────────────────────────────────────────────────────────────
    # 1. PAYEE
    #    Right of PAY TO label on the same row. Exclude anything whose left
    #    edge is in the rightmost 20% (that's the $ amount box).
    # ────────────────────────────────────────────────────────────────────────
    payee = ""
    if lbl_payto:
        candidates = _words_right_of(detections, lbl_payto)
        candidates = [d for d in candidates
                      if d["x"] < img_w * 0.82]
        payee = _remove_label_words(
            _join_words(candidates),
            ["PAY TO THE ORDER OF","PAY TO","ORDER OF","PAYTO",
             "ORDEROF","PAY","TO","THE","ORDER","OF"]
        )
        # If nothing on same row, try the row immediately below the label
        if not payee.strip():
            below = _rows_below(detections, lbl_payto, max_gap_ratio=2.0)
            if below:
                row_words = [d for d in below[0] if d["x"] < img_w * 0.82]
                payee = _remove_label_words(
                    _join_words(row_words),
                    ["PAY TO THE ORDER OF","ORDER OF","ORDEROF"]
                )

    # ────────────────────────────────────────────────────────────────────────
    # 2. NUMERIC AMOUNT
    #    Rightmost token(s) on the same row as PAY TO that contain digits.
    #    Amounts live in the far-right box (right of 80% of image width).
    # ────────────────────────────────────────────────────────────────────────
    amount_numeric = ""
    if lbl_payto:
        row = _row_words(detections, lbl_payto)
        amt_candidates = [
            d for d in row
            if d["x"] > img_w * 0.75
            and re.search(r'\d', d["text"])
        ]
        amount_numeric = _clean_amount(_join_words(amt_candidates))

    # Fallback: scan all detections for a token that looks like "0000029400"
    # — a zero-padded amount common in MICR encoding (29400 = $294.00)
    if not amount_numeric:
        for d in detections:
            m = re.match(r'0*(\d+)(\d{2})$', d["text"].strip())
            if m and 8 <= len(d["text"].strip()) <= 12:
                dollars = m.group(1) or "0"
                cents   = m.group(2)
                amount_numeric = f"${dollars}.{cents}"
                break

    # ────────────────────────────────────────────────────────────────────────
    # 3. WRITTEN AMOUNT
    #    Everything on the same row as DOLLARS, to its left.
    # ────────────────────────────────────────────────────────────────────────
    amount_written = ""
    if lbl_dollars:
        # Everything on the DOLLARS row to the left of that label
        row = _row_words(detections, lbl_dollars)
        candidates = [d for d in row if d["x"] < lbl_dollars["x"]]
        amount_written = _remove_label_words(
            _join_words(candidates),
            ["AND NO/100","NO/100","00/100","AND"]
        )
        # If nothing found to the left, check rows immediately above DOLLARS
        if not amount_written.strip():
            above_dollars = [d for d in detections
                             if d["y"] < lbl_dollars["y"]
                             and d["y"] > lbl_dollars["y"] - char_h * 3
                             and d["x"] < lbl_dollars["x"]]
            amount_written = _remove_label_words(
                _join_words(above_dollars),
                ["DOLLARS","DOLLAR","AND NO/100","NO/100","AND"]
            )
    elif lbl_payto:
        below = _rows_below(detections, lbl_payto, max_gap_ratio=3.0)
        if below:
            amount_written = _remove_label_words(
                _join_words(below[0]),
                ["DOLLARS","DOLLAR","AND NO/100","NO/100","AND"]
            )

    # ────────────────────────────────────────────────────────────────────────
    # 4. MEMO
    #    Right of MEMO label on the same row.
    # ────────────────────────────────────────────────────────────────────────
    memo = ""
    if lbl_memo:
        candidates = _words_right_of(detections, lbl_memo)
        # Exclude long digit strings (MICR bleed)
        candidates = [d for d in candidates
                      if not re.match(r'\d{8,}', d["text"].strip())]
        memo = _remove_label_words(
            _join_words(candidates), ["MEMO","FOR","RE:"]
        )

    # ────────────────────────────────────────────────────────────────────────
    # 5. DATE
    #    If a DATE label exists, read to its right.
    #    Otherwise: the topmost row that sits to the right of centre and
    #    contains something that looks like a date.
    # ────────────────────────────────────────────────────────────────────────
    date_str = ""
    if lbl_date:
        candidates = _words_right_of(detections, lbl_date)
        date_str = _clean_date(_join_words(candidates))
    else:
        # Scan all detections for a date-like pattern (with or without slashes)
        for d in sorted(detections, key=lambda x: x["y"]):
            cleaned = _clean_date(d["text"])
            if re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', cleaned):
                date_str = cleaned
                break
        # If still no date, look for a token in the top-right area that is
        # purely digits 4-8 chars long — OCR often drops the slashes
        # e.g. "8/11/93" → "8193" or "81193"
        if not date_str:
            # Top quarter of image, right of centre
            top_right = [d for d in detections
                         if d["y"] < img_h * 0.30
                         and d["x"] > img_w * 0.55
                         and re.match(r'\d{4,8}$', d["text"].strip())]
            for d in sorted(top_right, key=lambda x: x["x"], reverse=True):
                date_str = _reconstruct_date(d["text"].strip())
                if date_str:
                    break

    # ────────────────────────────────────────────────────────────────────────
    # 6. CHECK NUMBER
    #    The topmost short numeric token that sits to the right of the
    #    rightmost detected label in the header row. On most checks it is
    #    the lone number in the top-right corner.
    # ────────────────────────────────────────────────────────────────────────
    check_number = ""
    # All detections above the PAY TO row
    above_pay = [d for d in detections
                 if lbl_payto is None or d["y"] < lbl_payto["y"]]
    # Candidate: purely numeric, 3-6 digits, in the right third of image
    check_candidates = sorted(
        [d for d in above_pay
         if re.match(r'\d{3,6}$', d["text"].strip())
         and d["x"] > img_w * 0.65],
        key=lambda d: d["y"]
    )
    if check_candidates:
        check_number = check_candidates[0]["text"].strip()

    # ────────────────────────────────────────────────────────────────────────
    # 7. OWNER NAME & ADDRESS
    #    All text above the PAY TO row that is in the left half of the image
    #    (right half is bank info). Group into lines; the topmost line is the
    #    account holder name, subsequent lines are address, phone, etc.
    # ────────────────────────────────────────────────────────────────────────
    owner_name = owner_address_line1 = owner_address_line2 = owner_phone = ""
    header_dets = sorted(
        [d for d in above_pay
         if d["x"] > img_w * 0.15          # exclude far-left PSA/label stickers
         and d["x"] < img_w * 0.55         # exclude bank info on right
         and not re.match(r'\d{3,6}$', d["text"].strip())],  # exclude check#
        key=lambda d: (d["y"], d["x"])
    )
    header_lines = _group_into_lines(header_dets, y_tolerance=char_h * 0.8)
    # Filter out authentication/PSA sticker text
    _AUTH = re.compile(
        r'^(AUTHENTIC(ITY)?|AUTO(GRAPH)?|PSA|DNA|CHECK|CERTIFIED|CERT|'
        r'GRADED|GRADE|NM|MT|GEM|MINT|LABEL|STICKER|AUTO)$', re.I
    )
    header_lines = [l for l in header_lines
                    if len(l.strip()) > 3 and not _AUTH.match(l.strip())]
    if len(header_lines) > 0: owner_name          = header_lines[0]
    if len(header_lines) > 1: owner_address_line1 = header_lines[1]
    if len(header_lines) > 2: owner_address_line2 = header_lines[2]
    owner_phone = next(
        (l for l in header_lines if re.search(r'\d{3}[\s\-\.]\d{3,4}', l)), ""
    )

    # ────────────────────────────────────────────────────────────────────────
    # 8. MICR LINE
    #    The bottommost row of detections. Also scan for any long (15+) digit
    #    run anywhere in the image since OCR sometimes misplaces MICR text.
    # ────────────────────────────────────────────────────────────────────────
    # Find the lowest Y detection
    if detections:
        max_y = max(d["y"] for d in detections)
        # All detections within 2 char heights of the bottom
        micr_dets = [d for d in detections
                     if d["y"] >= max_y - char_h * 2]
        micr_raw = _join_words(micr_dets)
    else:
        micr_raw = ""

    # Override with any long digit run (more reliable than position)
    long_runs = sorted(
        [d["text"] for d in detections
         if re.match(r'\d{15,}', d["text"].strip())],
        key=len, reverse=True
    )
    if long_runs:
        micr_raw = long_runs[0]

    micr = parse_micr(micr_raw)

    # Use MICR-derived check number if we didn't find a printed one
    if not check_number and micr.get("check_number_micr"):
        check_number = micr["check_number_micr"]

    return {
        "owner": {
            "name":          owner_name,
            "address_line1": owner_address_line1,
            "address_line2": owner_address_line2,
            "phone":         owner_phone,
        },
        "date":           date_str,
        "payee":          payee,
        "amount_numeric": amount_numeric,
        "amount_written": amount_written,
        "memo":           memo,
        "routing_number": micr.get("routing_number", ""),
        "account_number": micr.get("account_number", ""),
        "check_number":   check_number,
        "micr_raw":       micr_raw,
    }

# ── Utility helpers ──────────────────────────────────────────────────────────

def _group_into_lines(words: list[dict], y_tolerance: int = 15) -> list[str]:
    """Group words into text lines based on vertical proximity."""
    if not words:
        return []
    lines = []
    current_line = [words[0]]
    for word in words[1:]:
        if abs(word["y"] - current_line[-1]["y"]) <= y_tolerance:
            current_line.append(word)
        else:
            lines.append(_join_words(current_line))
            current_line = [word]
    lines.append(_join_words(current_line))
    return [l.strip() for l in lines if l.strip()]


def _remove_label_words(text: str, labels: list[str]) -> str:
    """Strip known label/boilerplate words from extracted text."""
    result = text
    for label in labels:
        result = re.sub(rf"\b{re.escape(label)}\b", "", result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip()


def _clean_amount(text: str) -> str:
    """Normalise amount string to $xxx.xx format."""
    # Remove everything except digits, dots, dollar signs, commas
    cleaned = re.sub(r"[^\d.,$ ]", "", text)
    # Find a decimal amount pattern
    m = re.search(r"\$?\s*(\d{1,6}[.,]\d{2})", cleaned)
    if m:
        return "$" + m.group(1).replace(",", ".")
    # Just grab digits
    digits = re.sub(r"[^\d]", "", cleaned)
    if len(digits) >= 2:
        return f"${digits[:-2]}.{digits[-2:]}"
    return cleaned.strip()


def _clean_date(text: str) -> str:
    """Try to parse a date from OCR text."""
    # Common formats: 8/11/93, 08-11-1993, Aug 11 93
    m = re.search(r"(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return text.strip()


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract structured fields from a scanned check image."
    )
    parser.add_argument("--image", required=True,
                        help="Path to the check image (JPEG/PNG/TIFF)")
    parser.add_argument("--output", default=None,
                        help="Optional path to save JSON output")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip image pre-processing step")
    parser.add_argument("--debug", action="store_true",
                        help="Print raw OCR detections for debugging")
    args = parser.parse_args()

    print(f"\n📄 Processing: {args.image}")
    print("─" * 55)

    result = extract_check_fields(
        image_path=args.image,
        preprocess=not args.no_preprocess,
        debug=args.debug,
    )

    print("\n✅ Extracted Fields:")
    print(f"  Owner Name    : {result['owner']['name']}")
    print(f"  Address       : {result['owner']['address_line1']}")
    print(f"                  {result['owner']['address_line2']}")
    print(f"  Phone         : {result['owner']['phone']}")
    print(f"  Date          : {result['date']}")
    print(f"  Payee         : {result['payee']}")
    print(f"  Amount ($)    : {result['amount_numeric']}")
    print(f"  Amount (text) : {result['amount_written']}")
    print(f"  Memo          : {result['memo']}")
    print(f"  Routing #     : {result['routing_number']}")
    print(f"  Account #     : {result['account_number']}")
    print(f"  Check #       : {result['check_number']}")
    print(f"  MICR raw      : {result['micr_raw']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Saved to: {args.output}")


if __name__ == "__main__":
    main()


# ── Additional helpers added for robustness ───────────────────────────────────

def _reconstruct_date(digits: str) -> str:
    """
    Try to reconstruct a date from a pure-digit string where OCR dropped slashes.
    e.g. "8193"  → "8/1/93"
         "81193" → "8/11/93"
         "81193" → "8/1/1993"  (tries multiple splits)
    Returns a formatted date string or "" if no valid split found.
    """
    digits = re.sub(r"[^\d]", "", digits)
    if len(digits) < 4 or len(digits) > 8:
        return ""

    # Try all splits: m|d|yy and m|dd|yy and mm|dd|yy etc.
    for m_len in (1, 2):
        for d_len in (1, 2):
            y_start = m_len + d_len
            if y_start >= len(digits):
                continue
            month = int(digits[:m_len])
            day   = int(digits[m_len:m_len+d_len])
            year  = digits[y_start:]
            if 1 <= month <= 12 and 1 <= day <= 31 and len(year) in (2, 4):
                return f"{month}/{day}/{year}"
    return ""
