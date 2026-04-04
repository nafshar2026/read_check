"""
check_reader.py – Extract structured information from an image of a handwritten check.

Extracts:
  - Account-holder name and address (top-left of check)
  - Dollar amount (numeric box and written line)
  - ABA routing number (MICR line, bottom of check)
  - Bank account number (MICR line, bottom of check)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CheckData:
    """Structured data extracted from a check image."""

    name_address: str = ""
    amount: str = ""
    routing_number: str = ""
    account_number: str = ""
    raw_text: str = ""
    micr_text: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name_address": self.name_address,
            "amount": self.amount,
            "routing_number": self.routing_number,
            "account_number": self.account_number,
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# ABA routing-number checksum validation
def _validate_routing_number(routing: str) -> bool:
    """Return True if *routing* passes the ABA checksum (mod-10 rule)."""
    if not re.fullmatch(r"\d{9}", routing):
        return False
    d = [int(c) for c in routing]
    total = (
        3 * (d[0] + d[3] + d[6])
        + 7 * (d[1] + d[4] + d[7])
        + 1 * (d[2] + d[5] + d[8])
    )
    return total % 10 == 0


def _clean_digits(text: str) -> str:
    """Strip all non-digit characters from *text*."""
    return re.sub(r"\D", "", text)


def _preprocess_for_ocr(image: np.ndarray, *, scale: float = 2.0) -> np.ndarray:
    """
    Convert to greyscale, upscale, and apply adaptive thresholding so that
    Tesseract has an easier time reading the text.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Upscale for better OCR accuracy
    h, w = gray.shape[:2]
    gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Denoise then threshold
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return binary


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class CheckReader:
    """
    Read a check image and extract structured financial information.

    Parameters
    ----------
    image_path:
        Path to the check image (JPEG, PNG, TIFF, etc.)
    tesseract_cmd:
        Optional path to the ``tesseract`` binary if it is not on ``PATH``.
    """

    # Approximate vertical fractions of a standard US personal check
    _REGION_NAME_ADDR = (0.00, 0.00, 0.45, 0.40)  # (x1, y1, x2, y2) normalised
    _REGION_AMOUNT_BOX = (0.60, 0.20, 1.00, 0.45)  # numeric dollar box (top-right)
    _REGION_AMOUNT_LINE = (0.10, 0.40, 0.85, 0.60)  # written-amount line
    _REGION_MICR = (0.00, 0.82, 1.00, 1.00)          # MICR strip at the bottom

    def __init__(self, image_path: str | Path, tesseract_cmd: Optional[str] = None):
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self._bgr: Optional[np.ndarray] = None  # loaded later

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> CheckData:
        """
        Process the check image and return a :class:`CheckData` instance.
        """
        result = CheckData()
        try:
            self._bgr = self._load_image()
        except Exception as exc:
            result.errors.append(f"Failed to load image: {exc}")
            return result

        # --- Full-image OCR (provides fallback for all fields) ---
        try:
            result.raw_text = self._ocr_region(self._bgr, region=None)
        except Exception as exc:
            result.errors.append(f"Full-image OCR failed: {exc}")

        # --- Name / address ---
        try:
            result.name_address = self._extract_name_address()
        except Exception as exc:
            result.errors.append(f"Name/address extraction failed: {exc}")

        # --- Amount ---
        try:
            result.amount = self._extract_amount()
        except Exception as exc:
            result.errors.append(f"Amount extraction failed: {exc}")

        # --- MICR (routing + account) ---
        try:
            micr_text = self._extract_micr_text()
            result.micr_text = micr_text
            result.routing_number, result.account_number = self._parse_micr(micr_text)
        except Exception as exc:
            result.errors.append(f"MICR extraction failed: {exc}")

        # --- Fallback: try full-page text for any missing field ---
        if not result.routing_number or not result.account_number:
            rt, ac = self._parse_micr(result.raw_text)
            if rt and not result.routing_number:
                result.routing_number = rt
                result.errors.append("Routing number found via full-page fallback OCR.")
            if ac and not result.account_number:
                result.account_number = ac
                result.errors.append("Account number found via full-page fallback OCR.")

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_image(self) -> np.ndarray:
        """Load the image file via Pillow (handles wide format support)."""
        pil_img = Image.open(self.image_path).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _crop_region(self, img: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
        """
        Crop *img* to *region* where region is (x1, y1, x2, y2) in normalised
        [0, 1] coordinates.
        """
        h, w = img.shape[:2]
        x1 = int(region[0] * w)
        y1 = int(region[1] * h)
        x2 = int(region[2] * w)
        y2 = int(region[3] * h)
        return img[y1:y2, x1:x2]

    def _ocr_region(
        self,
        img: np.ndarray,
        *,
        region: Optional[tuple[float, float, float, float]] = None,
        config: str = "--psm 6",
        scale: float = 2.0,
    ) -> str:
        """
        Run Tesseract OCR on *img* (or a cropped region of it).

        Parameters
        ----------
        img:    Source image (BGR numpy array)
        region: Optional normalised crop coordinates.
        config: Tesseract page-segmentation-mode flags.
        scale:  Upscale factor before OCR.
        """
        if region is not None:
            img = self._crop_region(img, region)
        processed = _preprocess_for_ocr(img, scale=scale)
        pil_img = Image.fromarray(processed)
        text = pytesseract.image_to_string(pil_img, config=config)
        return text.strip()

    # --- Name / address ------------------------------------------------

    def _extract_name_address(self) -> str:
        """OCR the top-left area of the check where name/address usually live."""
        text = self._ocr_region(self._bgr, region=self._REGION_NAME_ADDR, config="--psm 6")
        # Remove blank lines and strip excess whitespace
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)

    # --- Amount --------------------------------------------------------

    def _extract_amount(self) -> str:
        """
        Try the numeric amount box first, then fall back to the written-amount
        line, then to a regex scan of the full raw text.
        """
        # 1. Numeric box (top-right of check)
        box_text = self._ocr_region(
            self._bgr,
            region=self._REGION_AMOUNT_BOX,
            config="--psm 7 -c tessedit_char_whitelist=0123456789$.,",
        )
        amount = self._parse_numeric_amount(box_text)
        if amount:
            return amount

        # 2. Written-amount line
        line_text = self._ocr_region(self._bgr, region=self._REGION_AMOUNT_LINE, config="--psm 7")
        amount = self._parse_numeric_amount(line_text)
        if amount:
            return amount

        # 3. Regex over full-page text
        if self._bgr is not None:
            return self._parse_numeric_amount(self._ocr_region(self._bgr, region=None))
        return ""

    @staticmethod
    def _parse_numeric_amount(text: str) -> str:
        """
        Return the first dollar-amount pattern found in *text*, e.g. ``"$1,234.56"``.
        """
        # Match patterns like $1,234.56 or 1234.56 or 1,234
        # When decimals are present, require exactly two places (standard currency)
        pattern = r"\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?"
        match = re.search(pattern, text)
        if match:
            value = re.sub(r"\s+", "", match.group())
            if not value.startswith("$"):
                value = "$" + value
            return value
        return ""

    # --- MICR ----------------------------------------------------------

    def _extract_micr_text(self) -> str:
        """
        Extract the MICR strip (bottom of the check) with digit-optimised OCR.
        """
        return self._ocr_region(
            self._bgr,
            region=self._REGION_MICR,
            config="--psm 7 -c tessedit_char_whitelist=0123456789TCADco|:",
            scale=3.0,
        )

    @staticmethod
    def _parse_micr(text: str) -> tuple[str, str]:
        """
        Parse *text* for an ABA routing number and a bank account number.

        MICR E-13B layout:  ⑆ routing ⑆  account ⑇  check#
        After OCR the transit symbols often become ``T``, ``|``, ``:``, ``o``
        or similar noise characters; we rely on positional digit runs.

        Returns
        -------
        (routing_number, account_number) – each as a plain digit string, or
        empty string if not found.
        """
        routing = ""
        account = ""

        # --- Strategy 1: find digit groups separated by MICR-like symbols ---
        # Remove spaces for cleaner matching
        compact = re.sub(r"\s+", "", text)

        # Known MICR substitution characters: T, |, :, o, O, D, C, A
        micr_sep = r"[T|:oODCAc\[\]{}]+"

        # Routing: 9-digit group flanked by MICR symbols
        routing_pattern = rf"{micr_sep}(\d{{9}}){micr_sep}"
        m = re.search(routing_pattern, compact)
        if m:
            candidate = m.group(1)
            if _validate_routing_number(candidate):
                routing = candidate

        # If no flanked group, look for any 9-digit run that passes ABA checksum
        if not routing:
            for m in re.finditer(r"\d{9}", compact):
                candidate = m.group()
                if _validate_routing_number(candidate):
                    routing = candidate
                    break

        # Account: digit run after routing, before the next MICR separator
        if routing:
            after_routing = compact[compact.find(routing) + 9:]
            # Strip leading MICR noise
            after_routing = re.sub(r"^" + micr_sep, "", after_routing)
            m = re.match(r"(\d{5,17})", after_routing)
            if m:
                account = m.group(1)

        # --- Strategy 2: all digit groups of meaningful length ---
        if not routing or not account:
            # Grab all runs of 5+ digits
            groups = re.findall(r"\d{5,}", text)
            for g in groups:
                if len(g) >= 9 and not routing:
                    candidate = g[:9]
                    if _validate_routing_number(candidate):
                        routing = candidate
                        remainder = g[9:]
                        if len(remainder) >= 5:
                            account = remainder
                        continue
                if routing and not account and 5 <= len(g) <= 17:
                    account = g

        return routing, account
