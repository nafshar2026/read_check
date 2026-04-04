"""
tests/test_check_reader.py – Unit tests for check_reader.py

These tests exercise the pure-Python parsing logic without requiring
Tesseract or real check images.
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import patch
from pathlib import Path

from check_reader import CheckData, CheckReader, _validate_routing_number, _clean_digits


# ---------------------------------------------------------------------------
# _validate_routing_number
# ---------------------------------------------------------------------------


class TestValidateRoutingNumber:
    def test_valid_routing_number(self):
        # 021000021 is Chase Bank's well-known ABA routing number
        assert _validate_routing_number("021000021") is True

    def test_valid_routing_number_2(self):
        # 111000025 is Federal Reserve Bank routing number
        assert _validate_routing_number("111000025") is True

    def test_invalid_routing_number_wrong_checksum(self):
        assert _validate_routing_number("021000022") is False

    def test_invalid_routing_number_too_short(self):
        assert _validate_routing_number("02100002") is False

    def test_invalid_routing_number_too_long(self):
        assert _validate_routing_number("0210000210") is False

    def test_invalid_routing_number_with_letters(self):
        assert _validate_routing_number("02100002A") is False

    def test_empty_string(self):
        assert _validate_routing_number("") is False


# ---------------------------------------------------------------------------
# _clean_digits
# ---------------------------------------------------------------------------


class TestCleanDigits:
    def test_removes_spaces(self):
        assert _clean_digits("1 2 3") == "123"

    def test_removes_special_chars(self):
        assert _clean_digits("T021000021T") == "021000021"

    def test_already_clean(self):
        assert _clean_digits("123456789") == "123456789"

    def test_empty_string(self):
        assert _clean_digits("") == ""


# ---------------------------------------------------------------------------
# CheckReader._parse_micr
# ---------------------------------------------------------------------------


class TestParseMicr:
    def test_flanked_routing_number(self):
        # Typical MICR after OCR noise: T<routing>T<account>A
        text = "T021000021T123456789A001"
        routing, account = CheckReader._parse_micr(text)
        assert routing == "021000021"
        assert account == "123456789"

    def test_routing_in_plain_text(self):
        # No MICR symbols but valid 9-digit routing present
        text = "021000021 987654321"
        routing, account = CheckReader._parse_micr(text)
        assert routing == "021000021"

    def test_no_valid_routing(self):
        # 123456789 fails the ABA checksum:
        # 3*(d[0]+d[3]+d[6]) + 7*(d[1]+d[4]+d[7]) + 1*(d[2]+d[5]+d[8])
        # = 3*(1+4+7) + 7*(2+5+8) + 1*(3+6+9) = 36 + 105 + 18 = 159, 159 % 10 != 0
        text = "123456789 987654321"
        routing, account = CheckReader._parse_micr(text)
        assert routing == ""

    def test_empty_text(self):
        routing, account = CheckReader._parse_micr("")
        assert routing == ""
        assert account == ""

    def test_routing_with_colon_separator(self):
        text = ":021000021:78901234567:"
        routing, account = CheckReader._parse_micr(text)
        assert routing == "021000021"
        assert account == "78901234567"


# ---------------------------------------------------------------------------
# CheckReader._parse_numeric_amount
# ---------------------------------------------------------------------------


class TestParseNumericAmount:
    def test_dollar_sign_and_cents(self):
        assert CheckReader._parse_numeric_amount("Pay $1,234.56 only") == "$1,234.56"

    def test_no_dollar_sign(self):
        result = CheckReader._parse_numeric_amount("Amount: 500.00")
        assert result == "$500.00"

    def test_no_amount(self):
        assert CheckReader._parse_numeric_amount("no numbers here") == ""

    def test_large_amount(self):
        result = CheckReader._parse_numeric_amount("$10,000.00")
        assert result == "$10,000.00"

    def test_whole_dollars(self):
        result = CheckReader._parse_numeric_amount("$750")
        assert result == "$750"


# ---------------------------------------------------------------------------
# CheckData.to_dict
# ---------------------------------------------------------------------------


class TestCheckData:
    def test_to_dict_keys(self):
        data = CheckData(
            name_address="John Doe\n123 Main St",
            amount="$500.00",
            routing_number="021000021",
            account_number="1234567890",
        )
        d = data.to_dict()
        assert set(d.keys()) == {"name_address", "amount", "routing_number", "account_number"}

    def test_to_dict_values(self):
        data = CheckData(
            name_address="Jane Smith\n456 Oak Ave",
            amount="$1,200.00",
            routing_number="111000025",
            account_number="9876543210",
        )
        d = data.to_dict()
        assert d["name_address"] == "Jane Smith\n456 Oak Ave"
        assert d["amount"] == "$1,200.00"
        assert d["routing_number"] == "111000025"
        assert d["account_number"] == "9876543210"

    def test_default_empty_fields(self):
        data = CheckData()
        d = data.to_dict()
        assert all(v == "" for v in d.values())


# ---------------------------------------------------------------------------
# CheckReader.__init__ – file-not-found guard
# ---------------------------------------------------------------------------


class TestCheckReaderInit:
    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CheckReader(tmp_path / "nonexistent.jpg")

    def test_accepts_existing_file(self, tmp_path):
        img_path = tmp_path / "blank.png"
        from PIL import Image as PILImage
        PILImage.fromarray(np.full((100, 300, 3), 255, dtype=np.uint8)).save(img_path)
        reader = CheckReader(img_path)
        assert reader.image_path == img_path


# ---------------------------------------------------------------------------
# CheckReader.read – integration smoke test with a synthetic check image
# ---------------------------------------------------------------------------


class TestCheckReaderRead:
    """
    Build a synthetic check image containing known text and MICR digits, then
    verify that the reader can extract them.  We mock pytesseract so the test
    doesn't require Tesseract to be installed in every CI environment.
    """

    @staticmethod
    def _make_white_image(tmp_path: Path) -> Path:
        from PIL import Image as PILImage
        img = PILImage.fromarray(np.full((400, 800, 3), 255, dtype=np.uint8))
        p = tmp_path / "check.png"
        img.save(p)
        return p

    def test_read_returns_check_data(self, tmp_path):
        img_path = self._make_white_image(tmp_path)

        # Provide fake OCR responses for each region OCR call
        responses = [
            "John Doe\n123 Main St\nAnytown CA 90210\n"
            "Pay to the order of: ABC Corp   $1,234.56\n"
            "One thousand two hundred thirty-four and 56/100 ---- DOLLARS\n"
            "T021000021T1234567890A0042",  # full-image pass
            "John Doe\n123 Main St\nAnytown CA 90210",  # name/address region
            "$1,234.56",                                # amount box region
            "One thousand two hundred thirty-four",     # amount line region
            "T021000021T1234567890A0042",               # MICR region
        ]
        call_count = 0

        def fake_image_to_string(img, config=""):
            nonlocal call_count
            result = responses[call_count % len(responses)]
            call_count += 1
            return result

        with patch("pytesseract.image_to_string", side_effect=fake_image_to_string):
            reader = CheckReader(img_path)
            data = reader.read()

        assert isinstance(data, CheckData)
        assert "John Doe" in data.name_address
        assert data.amount == "$1,234.56"
        assert data.routing_number == "021000021"
        assert data.account_number == "1234567890"

    def test_read_gracefully_handles_ocr_failure(self, tmp_path):
        img_path = self._make_white_image(tmp_path)

        with patch("pytesseract.image_to_string", side_effect=RuntimeError("OCR failed")):
            reader = CheckReader(img_path)
            data = reader.read()

        # Should still return a CheckData with errors recorded
        assert isinstance(data, CheckData)
        assert any("OCR" in e or "failed" in e.lower() for e in data.errors)
