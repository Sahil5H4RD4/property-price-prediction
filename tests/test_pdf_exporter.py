"""
Unit tests for src/pdf_exporter.py

Tests cover:
- _clean_markdown: removes bold, italic, heading markers, code spans
- _to_latin1_safe: replaces rupee sign, dashes, quotes; encodes safely
- generate_report_pdf: creates a non-empty PDF file at a valid path
"""

import os
import pytest

from src.pdf_exporter import _clean_markdown, _to_latin1_safe, generate_report_pdf


# ─── _clean_markdown ─────────────────────────────────────────────

class TestCleanMarkdown:
    def test_removes_bold_markers(self):
        assert _clean_markdown("**bold text**") == "bold text"

    def test_removes_italic_markers(self):
        assert _clean_markdown("*italic text*") == "italic text"

    def test_removes_heading_prefix(self):
        assert _clean_markdown("## Section Title") == "Section Title"

    def test_removes_deep_heading_prefix(self):
        assert _clean_markdown("### Sub Section") == "Sub Section"

    def test_removes_inline_code_backtick(self):
        assert _clean_markdown("`some_var`") == "some_var"

    def test_plain_text_unchanged(self):
        text = "Plain text with no markdown."
        assert _clean_markdown(text) == text

    def test_nested_bold_and_heading(self):
        result = _clean_markdown("## **Bold Heading**")
        assert result == "Bold Heading"


# ─── _to_latin1_safe ─────────────────────────────────────────────

class TestToLatin1Safe:
    def test_rupee_sign_replaced(self):
        result = _to_latin1_safe("\u20b9 5,00,000")
        assert "\u20b9" not in result
        assert "Rs." in result

    def test_en_dash_replaced(self):
        result = _to_latin1_safe("price \u2013 range")
        assert "\u2013" not in result

    def test_em_dash_replaced(self):
        result = _to_latin1_safe("note\u2014this")
        assert "\u2014" not in result

    def test_curly_quotes_replaced(self):
        result = _to_latin1_safe("\u2018hello\u2019 and \u201cworld\u201d")
        assert "\u2018" not in result
        assert "\u201c" not in result

    def test_non_breaking_space_replaced(self):
        result = _to_latin1_safe("a\u00a0b")
        assert "\u00a0" not in result

    def test_output_encodeable_to_latin1(self):
        text = "\u20b9 property at \u2013 10 lakhs"
        result = _to_latin1_safe(text)
        # Should not raise
        result.encode('latin-1')

    def test_plain_ascii_unchanged(self):
        text = "Property at Rs. 5000000"
        assert _to_latin1_safe(text) == text


# ─── generate_report_pdf ─────────────────────────────────────────

class TestGenerateReportPdf:
    SAMPLE_REPORT = """
# Advisory Report

## Property Summary

**Area**: 7420 sqft, 4 BHK

Predicted Price: \u20b9 50,00,000

- Good location
- Market trend: rising

## Disclaimer

Not financial advice.
"""

    def test_returns_string_path(self):
        path = generate_report_pdf(self.SAMPLE_REPORT)
        assert isinstance(path, str)

    def test_file_exists_after_generation(self):
        path = generate_report_pdf(self.SAMPLE_REPORT)
        assert os.path.exists(path), f"PDF not found at {path}"

    def test_file_is_non_empty(self):
        path = generate_report_pdf(self.SAMPLE_REPORT)
        assert os.path.getsize(path) > 0

    def test_file_is_valid_pdf_header(self):
        path = generate_report_pdf(self.SAMPLE_REPORT)
        with open(path, 'rb') as f:
            header = f.read(4)
        assert header == b'%PDF', "File should start with PDF magic bytes"

    def test_rupee_in_content_does_not_crash(self):
        report = "Price is \u20b9 75,00,000 — good deal."
        path = generate_report_pdf(report)
        assert os.path.exists(path)

    def test_empty_report_generates_pdf(self):
        path = generate_report_pdf("")
        assert os.path.exists(path)
