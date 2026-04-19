"""
PDF Report Exporter
===================
Converts the LLM-generated Markdown advisory report to a downloadable PDF.
"""

import logging
import os
import re
import tempfile

from fpdf import FPDF

logger = logging.getLogger(__name__)

# Characters outside ASCII that must be replaced before writing to
# the built-in PDF fonts (Helvetica uses Latin-1 internally).
_UNICODE_REPLACEMENTS = {
    '\u20b9': 'Rs.',   # Indian Rupee sign
    '\u2013': '-',     # en dash
    '\u2014': '--',    # em dash
    '\u2018': "'",     # left single quote
    '\u2019': "'",     # right single quote
    '\u201c': '"',     # left double quote
    '\u201d': '"',     # right double quote
    '\u2022': '-',     # bullet
    '\u00b2': '^2',    # superscript 2
    '\u00b0': 'deg',   # degree sign
    '\u00a0': ' ',     # non-breaking space
}


def _to_latin1_safe(text: str) -> str:
    """Replace known Unicode characters, then drop anything outside Latin-1."""
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text.encode('latin-1', errors='replace').decode('latin-1')


def _clean_markdown(text: str) -> str:
    """Strip Markdown syntax for plain-text PDF rendering."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)        # bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)             # italic
    text = re.sub(r'^#{1,6}\s+(.*?)$', r'\1', text, flags=re.MULTILINE)  # headings
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)    # inline code
    return text


class ReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.set_text_color(16, 185, 129)
        self.cell(0, 10, 'Agentic AI Real Estate Advisory Report', border=0, new_x='LMARGIN', new_y='NEXT', align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C')


def generate_report_pdf(markdown_content: str) -> str:
    """Convert a Markdown advisory report to PDF and return the temp file path.

    The PDF uses built-in Helvetica (Latin-1). Unicode characters such as
    the Indian Rupee sign (U+20B9) are mapped to ASCII equivalents before
    rendering so no characters are silently dropped.

    Args:
        markdown_content: Markdown string from the LLM advisory agent.

    Returns:
        Absolute path to the generated temporary PDF file.

    Raises:
        RuntimeError: if PDF generation fails.
    """
    try:
        pdf = ReportPDF()
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)
        pdf.add_page()

        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Advisory Insights', border=0, new_x='LMARGIN', new_y='NEXT')
        pdf.ln(5)

        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(0)

        effective_width = pdf.w - pdf.l_margin - pdf.r_margin

        for line in markdown_content.split('\n'):
            clean_line = _to_latin1_safe(_clean_markdown(line)).strip()

            if not clean_line:
                pdf.ln(3)
                continue

            # Indent bullet/list items
            if line.strip().startswith(('-', '*', '+')):
                clean_line = '  ' + clean_line

            pdf.multi_cell(w=effective_width, h=6, text=clean_line, new_x='LMARGIN', new_y='NEXT')

        filepath = os.path.join(tempfile.gettempdir(), 'real_estate_advisory.pdf')
        pdf.output(filepath)
        logger.info("PDF report written to %s", filepath)
        return filepath

    except Exception as exc:
        logger.error("PDF generation failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Could not generate PDF report: {exc}") from exc
