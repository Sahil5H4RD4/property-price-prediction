import tempfile
import os
from fpdf import FPDF
import re

class ReportPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.set_text_color(16, 185, 129) # Emerald color
        self.cell(0, 10, 'Agentic AI Real Estate Advisory Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def clean_markdown(text):
    """Simple clean up of markdown symbols for basic PDF text rendering."""
    # Remove bold/italic stars
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove headers prefixes
    text = re.sub(r'^#+\s+(.*?)$', r'\1', text, flags=re.MULTILINE)
    return text

def generate_report_pdf(markdown_content: str) -> str:
    """
    Generates a PDF from the markdown report and returns the file path.
    """
    pdf = ReportPDF()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.add_page()
    
    # Title
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, 'Advisory Insights', ln=1)
    pdf.ln(5)
    
    # Body
    pdf.set_font('helvetica', '', 11)
    pdf.set_text_color(0)
    
    # Splitting by lines
    lines = markdown_content.split('\n')
    effective_width = pdf.w - pdf.l_margin - pdf.r_margin

    for line in lines:
        clean_line = clean_markdown(line).strip()
        
        # Handle empty lines
        if not clean_line:
            pdf.ln(3)
            continue
            
        # Very rudimentary handling of bullet points
        if line.strip().startswith('-') or line.strip().startswith('*'):
            clean_line = "  " + clean_line
            
        # Multi-cell auto line break
        try:
            # We use effective_width instead of 0 to be explicit
            # Also handle potential encoding issues more safely
            safe_text = clean_line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(w=effective_width, h=6, text=safe_text, ln=1)
        except Exception as e:
            # Fallback for line that fails to render
            pdf.ln(6)
    
    # Save to a temporary file
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, "real_estate_advisory.pdf")
    pdf.output(filepath)
    
    return filepath
