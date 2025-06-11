import pymupdf4llm
import pathlib
import os

pdf_dir = os.path.join('../', 'pdfdocs')
md_dir = os.path.join('../', 'mddocs')

# Create markdown directory if it doesn't exist
pathlib.Path(md_dir).mkdir(parents=True, exist_ok=True)

# Get all PDF files in the directory
pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

for pdf_filename in pdf_files:
    try:
        # Full path to the PDF file
        pdf_path = os.path.join(pdf_dir, pdf_filename)
        
        # Convert PDF to markdown
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # Create output filename with .md extension
        md_filename = os.path.splitext(pdf_filename)[0] + '.md'
        md_path = os.path.join(md_dir, md_filename)
        
        # Write markdown file
        pathlib.Path(md_path).write_text(md_text, encoding='utf-8')
        
        print(f"Converted: {pdf_filename} -> {md_filename}")
        
    except Exception as e:
        print(f"Error converting {pdf_filename}: {str(e)}")

print("Conversion complete!")
