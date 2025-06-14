#!/usr/bin/env python3
"""
PDF to Markdown Converter with Image Extraction
Converts PDFs to markdown and extracts images with vertical stitching,
then references the images in the markdown file.
"""

import pymupdf
import pymupdf4llm
import os
import sys
import pathlib
import re
from PIL import Image
import io


def postprocess_markdown(markdown_text):
    """
    Remove unwanted lines from markdown text.
    
    Removes:
    - Lines containing "ESG Guidelines"
    - Lines containing square brackets with dots [..........] (any number of dots)
    
    Args:
        markdown_text (str): Original markdown text
    
    Returns:
        str: Cleaned markdown text
    """
    
    lines = markdown_text.split('\n')
    cleaned_lines = []
    removed_count = 0
    
    for line in lines:
        should_remove = False 
        
        # Check for square brackets with dots pattern [..........] 
        if re.search(r'\[\.+\]', line):
            should_remove = True
            print(f"    Removed line with dot pattern: {line.strip()[:50]}...")
        
        if not should_remove:
            cleaned_lines.append(line)
        else:
            removed_count += 1
    
    if removed_count > 0:
        print(f"  Postprocessing: Removed {removed_count} unwanted line(s)")
    
    return '\n'.join(cleaned_lines)


def extract_and_stitch_page_regions(pdf_path, output_dir, dpi=300, min_region_size=200):
    """
    Extract image regions from each page and stitch them together vertically.
    
    Returns:
        dict: Dictionary mapping page numbers to created image filenames
    """
    
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    page_images = {}  # page_num -> filename
    
    try:
        doc = pymupdf.open(pdf_path)
        print(f"  Extracting images from: {os.path.basename(pdf_path)}")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get image blocks/areas on the page
            image_list = page.get_images()
            
            if not image_list:
                continue
            
            print(f"    Page {page_num + 1}: Found {len(image_list)} image(s)")
            
            # Get page dimensions
            page_rect = page.rect
            
            # Store regions as PIL Images for stitching
            region_images = []
            
            # Extract each image region
            for img_index, img in enumerate(image_list):
                try:
                    # Create region based on image position
                    if len(image_list) == 1:
                        # Single image - use most of the page
                        clip_rect = page_rect
                    else:
                        # Multiple images - create vertical sections
                        section_height = page_rect.height / len(image_list)
                        y0 = img_index * section_height
                        y1 = (img_index + 1) * section_height
                        clip_rect = pymupdf.Rect(0, y0, page_rect.width, y1)
                    
                    # Render this region
                    zoom = dpi / 72.0
                    mat = pymupdf.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                    
                    # Skip very small regions
                    if pix.width < min_region_size or pix.height < min_region_size:
                        pix = None
                        continue
                    
                    # Convert PyMuPDF pixmap to PIL Image
                    img_data = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_data))
                    region_images.append(pil_image)
                    
                    # Clean up PyMuPDF pixmap
                    pix = None
                    
                except Exception as e:
                    print(f"      Error extracting region {img_index + 1}: {e}")
                    continue
            
            # If we have regions to stitch, combine them vertically
            if region_images:
                try:
                    # Calculate dimensions for the stitched image
                    max_width = max(img.width for img in region_images)
                    total_height = sum(img.height for img in region_images)
                    
                    # Create a new image with the combined dimensions
                    stitched_image = Image.new('RGB', (max_width, total_height), 'white')
                    
                    # Paste each region vertically
                    current_y = 0
                    for region_img in region_images:
                        # Center the image horizontally if it's narrower than max_width
                        x_offset = (max_width - region_img.width) // 2
                        stitched_image.paste(region_img, (x_offset, current_y))
                        current_y += region_img.height
                    
                    # Save the stitched image
                    filename = f"page_{page_num + 1:03d}_images.png"
                    filepath = os.path.join(output_dir, filename)
                    stitched_image.save(filepath, 'PNG', quality=95)
                    
                    # Store the mapping
                    page_images[page_num + 1] = filename
                    
                    print(f"      ✓ Saved stitched image: {filename} ({max_width}x{total_height})")
                    
                    # Clean up
                    stitched_image.close()
                    for img in region_images:
                        img.close()
                    
                except Exception as e:
                    print(f"      Error stitching regions for page {page_num + 1}: {e}")
                    # Clean up on error
                    for img in region_images:
                        img.close()
        
        doc.close()
        return page_images
        
    except Exception as e:
        print(f"    Error processing PDF for images: {e}")
        return {}


def insert_image_references(markdown_text, page_images, images_folder_name):
    """
    Insert image references into markdown text at appropriate page locations.
    
    Args:
        markdown_text (str): Original markdown text
        page_images (dict): Dictionary mapping page numbers to image filenames
        images_folder_name (str): Name of the images folder for relative paths
    
    Returns:
        str: Modified markdown text with image references
    """
    
    if not page_images:
        return markdown_text
    
    # Split markdown into lines
    lines = markdown_text.split('\n')
    modified_lines = []
    
    # Track current page (look for page breaks or use heuristics)
    current_page = 1
    
    # Process each line
    for i, line in enumerate(lines):
        modified_lines.append(line)
        
        # Detect page breaks (common patterns in pymupdf4llm output)
        # This is heuristic-based and might need adjustment based on your PDFs
        if (line.strip() == '---' or  # Common page separator
            line.strip().startswith('---') or
            (line.strip() == '' and i > 0 and lines[i-1].strip() == '') or  # Double empty lines
            re.match(r'^#+\s+Page\s+\d+', line, re.IGNORECASE) or  # Explicit page headers
            re.match(r'^\s*\d+\s*$', line.strip())):  # Standalone page numbers
            
            # Check if we have an image for the current page
            if current_page in page_images:
                image_path = f"{images_folder_name}/{page_images[current_page]}"
                image_reference = f"\n![Page {current_page} Images]({image_path})\n"
                modified_lines.append(image_reference)
                print(f"    Added image reference for page {current_page}")
            
            current_page += 1
    
    # Add any remaining images that weren't inserted
    for page_num in page_images:
        if page_num >= current_page:
            image_path = f"{images_folder_name}/{page_images[page_num]}"
            image_reference = f"\n![Page {page_num} Images]({image_path})\n"
            modified_lines.append(image_reference)
            print(f"    Added image reference for page {page_num} at end")
    
    return '\n'.join(modified_lines)


def add_images_at_content_breaks(markdown_text, page_images, images_folder_name):
    """
    Alternative method: Add images at natural content breaks (headers, long gaps).
    """
    
    if not page_images:
        return markdown_text
    
    lines = markdown_text.split('\n')
    modified_lines = []
    
    # Sort page numbers
    sorted_pages = sorted(page_images.keys())
    current_image_index = 0
    
    # Look for good places to insert images (after headers, before major sections)
    for i, line in enumerate(lines):
        modified_lines.append(line)
        
        # Insert image after headers or at content breaks
        if (current_image_index < len(sorted_pages) and
            (line.startswith('#') or  # After headers
             (line.strip() == '' and i < len(lines) - 1 and lines[i+1].startswith('#')) or  # Before headers
             (i > 0 and lines[i-1].strip() == '' and line.strip() == ''))):  # At paragraph breaks
            
            page_num = sorted_pages[current_image_index]
            image_path = f"{images_folder_name}/{page_images[page_num]}"
            image_reference = f"\n![Page {page_num} Images]({image_path})\n"
            modified_lines.append(image_reference)
            print(f"    Added image reference for page {page_num}")
            current_image_index += 1
    
    # Add any remaining images at the end
    while current_image_index < len(sorted_pages):
        page_num = sorted_pages[current_image_index]
        image_path = f"{images_folder_name}/{page_images[page_num]}"
        image_reference = f"\n![Page {page_num} Images]({image_path})\n"
        modified_lines.append(image_reference)
        print(f"    Added image reference for page {page_num} at end")
        current_image_index += 1
    
    return '\n'.join(modified_lines)


def process_single_pdf(pdf_path, md_dir, method="content_breaks"):
    """
    Process a single PDF: convert to markdown and extract images.
    
    Args:
        pdf_path (str): Path to the PDF file
        md_dir (str): Directory to save markdown and images
        method (str): Method for inserting images ("page_breaks" or "content_breaks")
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        pdf_filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(pdf_filename)[0]
        
        print(f"\nProcessing: {pdf_filename}")
        
        # Convert PDF to markdown
        print("  Converting to markdown...")
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # Apply postprocessing to clean up the markdown
        print("  Applying postprocessing filters...")
        md_text = postprocess_markdown(md_text)
        
        # Create images directory
        images_folder_name = f"{base_name}_images"
        images_dir = os.path.join(md_dir, images_folder_name)
        
        # Extract and stitch images
        print("  Extracting and stitching images...")
        page_images = extract_and_stitch_page_regions(pdf_path, images_dir, dpi=300)
        
        # Insert image references into markdown
        if page_images:
            print(f"  Inserting {len(page_images)} image references into markdown...")
            if method == "page_breaks":
                modified_md_text = insert_image_references(md_text, page_images, images_folder_name)
            else:  # content_breaks
                modified_md_text = add_images_at_content_breaks(md_text, page_images, images_folder_name)
        else:
            print("  No images found, using original markdown")
            modified_md_text = md_text
        
        # Save markdown file
        md_filename = f"{base_name}.md"
        md_path = os.path.join(md_dir, md_filename)
        pathlib.Path(md_path).write_text(modified_md_text, encoding='utf-8')
        
        print(f"  ✓ Saved: {md_filename}")
        if page_images:
            print(f"  ✓ Images saved in: {images_folder_name}/")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {pdf_filename}: {str(e)}")
        return False


def main():
    """Main function to process all PDFs in a directory."""
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage: python converter.py [pdf_directory] [markdown_directory] [method]")
            print("\nArguments:")
            print("  pdf_directory      - Directory containing PDF files (default: '../pdfdocs')")
            print("  markdown_directory - Directory to save markdown and images (default: '../mddocs')")
            print("  method            - Image insertion method: 'page_breaks' or 'content_breaks' (default: 'content_breaks')")
            print("\nMethods:")
            print("  page_breaks    - Insert images at detected page breaks")
            print("  content_breaks - Insert images at natural content breaks (headers, paragraphs)")
            print("\nPostprocessing:")
            print("  - Automatically removes lines containing 'ESG Guidelines'")
            print("  - Automatically removes lines with dot patterns like [..........] ")
            print("\nExamples:")
            print("  python converter.py")
            print("  python converter.py ./pdfs ./output")
            print("  python converter.py ./pdfs ./output page_breaks")
            sys.exit(0)
    
    # Configuration
    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join('../', 'pdfdocs')
    md_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join('../', 'mddocs')
    method = sys.argv[3] if len(sys.argv) > 3 else "content_breaks"
    
    # Validate method
    if method not in ["page_breaks", "content_breaks"]:
        print(f"Error: Unknown method '{method}'. Use 'page_breaks' or 'content_breaks'.")
        sys.exit(1)
    
    # Check if PDF directory exists
    if not os.path.exists(pdf_dir):
        print(f"Error: PDF directory '{pdf_dir}' not found.")
        sys.exit(1)
    
    # Create markdown directory if it doesn't exist
    pathlib.Path(md_dir).mkdir(parents=True, exist_ok=True)
    
    # Check dependencies
    try:
        import pymupdf4llm
        from PIL import Image
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}")
        print("Install with: pip install pymupdf4llm Pillow")
        sys.exit(1)
    
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{pdf_dir}'")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF file(s) in '{pdf_dir}'")
    print(f"Output directory: '{md_dir}'")
    print(f"Image insertion method: {method}")
    print("=" * 60)
    
    # Process each PDF
    successful = 0
    failed = 0
    
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_filename)
        
        if process_single_pdf(pdf_path, md_dir, method):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {successful} PDF(s)")
    print(f"Failed: {failed} PDF(s)")
    print(f"Output saved to: {os.path.abspath(md_dir)}")
    
    if successful > 0:
        print("\nFiles created:")
        for pdf_filename in pdf_files:
            base_name = os.path.splitext(pdf_filename)[0]
            print(f"  {base_name}.md")
            print(f"  {base_name}_images/ (if images found)")


if __name__ == "__main__":
    main()
