def extract_and_stitch_page_regions(pdf_path, output_dir, current_logger, dpi=300, min_region_size=200):
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
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_image_list = page.get_images()
            if not page_image_list:
                continue
            page_dim = page.rect
            # Store regions as PIL Images for stitching
            region_images = []
            for img_index, img in enumerate(page_image_list):
                try:
                    # Create region based on image position
                    if len(page_image_list) == 1:
                        # Single image - use most of the page
                        clip_rect = page_dim
                    else:
                        # Multiple images - create vertical sections
                        section_height = page_dim.height / len(page_image_list)
                        y0 = img_index * section_height
                        y1 = (img_index + 1) * section_height
                        clip_rect = pymupdf.Rect(0, y0, page_dim.width, y1)
                    # Render this region
                    zoom = dpi / 72.0
                    mat = pymupdf.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                    # Skip very small regions
                    # TODO: min_region_size è parametrico, come anche dpi
                    if pix.width < min_region_size or pix.height < min_region_size:
                        continue
                    # Convert PyMuPDF pixmap to PIL Image
                    img_data = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_data))
                    region_images.append(pil_image)

                except Exception as e:
                    current_logger.info(f"Error extracting region {img_index + 1}: {e}")
                    continue
            # If we have regions to stitch, combine them vertically
            if region_images:
                try:
                    # Calculate dimensions for the stitched image
                    max_width = max(img.width for img in region_images)
                    total_height = sum(img.height for img in region_images)
                    # Create rect_a new image with the combined dimensions
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
                    current_logger.info(f"Saved image: {filename} ({max_width}x{total_height})")
                    # Clean up
                    stitched_image.close()
                    for img in region_images:
                        img.close()

                except Exception as e:
                    current_logger.info(f"Error stitching regions for page {page_num + 1}: {e}")
                    for img in region_images:
                        img.close()

        doc.close()
        return page_images

    except Exception as e:
        current_logger.info(f"Error processing PDF for images: {e}")
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
    current_page = 1
    # Process each line
    for i, line in enumerate(lines):
        modified_lines.append(line)
        # Detect page breaks (common patterns in pymupdf4llm output)
        # This is heuristic-based and might need adjustment based on your PDFs
        if (line.strip() == '---' or  # Common page separator
                line.strip().startswith('---') or
                (line.strip() == '' and i > 0 and lines[i - 1].strip() == '') or  # Double empty lines
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
                 (line.strip() == '' and i < len(lines) - 1 and lines[i + 1].startswith('#')) or  # Before headers
                 (i > 0 and lines[i - 1].strip() == '' and line.strip() == ''))):  # At paragraph breaks

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