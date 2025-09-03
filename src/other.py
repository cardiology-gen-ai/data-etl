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


def create_qdrant_vectorstore(chunks: List[Document],
                              embeddings_model,
                              collection_name: str = "cardio_protocols",  # TODO: config [forse direttamente app_id]
                              qdrant_url: str = "http://localhost:6333") -> QdrantVectorStore:
    """
    Create Qdrant vector store (recommended for sharing)

    Args:
        chunks: List of chunked documents
        embeddings_model: HuggingFaceEmbeddings model instance
        collection_name: Name for the Qdrant collection
        qdrant_url: URL for Qdrant server

    Returns:
        Qdrant vector store object
    """
    print(f"Creating Qdrant vector store at {qdrant_url}")
    print(f"Collection name: {collection_name}")

    try:
        # Test connection to Qdrant server
        client = QdrantClient(url=qdrant_url)

        # Check if collection exists, if not create it
        collections = client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)

        if not collection_exists:
            print(f"Creating new collection: {collection_name}")
            # Get embeddings dimension (assuming all-MiniLM-L6-v2 has 384 dimensions)
            sample_embedding = embeddings_model.embed_query("test")
            vector_size = len(sample_embedding)

            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
        else:
            print(f"Collection {collection_name} already exists")

        # Create vector store
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings_model,
        )

        # Add documents
        if chunks:
            print(f"Adding {len(chunks)} documents to Qdrant...")
            vectorstore.add_documents(chunks)
            print(f"Successfully added documents to Qdrant vector store")

        return vectorstore

    except Exception as e:
        print(f"Error creating Qdrant vector store: {e}")
        print("Make sure Qdrant server is running on the specified URL")
        raise


def create_faiss_vectorstore(chunks: List[Document],
                             embeddings_model,
                             save_path: str = "./faiss_index") -> None:
    """
    Create FAISS vector store (fallback option)

    Args:
        chunks: List of chunked documents
        embeddings_model: HuggingFaceEmbeddings model instance
        save_path: Path to save FAISS index
    """
    try:
        from langchain_community.vectorstores import FAISS

        print("Creating FAISS vector store...")

        if not chunks:
            print("No chunks provided for FAISS vectorstore")
            return

        # Create FAISS vector store
        vectorstore = FAISS.from_documents(chunks, embeddings_model)

        # Save the vector store
        vectorstore.save_local(save_path)
        print(f"FAISS vector store saved to {save_path}")

        return vectorstore

    except ImportError:
        print("FAISS not available. Install with: pip install faiss-cpu")
        raise
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        raise


def store_embeddings(chunks: List[Document],
                     embeddings_vectors: List[List[float]] = None,
                     vectorstore_type: str = "qdrant",
                     collection_name: str = "cardio_protocols") -> None:
    """
    Store embeddings in the specified vector store.
    This function is called by embedder.py

    Args:
        chunks: List of chunked documents
        embeddings_vectors: Pre-computed embeddings vectors (optional)
        vectorstore_type: Type of vector store ("qdrant" or "faiss")
        collection_name: Name for the collection/index
    """
    if not chunks:
        print("No chunks to store")
        return

    # Initialize the same embeddings model used in embedder.py
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    try:
        if vectorstore_type.lower() == "qdrant":
            vectorstore = create_qdrant_vectorstore(
                chunks=chunks,
                embeddings_model=embeddings_model,
                collection_name=collection_name
            )
            print("Embeddings successfully stored in Qdrant")

        elif vectorstore_type.lower() == "faiss":
            vectorstore = create_faiss_vectorstore(
                chunks=chunks,
                embeddings_model=embeddings_model,
                save_path=f"./faiss_{collection_name}"
            )
            print("Embeddings successfully stored in FAISS")

        else:
            print(f"Unknown vectorstore type: {vectorstore_type}")
            print("Supported types: 'qdrant', 'faiss'")

    except Exception as e:
        print(f"Failed to store embeddings with {vectorstore_type}: {e}")
        if vectorstore_type.lower() == "qdrant":
            print("Falling back to FAISS...")
            try:
                create_faiss_vectorstore(
                    chunks=chunks,
                    embeddings_model=embeddings_model,
                    save_path=f"./faiss_{collection_name}_fallback"
                )
                print("Fallback to FAISS successful")
            except Exception as fallback_error:
                print(f"Fallback to FAISS also failed: {fallback_error}")


def load_vectorstore(collection_name: str = "cardio_protocols",
                     vectorstore_type: str = "qdrant",
                     qdrant_url: str = "http://localhost:6333",
                     faiss_path: str = "./faiss_index"):
    """
    Load an existing vector store for similarity search

    Args:
        collection_name: Name of the collection
        vectorstore_type: Type of vector store ("qdrant" or "faiss")
        qdrant_url: URL for Qdrant server
        faiss_path: Path to FAISS index

    Returns:
        Loaded vector store object
    """
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    try:
        if vectorstore_type.lower() == "qdrant":
            client = QdrantClient(url=qdrant_url)
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings_model,
            )
            print(f"Loaded Qdrant vectorstore: {collection_name}")
            return vectorstore

        elif vectorstore_type.lower() == "faiss":
            from langchain_community.vectorstores import FAISS
            vectorstore = FAISS.load_local(faiss_path, embeddings_model)
            print(f"Loaded FAISS vectorstore from: {faiss_path}")
            return vectorstore

        else:
            print(f"Unknown vectorstore type: {vectorstore_type}")
            return None

    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None


def similarity_search(query: str,
                      collection_name: str = "cardio_protocols",
                      vectorstore_type: str = "qdrant",
                      k: int = 5):
    """
    Perform similarity search on the vector store

    Args:
        query: Search query
        collection_name: Name of the collection
        vectorstore_type: Type of vector store
        k: Number of results to return

    Returns:
        List of similar documents
    """
    vectorstore = load_vectorstore(collection_name, vectorstore_type)

    if vectorstore is None:
        print("Could not load vectorstore")
        return []

    try:
        results = vectorstore.similarity_search(query, k=k)
        print(f"Found {len(results)} similar documents for query: '{query}'")
        return results

    except Exception as e:
        print(f"Error performing similarity search: {e}")
        return []


"""
Embedder of .md files using Sentence Transformers.
Embeddings are then stored for similarity search.
"""
import sys
import os
from splitter import markdown_splitter
from vectorstore import store_embeddings

# Updated import to fix deprecation warning
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("⚠️  Installing langchain-huggingface...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-huggingface"])
    from langchain_huggingface import HuggingFaceEmbeddings


def embedder(text, source_file_name=None):
    """
    Embed markdown content after splitting it into chunks.

    Args:
        text: String content of the markdown file
        source_file_name: Optional name of the source file for metadata

    Returns:
        tuple: (chunks, embeddings) where chunks are Document objects and embeddings are vectors
    """
    # Initialize embeddings model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # use 'cuda' if you have Nvidia GPU, 'mps' for Apple Silicon
    )

    # Split text into chunks
    chunks = markdown_splitter(text)

    # Add source file information to metadata if provided
    if source_file_name:
        for chunk in chunks:
            chunk.metadata['source_file'] = source_file_name

    # Extract text content from chunks for embeddings
    chunk_texts = [chunk.page_content for chunk in chunks]

    # Generate embeddings
    print(f"Generating embeddings for {len(chunk_texts)} chunks...")
    embeddings = embedding_model.embed_documents(chunk_texts)

    print(f"Generated {len(embeddings)} embeddings")

    return chunks, embeddings


def process_file(file_path):
    """
    Process rect_a single markdown file: split, embed, and store.

    Args:
        file_path: Path to the markdown file

    Returns:
        tuple: (chunks, embeddings)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")

        chunks, embeddings = embedder(content, file_name)

        # Store in vector database
        store_embeddings(chunks, embeddings)

        return chunks, embeddings

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], []


def process_directory(md_dir):
    """
    Process all markdown files in rect_a directory.

    Args:
        md_dir: Directory containing markdown files

    Returns:
        tuple: (all_chunks, all_embeddings)
    """
    if not os.path.exists(md_dir):
        print(f"Directory {md_dir} does not exist")
        return [], []

    md_files = [f for f in os.listdir(md_dir) if f.lower().endswith('.md')]

    if not md_files:
        print(f"No markdown files found in {md_dir}")
        return [], []

    print(f"Found {len(md_files)} markdown files in {md_dir}")

    all_chunks = []
    all_embeddings = []

    for md_file in md_files:
        file_path = os.path.join(md_dir, md_file)
        chunks, embeddings = process_file(file_path)
        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)

    print(f"\nTotal processing complete:")
    print(f"  - Total chunks: {len(all_chunks)}")
    print(f"  - Total embeddings: {len(all_embeddings)}")

    return all_chunks, all_embeddings