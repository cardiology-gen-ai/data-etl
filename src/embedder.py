#!/usr/bin/env python3
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
    # Initialize embedding model
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
    
    # Extract text content from chunks for embedding
    chunk_texts = [chunk.page_content for chunk in chunks]
    
    # Generate embeddings
    print(f"Generating embeddings for {len(chunk_texts)} chunks...")
    embeddings = embedding_model.embed_documents(chunk_texts)
    
    print(f"Generated {len(embeddings)} embeddings")
    
    return chunks, embeddings

def process_file(file_path):
    """
    Process a single markdown file: split, embed, and store.
    
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
    Process all markdown files in a directory.
    
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

def main():
    if len(sys.argv) != 2:
        print("Usage: python embedder.py <markdown_directory>")
        sys.exit(1)
    
    md_dir = sys.argv[1]
    chunks, embeddings = process_directory(md_dir)
    
    return chunks, embeddings

if __name__ == "__main__":
    chunks, embeddings = main()
