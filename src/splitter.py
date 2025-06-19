#!/usr/bin/env python3
"""
Markdown document Splitter.
Splits a .md document in chunks using LangChain.
"""
import os
import sys
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def markdown_splitter(text):
    """
    Split markdown content using headers
    
    Args:
        doc_content: String content of the markdown file
    
    Returns:
        List of Document objects with split content
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4")  
    ]
    
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    chunks = md_splitter.split_text(text)  
    
    # Adding a control for over-sized text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Adjust based on your content
        chunk_overlap=200,    # Keep some context between chunks
        length_function=len
    )

    # Apply to each markdown chunk
    final_chunks = []
    for chunk in chunks:
        size_controlled_chunks = text_splitter.split_documents([chunk])
        final_chunks.extend(size_controlled_chunks)

    return final_chunks

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <markdown_directory>")
        sys.exit(0)  # Fixed: was sys.exit[0] - needs parentheses
    
    all_splits = []
    md_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join('../', 'mddocs')
    
    # Fixed: was 'mddir' instead of 'md_dir'
    md_files = [md for md in os.listdir(md_dir) if md.lower().endswith('.md')]
    
    print(f"Found {len(md_files)} markdown files in {md_dir}")
    
    for md_file in md_files:
        print(f"Processing: {md_file}")
        
        # Fixed: Need to read the file content, not just pass the filename
        file_path = os.path.join(md_dir, md_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fixed: Added missing closing parenthesis and pass content instead of filename
            file_splits = markdown_splitter(content)
            
            # Add metadata about source file to each split
            for split in file_splits:
                split.metadata['source_file'] = md_file
                split.metadata['file_path'] = file_path
            
            all_splits.extend(file_splits)
            print(f"  - Created {len(file_splits)} chunks")
            
        except Exception as e:
            print(f"Error processing {md_file}: {e}")
    
    print(f"\nTotal chunks created: {len(all_splits)}")
    
    # Example: Print first few chunks to verify
    if all_splits:
        print("\nFirst chunk preview:")
        print("-" * 50)
        print(f"Content: {all_splits[0].page_content[:200]}...")
        print(f"Metadata: {all_splits[0].metadata}")
    
    return all_splits

if __name__ == "__main__":  # Fixed: was **name** instead of __name__
    splits = main()
