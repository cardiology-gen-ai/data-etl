#!/usr/bin/env python3
"""
Vectorstore creator using Qdrant.
Creates and manages vector stores for similarity search.
"""
import sys
import os
from typing import List
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Updated imports to fix deprecation warnings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("⚠️  Installing langchain-huggingface...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-huggingface"])
    from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_qdrant import Qdrant
except ImportError:
    print("⚠️  Installing langchain-qdrant...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-qdrant"])
    from langchain_qdrant import Qdrant

def create_qdrant_vectorstore(chunks: List[Document],
                              embeddings_model,
                              collection_name: str = "cardio_protocols",
                              qdrant_url: str = "http://localhost:6333") -> Qdrant:
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
            # Get embedding dimension (assuming all-MiniLM-L6-v2 has 384 dimensions)
            sample_embedding = embeddings_model.embed_query("test")
            vector_size = len(sample_embedding)
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
        else:
            print(f"Collection {collection_name} already exists")
        
        # Create vector store
        vectorstore = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings_model,
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
        embeddings_vectors: Pre-computed embedding vectors (optional)
        vectorstore_type: Type of vector store ("qdrant" or "faiss")
        collection_name: Name for the collection/index
    """
    if not chunks:
        print("No chunks to store")
        return
    
    # Initialize the same embedding model used in embedder.py
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
            vectorstore = Qdrant(
                client=client,
                collection_name=collection_name,
                embeddings=embeddings_model,
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

def main():
    """
    Test function - can be used to test vectorstore functionality
    """
    if len(sys.argv) < 2:
        print("Usage: python vectorstore.py <test_query>")
        print("Example: python vectorstore.py 'cardiac protocols'")
        sys.exit(1)
    
    query = sys.argv[1]
    print(f"Testing similarity search with query: '{query}'")
    
    # Test similarity search
    results = similarity_search(query, k=3)
    
    if results:
        print("\nTop results:")
        print("-" * 50)
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content[:200]}...")
            print(f"   Metadata: {doc.metadata}")
            print()
    else:
        print("No results found or vectorstore not available")

if __name__ == "__main__":
    main()
