"""
AIzaSyDaqFj9wL0zSQEwiZTMfjWKl1TFtrE8BGw
Local Vector Search Client
FREE alternative to Azure AI Search using ChromaDB and sentence-transformers

Uses:
- ChromaDB: Free, local, persistent vector database
- sentence-transformers: Free embeddings (no API key needed)
- Runs completely offline
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB or sentence-transformers not installed. Run: pip install chromadb sentence-transformers")


class LocalSearchClient:
    """
    Local vector search client using ChromaDB

    FREE alternative to Azure AI Search
    - No API keys needed
    - Runs locally
    - Persistent storage
    - Fast vector search
    """

    def __init__(self, persist_directory: str = "./outputs/vector_db"):
        """
        Initialize local search client

        Args:
            persist_directory: Where to store the vector database
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB and sentence-transformers required. "
                "Install with: pip install chromadb sentence-transformers"
            )

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing local vector search at: {self.persist_directory}")

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Initialize embedding model (free, runs locally)
        logger.info("Loading sentence-transformers model (this may take a moment first time)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # This model is:
        # - FREE
        # - Lightweight (80MB)
        # - Fast
        # - Good quality (384 dimensions)
        # - Runs on CPU
        logger.info("✓ Embedding model loaded")

        self.collection = None

    def create_index(self, index_name: str = "codebase"):
        """
        Create or get collection (equivalent to Azure AI Search index)

        Args:
            index_name: Name of the collection/index
        """
        try:
            self.collection = self.client.get_or_create_collection(
                name=index_name,
                metadata={"description": "Codebase intelligence search"},
            )
            logger.info(f"✓ Collection '{index_name}' ready")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents into the vector database

        Args:
            documents: List of documents to index
                      Each doc should have: id, content, doc_type, system, metadata
        """
        if not self.collection:
            self.create_index()

        logger.info(f"Indexing {len(documents)} documents...")

        try:
            # CRITICAL FIX: Deduplicate documents by ID before processing
            # ChromaDB's upsert() requires unique IDs within a single batch
            # Keep the last occurrence of each ID
            seen_ids = {}
            for doc in documents:
                doc_id = doc.get("id", "")
                if doc_id:
                    seen_ids[doc_id] = doc

            unique_documents = list(seen_ids.values())

            if len(unique_documents) < len(documents):
                duplicates_removed = len(documents) - len(unique_documents)
                logger.warning(f"⚠️ Removed {duplicates_removed} duplicate documents with same IDs")

            # Extract components
            ids = []
            texts = []
            metadatas = []

            for doc in unique_documents:
                doc_id = doc.get("id", "")
                content = doc.get("content", "")

                if not doc_id or not content:
                    logger.warning(f"Skipping document with missing id or content")
                    continue

                ids.append(doc_id)
                texts.append(content)

                # Metadata (ChromaDB requires dict of strings/numbers/bools)
                metadata = {
                    "doc_type": str(doc.get("doc_type", "")),
                    "system": str(doc.get("system", "")),
                    "title": str(doc.get("title", ""))[:500],  # Limit length
                }

                # Add any additional metadata
                if "metadata" in doc and isinstance(doc["metadata"], dict):
                    for key, value in doc["metadata"].items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value

                metadatas.append(metadata)

            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

            # Upsert to collection (update existing or insert new)
            # This prevents "duplicate ID" errors when re-indexing
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
            )

            logger.info(f"✓ Indexed {len(ids)} documents successfully (upserted)")

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise

    def search(
        self,
        query: str,
        top: int = 5,
        filters: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Search the vector database

        Args:
            query: Search query
            top: Number of results to return
            filters: Optional filters (e.g., {"doc_type": "process"})

        Returns:
            Dict with "results" key containing list of search results
        """
        if not self.collection:
            logger.warning("No collection initialized")
            return {"results": [], "total": 0}

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]

            # Build where clause for filters
            where_clause = None
            if filters:
                where_clause = {k: {"$eq": v} for k, v in filters.items()}

            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top,
                where=where_clause,
            )

            # Format results
            search_results = []
            if results and "ids" in results and results["ids"]:
                for i, doc_id in enumerate(results["ids"][0]):
                    result = {
                        "id": doc_id,
                        "content": results["documents"][0][i] if "documents" in results else "",
                        "metadata": results["metadatas"][0][i] if "metadatas" in results else {},
                        "score": float(results["distances"][0][i]) if "distances" in results else 0.0,
                    }
                    search_results.append(result)

            return {
                "results": search_results,
                "total": len(search_results),
            }

        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {"results": [], "total": 0, "error": str(e)}

    def hybrid_search(
        self,
        query: str,
        top: int = 5,
        filters: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Hybrid search (semantic + keyword)

        For ChromaDB, we'll use semantic search (keyword search requires additional setup)

        Args:
            query: Search query
            top: Number of results
            filters: Optional filters

        Returns:
            Dict with "results" key containing list of search results
        """
        # For now, just use semantic search
        # ChromaDB doesn't have built-in keyword search like Azure AI Search
        return self.search(query, top, filters)

    def delete_index(self, index_name: str = "codebase"):
        """Delete the collection/index"""
        try:
            self.client.delete_collection(name=index_name)
            logger.info(f"✓ Deleted collection '{index_name}'")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.collection:
            return {
                "error": "No collection initialized",
                "document_count": 0,
                "collection_count": 0,
            }

        try:
            count = self.collection.count()
            return {
                "document_count": count,  # Use consistent key name
                "total_documents": count,  # Keep for backwards compatibility
                "collection_count": 1,
                "collection_name": self.collection.name,
                "persist_directory": str(self.persist_directory),
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "error": str(e),
                "document_count": 0,
                "collection_count": 0,
            }


# Convenience function for quick testing
def test_local_search():
    """Test the local search client"""
    print("Testing Local Search Client...")

    client = LocalSearchClient()
    client.create_index("test_index")

    # Test documents
    docs = [
        {
            "id": "doc1",
            "content": "This is a Python script that processes patient data",
            "doc_type": "component",
            "system": "hadoop",
            "title": "Patient Data Processor",
        },
        {
            "id": "doc2",
            "content": "SQL query to extract hospital information from database",
            "doc_type": "component",
            "system": "hadoop",
            "title": "Hospital Query",
        },
        {
            "id": "doc3",
            "content": "Data transformation pipeline for claims processing",
            "doc_type": "process",
            "system": "databricks",
            "title": "Claims Pipeline",
        },
    ]

    # Index
    client.index_documents(docs)

    # Search
    results = client.search("patient data", top=2)
    print(f"\nSearch results for 'patient data':")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['metadata'].get('title', 'N/A')} (score: {result['score']:.4f})")
        print(f"      {result['content'][:80]}...")

    # Stats
    stats = client.get_stats()
    print(f"\nStats: {stats}")

    # Cleanup
    client.delete_index("test_index")
    print("\n✓ Test completed")


if __name__ == "__main__":
    test_local_search()
