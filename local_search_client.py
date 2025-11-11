"""
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

            # Generate embeddings in batches to prevent memory overflow
            logger.info("Generating embeddings...")
            BATCH_SIZE = 10  # Process 10 documents at a time (reduced for low-memory environments)

            if len(texts) <= BATCH_SIZE:
                # Small batch - process all at once
                embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                )
            else:
                # Large batch - process in chunks
                logger.info(f"Processing {len(texts)} documents in batches of {BATCH_SIZE}...")

                current_batch_size = BATCH_SIZE
                i = 0

                while i < len(texts):
                    batch_end = min(i + current_batch_size, len(texts))
                    batch_texts = texts[i:batch_end]
                    batch_ids = ids[i:batch_end]
                    batch_metadatas = metadatas[i:batch_end]

                    batch_num = i // current_batch_size + 1
                    total_batches = (len(texts) + current_batch_size - 1) // current_batch_size
                    logger.info(f"  Batch {batch_num}/{total_batches}: Documents {i+1}-{batch_end} (batch size: {current_batch_size})")

                    try:
                        batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=True)

                        self.collection.upsert(
                            ids=batch_ids,
                            embeddings=batch_embeddings.tolist(),
                            documents=batch_texts,
                            metadatas=batch_metadatas,
                        )

                        # Add small delay between batches to allow ChromaDB to compact
                        if batch_end < len(texts):
                            import time
                            time.sleep(0.5)  # 500ms delay to ease compaction pressure

                        # Success - move to next batch
                        i = batch_end

                    except Exception as batch_error:
                        error_msg = str(batch_error).lower()

                        # Check if it's a SQLite "database or disk is full" error (Error code 13)
                        if 'database or disk is full' in error_msg or 'disk is full' in error_msg or 'code: 13' in error_msg:
                            logger.error(f"  ❌ SQLite database size limit reached: {batch_error}")
                            logger.error(f"  Document {i+1} is too large or database has hit SQLite size limits")
                            logger.error(f"  Document ID: {batch_ids[0] if batch_ids else 'unknown'}")
                            logger.error(f"  Document size: {len(batch_texts[0])} chars")

                            # Log the solution
                            logger.warning(f"  💡 SOLUTION: This document contains too much content.")
                            logger.warning(f"  The indexing will continue, but this large document will be skipped.")
                            logger.warning(f"  Consider reducing content size by truncating large files.")

                            # Skip this document and continue
                            i += 1

                        # Check if it's a ChromaDB compaction error
                        elif 'compaction' in error_msg or 'metadata segment' in error_msg or 'failed to apply logs' in error_msg:
                            logger.warning(f"  ⚠️ ChromaDB compaction error: {batch_error}")

                            # Reduce batch size significantly for compaction issues
                            if current_batch_size > 5:
                                current_batch_size = max(5, current_batch_size // 4)
                                logger.warning(f"  Reducing batch size to {current_batch_size} to ease compaction pressure...")
                                # Add delay to let ChromaDB recover
                                import time
                                time.sleep(2.0)
                                # Don't increment i - retry with smaller batch
                            elif current_batch_size > 1:
                                current_batch_size = 1
                                logger.warning(f"  Reducing to batch_size=1 with extended recovery delay...")
                                import time
                                time.sleep(5.0)  # Longer delay for recovery
                                # Don't increment i - retry
                            else:
                                # Already at batch_size=1, skip problematic document
                                logger.error(f"  ❌ Skipping document {i+1} - cannot resolve compaction error even with batch_size=1")
                                logger.error(f"  Document may have corrupted metadata: {batch_ids[0] if batch_ids else 'unknown'}")
                                i += 1

                        # Check if it's a memory error
                        elif 'memory' in error_msg or 'allocation' in error_msg:
                            # Reduce batch size and retry
                            if current_batch_size > 1:
                                current_batch_size = max(1, current_batch_size // 2)
                                logger.warning(f"  ⚠️ Memory error - reducing batch size to {current_batch_size} and retrying...")
                                # Don't increment i - retry with smaller batch
                            else:
                                # Can't reduce further - this document is too large
                                logger.error(f"  ❌ Cannot process document {i+1} even with batch_size=1: {batch_error}")
                                logger.error(f"  Document size: {len(batch_texts[0])} chars")
                                # Skip this document
                                i += 1
                        else:
                            # Non-memory/non-compaction error - raise it
                            raise batch_error

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
        """Get collection statistics with corruption recovery"""
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
            error_msg = str(e).lower()

            # Check if it's a ChromaDB corruption error
            if 'compaction' in error_msg or 'metadata segment' in error_msg or 'backfill' in error_msg:
                logger.error(f"⚠️ ChromaDB database appears corrupted: {e}")
                logger.error(f"Database location: {self.persist_directory}")
                logger.warning("💡 To fix: Delete the vector database and re-index your data")
                logger.warning(f"   Command: rm -rf {self.persist_directory}")

                return {
                    "error": "Database corrupted - needs reset",
                    "error_detail": str(e),
                    "document_count": 0,
                    "collection_count": 0,
                    "corrupted": True,
                    "fix_instructions": f"Delete {self.persist_directory} and re-index"
                }
            else:
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
