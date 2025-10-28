"""
Search Client Factory
Creates either Azure AI Search or Local Search client based on configuration

This allows easy switching between:
- Azure AI Search (cloud, requires subscription)
- ChromaDB Local Search (free, runs locally)
"""

import os
from typing import Optional
from loguru import logger


class SearchClientFactory:
    """
    Factory to create search clients

    Automatically selects the right implementation based on:
    1. Environment variables
    2. Configuration
    3. Availability
    """

    @staticmethod
    def create_search_client(
        mode: str = "auto",
        azure_endpoint: Optional[str] = None,
        azure_key: Optional[str] = None,
        local_path: Optional[str] = None,
    ):
        """
        Create a search client

        Args:
            mode: "auto", "azure", or "local"
            azure_endpoint: Azure Search endpoint (optional)
            azure_key: Azure Search API key (optional)
            local_path: Local database path (optional)

        Returns:
            Search client instance (Azure or Local)
        """

        # Auto-detect mode
        if mode == "auto":
            # Check if Azure credentials are available
            azure_endpoint = azure_endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
            azure_key = azure_key or os.getenv("AZURE_SEARCH_API_KEY")

            if azure_endpoint and azure_key:
                logger.info("🔵 Azure AI Search credentials found - using Azure")
                mode = "azure"
            else:
                logger.info("💾 No Azure credentials - using Local Search (FREE)")
                mode = "local"

        # Create the appropriate client
        if mode == "azure":
            return SearchClientFactory._create_azure_client(azure_endpoint, azure_key)
        elif mode == "local":
            return SearchClientFactory._create_local_client(local_path)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'auto', 'azure', or 'local'")

    @staticmethod
    def _create_azure_client(endpoint: Optional[str], api_key: Optional[str]):
        """Create Azure AI Search client"""
        try:
            from services.azure_search.search_client import CodebaseSearchClient

            if not endpoint or not api_key:
                raise ValueError(
                    "Azure Search requires AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY"
                )

            logger.info(f"Initializing Azure AI Search at: {endpoint}")
            client = CodebaseSearchClient(endpoint=endpoint, api_key=api_key)

            logger.info("✓ Azure AI Search client created")
            return client

        except ImportError as e:
            logger.error(f"Azure AI Search libraries not available: {e}")
            logger.info("Falling back to Local Search...")
            return SearchClientFactory._create_local_client()
        except Exception as e:
            logger.error(f"Error creating Azure client: {e}")
            logger.info("Falling back to Local Search...")
            return SearchClientFactory._create_local_client()

    @staticmethod
    def _create_local_client(persist_directory: Optional[str] = None):
        """Create Local Search client (ChromaDB)"""
        try:
            from services.local_search.local_search_client import LocalSearchClient

            persist_directory = persist_directory or "./outputs/vector_db"

            logger.info(f"Initializing Local Search at: {persist_directory}")
            client = LocalSearchClient(persist_directory=persist_directory)

            logger.info("✓ Local Search client created (FREE, no API key needed)")
            return client

        except ImportError as e:
            logger.error(
                f"ChromaDB not available: {e}\n"
                "Install with: pip install chromadb sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Error creating Local client: {e}")
            raise


# Convenience function
def get_search_client(mode: str = "auto"):
    """
    Quick way to get a search client

    Usage:
        # Auto-detect (uses Azure if credentials available, otherwise local)
        client = get_search_client()

        # Force local (free)
        client = get_search_client("local")

        # Force Azure (requires credentials)
        client = get_search_client("azure")
    """
    return SearchClientFactory.create_search_client(mode=mode)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Search Client Factory - Example Usage")
    print("=" * 60)

    # Example 1: Auto-detect
    print("\n1. Auto-detect mode:")
    try:
        client = get_search_client("auto")
        print(f"   ✓ Client created: {type(client).__name__}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Example 2: Force local
    print("\n2. Force local mode (FREE):")
    try:
        client = get_search_client("local")
        print(f"   ✓ Client created: {type(client).__name__}")
        stats = client.get_stats()
        print(f"   Stats: {stats}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Example 3: Try Azure (will fall back to local if no credentials)
    print("\n3. Try Azure mode:")
    try:
        client = get_search_client("azure")
        print(f"   ✓ Client created: {type(client).__name__}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 60)
