"""
Azure AI Search Client
Handles indexing and searching of parsed codebase data
"""

import os
from typing import List, Dict, Any, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)
from loguru import logger
import openai


class CodebaseSearchClient:
    """Azure AI Search client for codebase intelligence"""

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        self.endpoint = endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "codebase-intelligence")

        if not self.endpoint or not self.api_key:
            raise ValueError("Azure Search endpoint and API key must be provided")

        self.credential = AzureKeyCredential(self.api_key)
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint, credential=self.credential
        )
        self.search_client = None

        # Setup OpenAI for embeddings
        openai.api_type = "azure"
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.embedding_deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
        )

    def create_index(self):
        """Create search index with vector fields"""
        logger.info(f"Creating search index: {self.index_name}")

        fields = [
            SearchField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchField(
                name="doc_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),  # process, component, sttm, gap
            SearchField(
                name="system",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),  # abinitio, hadoop, databricks
            SearchField(
                name="name",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
            ),
            SearchField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchField(
                name="description",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchField(
                name="code_snippet",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchField(
                name="process_name",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SearchField(
                name="component_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SearchField(
                name="tables",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
            ),
            SearchField(
                name="tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True,
            ),
            SearchField(
                name="metadata",
                type=SearchFieldDataType.String,
                filterable=False,
            ),
            # Vector field for semantic search
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # Ada-002 dimensions
                vector_search_profile_name="vector-profile",
            ),
        ]

        # Vector search configuration
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config",
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(name="hnsw-config")
            ],
        )

        # Semantic search configuration
        semantic_config = SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="name"),
                content_fields=[SemanticField(field_name="content")],
                keywords_fields=[
                    SemanticField(field_name="tags"),
                    SemanticField(field_name="component_type"),
                ],
            ),
        )

        semantic_search = SemanticSearch(configurations=[semantic_config])

        # Create index
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )

        try:
            self.index_client.create_or_update_index(index)
            logger.info(f"Index '{self.index_name}' created successfully")

            # Initialize search client
            self.search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self.credential,
            )

        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Azure OpenAI"""
        try:
            response = openai.Embedding.create(
                engine=self.embedding_deployment,
                input=text,
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 1536  # Return zero vector on error

    def index_process(self, process: Any):
        """Index a process document"""
        content = f"{process.name}\n{process.description or ''}\n{process.business_function or ''}"

        document = {
            "id": process.id,
            "doc_type": "process",
            "system": process.system.value,
            "name": process.name,
            "content": content,
            "description": process.description or "",
            "process_name": process.name,
            "tables": process.tables_involved,
            "tags": process.tags or [],
            "metadata": str(process.metadata or {}),
            "content_vector": self.generate_embedding(content),
        }

        self._upload_document(document)

    def index_component(self, component: Any):
        """Index a component document"""
        content = (
            f"{component.name}\n"
            f"{component.business_description or ''}\n"
            f"{component.transformation_logic or ''}"
        )

        document = {
            "id": component.id,
            "doc_type": "component",
            "system": component.system,
            "name": component.name,
            "content": content,
            "description": component.business_description or "",
            "code_snippet": component.code_snippet or "",
            "process_name": component.process_name or "",
            "component_type": component.component_type.value,
            "tables": component.tables_read + component.tables_written,
            "tags": component.tags or [],
            "metadata": str(component.metadata or {}),
            "content_vector": self.generate_embedding(content),
        }

        self._upload_document(document)

    def index_sttm_mapping(self, mapping: Any):
        """Index STTM mapping"""
        content = (
            f"Mapping: {mapping.source_column} -> {mapping.target_column}\n"
            f"Source: {mapping.source_table}.{mapping.source_column}\n"
            f"Target: {mapping.target_table}.{mapping.target_column}\n"
            f"Transformation: {mapping.transformation_summary or ''}"
        )

        document = {
            "id": mapping.mapping_id,
            "doc_type": "sttm",
            "system": mapping.system,
            "name": f"{mapping.source_column} -> {mapping.target_column}",
            "content": content,
            "description": f"STTM mapping for {mapping.process_name}",
            "process_name": mapping.process_name,
            "tables": [mapping.source_table, mapping.target_table],
            "tags": mapping.tags or [],
            "metadata": str(mapping.metadata or {}),
            "content_vector": self.generate_embedding(content),
        }

        self._upload_document(document)

    def index_gap(self, gap: Any):
        """Index gap analysis result"""
        content = (
            f"Gap: {gap.title}\n"
            f"{gap.description}\n"
            f"Impact: {gap.business_impact or ''}\n"
            f"Recommendation: {gap.recommendation or ''}"
        )

        document = {
            "id": gap.gap_id,
            "doc_type": "gap",
            "system": gap.source_system,
            "name": gap.title,
            "content": content,
            "description": gap.description,
            "process_name": gap.source_process_name or "",
            "tags": [gap.gap_type.value, gap.severity.value] + (gap.tags or []),
            "metadata": str(gap.metadata or {}),
            "content_vector": self.generate_embedding(content),
        }

        self._upload_document(document)

    def _upload_document(self, document: Dict):
        """Upload single document to index"""
        if not self.search_client:
            self.search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self.credential,
            )

        try:
            self.search_client.upload_documents(documents=[document])
            logger.debug(f"Indexed document: {document['id']}")
        except Exception as e:
            logger.error(f"Error uploading document {document['id']}: {e}")

    def search(
        self,
        query: str,
        filters: Optional[str] = None,
        top: int = 5,
        use_semantic: bool = True,
    ) -> List[Dict]:
        """
        Search the index

        Args:
            query: Search query
            filters: OData filter string (e.g., "doc_type eq 'process' and system eq 'abinitio'")
            top: Number of results to return
            use_semantic: Use semantic search
        """
        if not self.search_client:
            self.search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self.credential,
            )

        # Generate query vector
        query_vector = self.generate_embedding(query)

        search_params = {
            "search_text": query,
            "vector_queries": [
                {
                    "vector": query_vector,
                    "k_nearest_neighbors": top,
                    "fields": "content_vector",
                }
            ],
            "select": [
                "id",
                "doc_type",
                "system",
                "name",
                "content",
                "description",
                "process_name",
                "component_type",
                "tables",
                "tags",
            ],
            "top": top,
        }

        if filters:
            search_params["filter"] = filters

        if use_semantic:
            search_params["query_type"] = "semantic"
            search_params["semantic_configuration_name"] = "semantic-config"

        try:
            results = self.search_client.search(**search_params)
            return [dict(result) for result in results]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        doc_types: Optional[List[str]] = None,
        systems: Optional[List[str]] = None,
        top: int = 5,
    ) -> List[Dict]:
        """Hybrid search with filters"""
        filters = []

        if doc_types:
            type_filters = " or ".join([f"doc_type eq '{dt}'" for dt in doc_types])
            filters.append(f"({type_filters})")

        if systems:
            sys_filters = " or ".join([f"system eq '{s}'" for s in systems])
            filters.append(f"({sys_filters})")

        filter_str = " and ".join(filters) if filters else None

        return self.search(query, filters=filter_str, top=top)
