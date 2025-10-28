"""
Codebase Indexer - Converts parsed codebase into vector database documents
"""

from typing import List, Dict, Any
from pathlib import Path
import json
from loguru import logger

from core.models import Process, Component
from services.local_search.local_search_client import LocalSearchClient


class CodebaseIndexer:
    """
    Indexes parsed codebase data into vector database
    Converts processes, components, STTM, and gaps into searchable documents
    """

    def __init__(self, vector_db_path: str = "./outputs/vector_db"):
        self.search_client = LocalSearchClient(persist_directory=vector_db_path)
        self.indexed_count = 0

    def index_from_analysis(
        self,
        processes: List[Process],
        components: List[Component],
        sttm_data: List[Dict[str, Any]] = None,
        gap_data: List[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Index all analysis data

        Args:
            processes: List of Process objects
            components: List of Component objects
            sttm_data: Optional STTM mappings
            gap_data: Optional gap analysis data

        Returns:
            Dict with counts of indexed documents by type
        """
        logger.info("Starting codebase indexing...")

        documents = []
        counts = {"processes": 0, "components": 0, "sttm": 0, "gaps": 0}

        # Index processes
        process_docs = self._create_process_documents(processes)
        documents.extend(process_docs)
        counts["processes"] = len(process_docs)

        # Index components
        component_docs = self._create_component_documents(components)
        documents.extend(component_docs)
        counts["components"] = len(component_docs)

        # Index STTM if provided
        if sttm_data:
            sttm_docs = self._create_sttm_documents(sttm_data)
            documents.extend(sttm_docs)
            counts["sttm"] = len(sttm_docs)

        # Index gaps if provided
        if gap_data:
            gap_docs = self._create_gap_documents(gap_data)
            documents.extend(gap_docs)
            counts["gaps"] = len(gap_docs)

        # Index all documents
        logger.info(f"Indexing {len(documents)} total documents...")
        self.search_client.index_documents(documents)

        self.indexed_count = len(documents)
        logger.info(f"✓ Successfully indexed {self.indexed_count} documents")

        return counts

    def _create_process_documents(self, processes: List[Process]) -> List[Dict[str, Any]]:
        """Convert Process objects to searchable documents"""
        documents = []

        for process in processes:
            # Create main content for search
            content_parts = [
                f"Process: {process.name}",
                f"System: {process.system.value}",
                f"Type: {process.process_type.value}",
                f"Description: {process.description or 'N/A'}",
                f"Business Function: {process.business_function or 'N/A'}",
                f"File Path: {process.file_path}",
            ]

            # Add parameters
            if process.parameters:
                content_parts.append("\nParameters:")
                for key, value in list(process.parameters.items())[:10]:  # First 10 params
                    content_parts.append(f"  - {key}: {value}")

            # Add component info
            content_parts.append(f"\nComponents: {process.component_count} components")

            content = "\n".join(content_parts)

            # Create document (id must be at TOP LEVEL for ChromaDB)
            doc = {
                "id": process.id,  # TOP LEVEL - required by ChromaDB
                "content": content,
                "metadata": {
                    "doc_type": "process",
                    "name": process.name,
                    "system": process.system.value,
                    "process_type": process.process_type.value,
                    "file_path": process.file_path,
                    "business_function": process.business_function or "",
                    "component_count": process.component_count,
                },
            }

            documents.append(doc)

        return documents

    def _create_component_documents(self, components: List[Component]) -> List[Dict[str, Any]]:
        """Convert Component objects to searchable documents"""
        documents = []

        for component in components:
            # Create main content
            content_parts = [
                f"Component: {component.name}",
                f"Type: {component.component_type.value}",
                f"System: {component.system}",
                f"Process: {component.process_name}",
                f"Description: {component.business_description or 'N/A'}",
            ]

            # Add input datasets
            if component.input_datasets:
                content_parts.append(f"\nInput Datasets: {', '.join(component.input_datasets[:5])}")

            # Add output datasets
            if component.output_datasets:
                content_parts.append(f"Output Datasets: {', '.join(component.output_datasets[:5])}")

            # Add transformation logic
            if component.transformation_logic:
                content_parts.append(f"\nTransformation Logic:\n{component.transformation_logic[:300]}")

            # Add DML definition
            if component.dml_definition:
                content_parts.append(f"\nDML Definition:\n{component.dml_definition[:300]}")

            content = "\n".join(content_parts)

            # Create document (id must be at TOP LEVEL for ChromaDB)
            doc = {
                "id": component.id,  # TOP LEVEL - required by ChromaDB
                "content": content,
                "metadata": {
                    "doc_type": "component",
                    "name": component.name,
                    "system": component.system,
                    "component_type": component.component_type.value,
                    "process_name": component.process_name,
                    "process_id": component.process_id,
                    "file_path": component.file_path,
                },
            }

            documents.append(doc)

        return documents

    def _create_sttm_documents(self, sttm_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert STTM mappings to searchable documents"""
        documents = []

        for sttm in sttm_data:
            # Create content
            content_parts = [
                f"STTM Mapping: {sttm.get('mapping_id', 'N/A')}",
                f"Source Column: {sttm.get('source_column_name', 'N/A')}",
                f"Source Table: {sttm.get('source_table', 'N/A')}",
                f"Source System: {sttm.get('source_system', 'N/A')}",
                f"Target Column: {sttm.get('target_column_name', 'N/A')}",
                f"Target Table: {sttm.get('target_table', 'N/A')}",
                f"Target System: {sttm.get('target_system', 'N/A')}",
                f"Transformation: {sttm.get('transformation_rule', 'N/A')}",
                f"Data Type: {sttm.get('source_datatype', 'N/A')} → {sttm.get('target_datatype', 'N/A')}",
            ]

            content = "\n".join(content_parts)

            # Create document (id must be at TOP LEVEL for ChromaDB)
            doc = {
                "id": sttm.get("mapping_id", f"sttm_{len(documents)}"),  # TOP LEVEL
                "content": content,
                "metadata": {
                    "doc_type": "sttm",
                    "name": f"{sttm.get('source_column_name', '')} → {sttm.get('target_column_name', '')}",
                    "source_system": sttm.get("source_system", ""),
                    "target_system": sttm.get("target_system", ""),
                    "source_table": sttm.get("source_table", ""),
                    "target_table": sttm.get("target_table", ""),
                },
            }

            documents.append(doc)

        return documents

    def _create_gap_documents(self, gap_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert gap analysis data to searchable documents"""
        documents = []

        for gap in gap_data:
            # Create content
            content_parts = [
                f"Gap: {gap.get('gap_type', 'N/A')}",
                f"Name: {gap.get('name', 'N/A')}",
                f"Description: {gap.get('description', 'N/A')}",
                f"Severity: {gap.get('severity', 'N/A')}",
                f"Source System: {gap.get('source_system', 'N/A')}",
                f"Target System: {gap.get('target_system', 'N/A')}",
                f"Impact: {gap.get('impact', 'N/A')}",
                f"Recommendation: {gap.get('recommendation', 'N/A')}",
            ]

            content = "\n".join(content_parts)

            # Create document (id must be at TOP LEVEL for ChromaDB)
            doc = {
                "id": gap.get("gap_id", f"gap_{len(documents)}"),  # TOP LEVEL
                "content": content,
                "metadata": {
                    "doc_type": "gap",
                    "name": gap.get("name", ""),
                    "gap_type": gap.get("gap_type", ""),
                    "severity": gap.get("severity", ""),
                    "source_system": gap.get("source_system", ""),
                    "target_system": gap.get("target_system", ""),
                },
            }

            documents.append(doc)

        return documents

    def index_from_json(self, json_file_path: str) -> Dict[str, int]:
        """
        Index from JSON file (useful for testing)

        Args:
            json_file_path: Path to JSON file with parsed data

        Returns:
            Dict with counts
        """
        logger.info(f"Loading data from {json_file_path}...")

        with open(json_file_path, "r") as f:
            data = json.load(f)

        # Convert JSON to Process/Component objects if needed
        # For now, create simple documents from JSON
        documents = []

        if isinstance(data, list):
            # List of items
            for idx, item in enumerate(data):
                doc = {
                    "id": item.get("id", f"json_item_{idx}"),  # TOP LEVEL
                    "content": json.dumps(item, indent=2),
                    "metadata": {
                        "doc_type": item.get("type", "unknown"),
                        "name": item.get("name", "unknown"),
                        "source": "json_file",
                    },
                }
                documents.append(doc)
        elif isinstance(data, dict):
            # Single object or structured data
            doc_idx = 0
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        doc = {
                            "id": item.get("id", f"json_{key}_{doc_idx}"),  # TOP LEVEL
                            "content": json.dumps(item, indent=2),
                            "metadata": {
                                "doc_type": key,
                                "source": "json_file",
                            },
                        }
                        documents.append(doc)
                        doc_idx += 1

        logger.info(f"Indexing {len(documents)} documents from JSON...")
        self.search_client.index_documents(documents)

        return {"total": len(documents)}

    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        stats = self.search_client.get_stats()
        stats["indexed_by_this_session"] = self.indexed_count
        return stats

    def clear_index(self):
        """Clear all indexed documents (use with caution!)"""
        logger.warning("Clearing vector database...")
        # This would need to be implemented in LocalSearchClient
        # For now, just log a warning
        logger.warning("Clear index not fully implemented - delete the vector_db directory manually")
