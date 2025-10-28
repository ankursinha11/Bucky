from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime


class SystemType(Enum):
    """System types"""
    ABINITIO = "abinitio"
    HADOOP = "hadoop"
    DATABRICKS = "databricks"


class ProcessType(Enum):
    """Types of processes"""
    # Ab Initio
    GRAPH = "graph"

    # Hadoop
    OOZIE_WORKFLOW = "oozie_workflow"
    OOZIE_COORDINATOR = "oozie_coordinator"

    # Databricks
    NOTEBOOK = "notebook"
    ADF_PIPELINE = "adf_pipeline"

    # Generic
    DATA_INGESTION = "data_ingestion"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    LEAD_GENERATION = "lead_generation"
    RECONCILIATION = "reconciliation"
    UNKNOWN = "unknown"


@dataclass
class Process:
    """
    Represents a complete processing workflow/graph/pipeline
    Top-level entity that contains multiple components
    """
    id: str
    name: str
    system: SystemType
    process_type: ProcessType

    # File information
    file_path: Optional[str] = None
    repo_name: Optional[str] = None

    # Description
    description: Optional[str] = None
    business_function: Optional[str] = None

    # Components
    component_ids: List[str] = field(default_factory=list)
    component_count: int = 0

    # Data flow
    input_sources: List[str] = field(default_factory=list)
    output_targets: List[str] = field(default_factory=list)
    tables_involved: List[str] = field(default_factory=list)

    # Execution
    schedule: Optional[str] = None  # e.g., "every 3 hours", "daily"
    execution_order: Optional[int] = None
    estimated_runtime: Optional[str] = None

    # Dependencies
    depends_on_processes: List[str] = field(default_factory=list)
    triggers_processes: List[str] = field(default_factory=list)

    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    graph_parameters: Dict[str, Any] = field(default_factory=dict)

    # Business context
    business_domain: Optional[str] = None  # e.g., "Lead Discovery", "CDD", "GMRN"
    functional_area: Optional[str] = None  # e.g., "Patient Matching", "Coverage Discovery"

    # Data statistics
    record_volume: Optional[int] = None
    data_sources: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Embedding for RAG
    embedding: Optional[List[float]] = None

    # Mapping to other systems
    equivalent_processes: Dict[str, str] = field(default_factory=dict)  # {system: process_id}
    mapping_confidence: Dict[str, float] = field(default_factory=dict)  # {system: confidence}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "system": self.system.value,
            "process_type": self.process_type.value,
            "file_path": self.file_path,
            "repo_name": self.repo_name,
            "description": self.description,
            "business_function": self.business_function,
            "component_ids": self.component_ids,
            "component_count": self.component_count,
            "input_sources": self.input_sources,
            "output_targets": self.output_targets,
            "tables_involved": self.tables_involved,
            "schedule": self.schedule,
            "execution_order": self.execution_order,
            "estimated_runtime": self.estimated_runtime,
            "depends_on_processes": self.depends_on_processes,
            "triggers_processes": self.triggers_processes,
            "parameters": self.parameters,
            "graph_parameters": self.graph_parameters,
            "business_domain": self.business_domain,
            "functional_area": self.functional_area,
            "record_volume": self.record_volume,
            "data_sources": self.data_sources,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "equivalent_processes": self.equivalent_processes,
            "mapping_confidence": self.mapping_confidence,
        }

    def get_summary(self) -> str:
        """Get human-readable summary"""
        summary = f"Process: {self.name}\n"
        summary += f"System: {self.system.value}\n"
        summary += f"Type: {self.process_type.value}\n"

        if self.description:
            summary += f"Description: {self.description}\n"

        if self.business_function:
            summary += f"Business Function: {self.business_function}\n"

        if self.business_domain:
            summary += f"Domain: {self.business_domain}\n"

        if self.input_sources:
            summary += f"Inputs: {', '.join(self.input_sources[:5])}\n"

        if self.output_targets:
            summary += f"Outputs: {', '.join(self.output_targets[:5])}\n"

        if self.tables_involved:
            summary += f"Tables: {', '.join(self.tables_involved[:5])}\n"

        if self.schedule:
            summary += f"Schedule: {self.schedule}\n"

        summary += f"Components: {self.component_count}\n"

        return summary

    def __repr__(self):
        return f"<Process {self.system.value}:{self.name} ({self.process_type.value})>"
