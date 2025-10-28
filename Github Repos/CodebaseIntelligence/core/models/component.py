from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime


class ComponentType(Enum):
    """Types of components across systems"""
    # Ab Initio
    INPUT_FILE = "Input_File"
    OUTPUT_FILE = "Output_File"
    LOOKUP_FILE = "Lookup_File"
    TRANSFORM = "Transform"
    JOIN = "Join"
    FILTER = "Filter"
    AGGREGATE = "Aggregate"
    SORT = "Sort"

    # Hadoop
    SPARK_JOB = "Spark_Job"
    PIG_SCRIPT = "Pig_Script"
    HIVE_QUERY = "Hive_Query"
    SQOOP_IMPORT = "Sqoop_Import"
    OOZIE_WORKFLOW = "Oozie_Workflow"
    OOZIE_COORDINATOR = "Oozie_Coordinator"
    SHELL_SCRIPT = "Shell_Script"

    # Databricks
    NOTEBOOK = "Notebook"
    DELTA_TABLE = "Delta_Table"
    SQL_QUERY = "SQL_Query"
    ADF_PIPELINE = "ADF_Pipeline"

    # Custom Documents
    DOCUMENT = "Document"

    # Generic
    DATABASE_TABLE = "Database_Table"
    API_CALL = "API_Call"
    UNKNOWN = "Unknown"


@dataclass
class DataFlow:
    """Represents data flow between components"""
    source_component_id: str
    target_component_id: str
    dataset_name: Optional[str] = None
    flow_type: str = "data"  # data, control, dependency
    transformation_applied: Optional[str] = None
    records_estimated: Optional[int] = None

    def __repr__(self):
        return f"{self.source_component_id} -> {self.target_component_id} [{self.dataset_name}]"


@dataclass
class Component:
    """
    Represents a processing component in any system
    Universal model for Ab Initio, Hadoop, and Databricks components
    """
    id: str
    name: str
    component_type: ComponentType
    system: str  # "abinitio", "hadoop", "databricks"

    # File/Location information
    file_path: Optional[str] = None
    line_number: Optional[int] = None

    # Parent process
    process_id: Optional[str] = None
    process_name: Optional[str] = None

    # Data information
    input_datasets: List[str] = field(default_factory=list)
    output_datasets: List[str] = field(default_factory=list)
    tables_read: List[str] = field(default_factory=list)
    tables_written: List[str] = field(default_factory=list)

    # Schema/Structure
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    dml_definition: Optional[str] = None

    # Business logic
    business_description: Optional[str] = None
    transformation_logic: Optional[str] = None
    filter_conditions: List[str] = field(default_factory=list)
    join_conditions: List[str] = field(default_factory=list)

    # Code
    source_code: Optional[str] = None
    code_snippet: Optional[str] = None

    # Key parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    key_parameter_value: Optional[str] = None  # e.g., lookup key

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Embedding for RAG
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "component_type": self.component_type.value,
            "system": self.system,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "process_id": self.process_id,
            "process_name": self.process_name,
            "input_datasets": self.input_datasets,
            "output_datasets": self.output_datasets,
            "tables_read": self.tables_read,
            "tables_written": self.tables_written,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "dml_definition": self.dml_definition,
            "business_description": self.business_description,
            "transformation_logic": self.transformation_logic,
            "filter_conditions": self.filter_conditions,
            "join_conditions": self.join_conditions,
            "source_code": self.source_code,
            "code_snippet": self.code_snippet,
            "parameters": self.parameters,
            "key_parameter_value": self.key_parameter_value,
            "depends_on": self.depends_on,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def get_summary(self) -> str:
        """Get human-readable summary for LLM context"""
        summary = f"Component: {self.name}\n"
        summary += f"Type: {self.component_type.value}\n"
        summary += f"System: {self.system}\n"

        if self.process_name:
            summary += f"Process: {self.process_name}\n"

        if self.input_datasets:
            summary += f"Inputs: {', '.join(self.input_datasets)}\n"

        if self.output_datasets:
            summary += f"Outputs: {', '.join(self.output_datasets)}\n"

        if self.tables_read:
            summary += f"Tables Read: {', '.join(self.tables_read)}\n"

        if self.tables_written:
            summary += f"Tables Written: {', '.join(self.tables_written)}\n"

        if self.business_description:
            summary += f"Description: {self.business_description}\n"

        if self.transformation_logic:
            summary += f"Logic: {self.transformation_logic}\n"

        return summary

    def __repr__(self):
        return f"<Component {self.system}:{self.name} ({self.component_type.value})>"
