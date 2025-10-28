from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime


class DataType(Enum):
    """Common data types across systems"""
    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    FLOAT = "float"
    DOUBLE = "double"
    BOOLEAN = "boolean"
    DATE = "date"
    TIMESTAMP = "timestamp"
    BINARY = "binary"
    ARRAY = "array"
    STRUCT = "struct"
    MAP = "map"
    UNKNOWN = "unknown"

    @classmethod
    def from_abinitio(cls, abi_type: str) -> "DataType":
        """Convert Ab Initio type to common type"""
        type_map = {
            "decimal": cls.DECIMAL,
            "string": cls.STRING,
            "integer": cls.INTEGER,
            "date": cls.DATE,
            "datetime": cls.TIMESTAMP,
            "real": cls.FLOAT,
        }
        return type_map.get(abi_type.lower(), cls.UNKNOWN)

    @classmethod
    def from_hive(cls, hive_type: str) -> "DataType":
        """Convert Hive type to common type"""
        type_map = {
            "string": cls.STRING,
            "int": cls.INTEGER,
            "bigint": cls.INTEGER,
            "double": cls.DOUBLE,
            "float": cls.FLOAT,
            "boolean": cls.BOOLEAN,
            "date": cls.DATE,
            "timestamp": cls.TIMESTAMP,
            "binary": cls.BINARY,
            "array": cls.ARRAY,
            "struct": cls.STRUCT,
            "map": cls.MAP,
        }
        return type_map.get(hive_type.lower(), cls.UNKNOWN)


@dataclass
class TransformationRule:
    """Represents a transformation rule applied to data"""
    rule_id: str
    rule_type: str  # "cast", "concatenate", "lookup", "calculation", "filter", etc.
    description: str
    expression: Optional[str] = None
    function_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"<TransformationRule {self.rule_type}: {self.description}>"


@dataclass
class ColumnMapping:
    """
    Represents a single column mapping from source to target
    Core model for STTM (Source-to-Target Mapping)
    """
    mapping_id: str

    # Process/Component context
    process_id: str
    process_name: str
    component_id: Optional[str] = None
    component_name: Optional[str] = None
    system: str = "cross-system"  # abinitio, hadoop, databricks, cross-system

    # Source information
    source_system: Optional[str] = None
    source_table: Optional[str] = None
    source_column: str = ""
    source_datatype: Optional[DataType] = None
    source_datatype_raw: Optional[str] = None  # Original type as written in code
    source_length: Optional[int] = None
    source_precision: Optional[int] = None
    source_scale: Optional[int] = None
    source_description: Optional[str] = None
    source_sample_values: List[str] = field(default_factory=list)
    source_dataset: Optional[str] = None

    # Target information
    target_system: Optional[str] = None
    target_table: Optional[str] = None
    target_column: str = ""
    target_datatype: Optional[DataType] = None
    target_datatype_raw: Optional[str] = None
    target_length: Optional[int] = None
    target_precision: Optional[int] = None
    target_scale: Optional[int] = None
    target_description: Optional[str] = None
    target_dataset: Optional[str] = None

    # Mapping characteristics
    is_primary_key: bool = False
    is_foreign_key: bool = False
    is_nullable: bool = True
    is_derived: bool = False  # Calculated/derived column
    is_lookup: bool = False  # Comes from lookup table

    # Processing information
    processing_order: Optional[int] = None
    transformation_rules: List[TransformationRule] = field(default_factory=list)
    transformation_summary: Optional[str] = None

    # Business rules
    business_rule: Optional[str] = None
    validation_rule: Optional[str] = None

    # Data quality
    data_quality_rule: Optional[str] = None
    null_handling: Optional[str] = None  # "reject", "default", "skip", etc.
    default_value: Optional[str] = None

    # Mapping metadata
    mapping_confidence: float = 1.0  # 0.0 to 1.0
    mapping_type: str = "direct"  # direct, derived, lookup, calculated, aggregated
    mapping_notes: Optional[str] = None
    inferred: bool = False  # Was this mapping inferred vs explicitly defined?

    # Lineage
    upstream_columns: List[str] = field(default_factory=list)  # Columns this depends on
    downstream_columns: List[str] = field(default_factory=list)  # Columns that depend on this

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "mapping_id": self.mapping_id,
            "process_id": self.process_id,
            "process_name": self.process_name,
            "component_id": self.component_id,
            "component_name": self.component_name,
            "system": self.system,
            "source_system": self.source_system,
            "source_table": self.source_table,
            "source_column": self.source_column,
            "source_datatype": self.source_datatype.value if self.source_datatype else None,
            "source_datatype_raw": self.source_datatype_raw,
            "source_length": self.source_length,
            "source_precision": self.source_precision,
            "source_scale": self.source_scale,
            "source_description": self.source_description,
            "source_sample_values": self.source_sample_values,
            "source_dataset": self.source_dataset,
            "target_system": self.target_system,
            "target_table": self.target_table,
            "target_column": self.target_column,
            "target_datatype": self.target_datatype.value if self.target_datatype else None,
            "target_datatype_raw": self.target_datatype_raw,
            "target_length": self.target_length,
            "target_precision": self.target_precision,
            "target_scale": self.target_scale,
            "target_description": self.target_description,
            "target_dataset": self.target_dataset,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "is_nullable": self.is_nullable,
            "is_derived": self.is_derived,
            "is_lookup": self.is_lookup,
            "processing_order": self.processing_order,
            "transformation_rules": [
                {
                    "rule_id": rule.rule_id,
                    "rule_type": rule.rule_type,
                    "description": rule.description,
                    "expression": rule.expression,
                    "function_name": rule.function_name,
                    "parameters": rule.parameters,
                }
                for rule in self.transformation_rules
            ],
            "transformation_summary": self.transformation_summary,
            "business_rule": self.business_rule,
            "validation_rule": self.validation_rule,
            "data_quality_rule": self.data_quality_rule,
            "null_handling": self.null_handling,
            "default_value": self.default_value,
            "mapping_confidence": self.mapping_confidence,
            "mapping_type": self.mapping_type,
            "mapping_notes": self.mapping_notes,
            "inferred": self.inferred,
            "upstream_columns": self.upstream_columns,
            "downstream_columns": self.downstream_columns,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def __repr__(self):
        return f"<ColumnMapping {self.source_column} -> {self.target_column}>"
