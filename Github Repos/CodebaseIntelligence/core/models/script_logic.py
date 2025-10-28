"""
Script Logic Model (Tier 3)
Deepest level - actual script content, transformations, and data lineage
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class TransformationType(Enum):
    """Type of data transformation"""
    FILTER = "filter"
    JOIN = "join"
    LEFT_JOIN = "left_join"
    INNER_JOIN = "inner_join"
    OUTER_JOIN = "outer_join"
    GROUP_BY = "group_by"
    AGGREGATE = "aggregate"
    SORT = "sort"
    DISTINCT = "distinct"
    UNION = "union"
    TRANSFORM = "transform"
    MAP = "map"
    REDUCE = "reduce"
    WINDOW_FUNCTION = "window_function"
    PIVOT = "pivot"
    UNPIVOT = "unpivot"
    LOOKUP = "lookup"
    UNKNOWN = "unknown"


@dataclass
class Transformation:
    """
    Single data transformation operation

    Represents one logical operation (FILTER, JOIN, etc.) in the script
    """
    transformation_id: str
    transformation_type: TransformationType

    # Code details
    code_snippet: str  # Actual code line(s)
    line_number: Optional[int] = None
    line_range: Optional[tuple] = None  # (start, end)

    # Transformation details
    condition: Optional[str] = None  # For FILTER, JOIN conditions
    columns: List[str] = field(default_factory=list)  # Columns involved
    functions_used: List[str] = field(default_factory=list)  # UDFs, built-in functions

    # For JOINs
    join_type: Optional[str] = None
    left_table: Optional[str] = None
    right_table: Optional[str] = None
    join_keys: List[str] = field(default_factory=list)

    # For GROUP BY
    group_by_columns: List[str] = field(default_factory=list)
    aggregations: List[Dict[str, str]] = field(default_factory=list)  # [{func: col}, ...]

    # Business context
    business_meaning: Optional[str] = None  # AI-generated explanation
    data_quality_impact: Optional[str] = None  # E.g., "Removes 30% of records"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transformation_id": self.transformation_id,
            "transformation_type": self.transformation_type.value,
            "code_snippet": self.code_snippet,
            "line_number": self.line_number,
            "condition": self.condition,
            "columns": self.columns,
            "join_type": self.join_type,
            "group_by_columns": self.group_by_columns,
            "aggregations": self.aggregations,
            "business_meaning": self.business_meaning,
        }


@dataclass
class ColumnLineage:
    """
    Column-level data lineage

    Tracks how source columns flow to target columns through transformations
    """
    source_table: str
    source_column: str
    target_table: str
    target_column: str

    # Transformation path
    transformations_applied: List[str] = field(default_factory=list)  # Transformation IDs
    transformation_logic: Optional[str] = None  # Human-readable transformation

    # Examples
    is_pass_through: bool = False  # Column copied as-is
    is_calculated: bool = False    # Derived from calculation
    is_aggregated: bool = False    # Aggregated (SUM, COUNT, etc.)
    is_filtered: bool = False      # Filtered out in some conditions

    calculation_logic: Optional[str] = None  # For calculated fields

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_table": self.source_table,
            "source_column": self.source_column,
            "target_table": self.target_table,
            "target_column": self.target_column,
            "transformations_applied": self.transformations_applied,
            "transformation_logic": self.transformation_logic,
            "is_pass_through": self.is_pass_through,
            "is_calculated": self.is_calculated,
            "calculation_logic": self.calculation_logic,
        }


@dataclass
class ScriptLogic:
    """
    Complete script logic and data flow

    Tier 3: Deepest level - detailed script content and transformations
    Contains actual code, transformations, and column-level lineage
    """
    # Required fields (no defaults)
    script_id: str
    script_name: str
    script_type: str  # pig, spark, hive, python, sql, etc.
    script_path: str
    raw_content: str  # Full script text
    content_hash: str  # MD5 hash for change detection

    # Links to upper tiers (optional)
    action_id: Optional[str] = None  # Links to Tier 2 ActionNode
    workflow_id: Optional[str] = None  # Links to Tier 2 WorkflowFlow
    repository_id: Optional[str] = None  # Links to Tier 1 Repository

    # Script metadata
    lines_of_code: int = 0
    language_version: Optional[str] = None  # e.g., "Pig 0.16", "Spark 3.2"

    # Input/Output
    input_tables: List[str] = field(default_factory=list)
    output_tables: List[str] = field(default_factory=list)
    input_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)

    # Variables and parameters
    variables: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    constants: Dict[str, Any] = field(default_factory=dict)

    # Transformations
    transformations: List[Transformation] = field(default_factory=list)
    transformation_count_by_type: Dict[str, int] = field(default_factory=dict)

    # Data lineage
    column_lineages: List[ColumnLineage] = field(default_factory=list)

    # Dependencies
    imports: List[str] = field(default_factory=list)  # Imported modules/libraries
    udfs_used: List[str] = field(default_factory=list)  # User-defined functions
    external_scripts: List[str] = field(default_factory=list)  # Other scripts called

    # Business logic (AI-generated)
    business_purpose: Optional[str] = None  # What this script does
    business_logic_summary: Optional[str] = None  # Detailed explanation
    key_business_rules: List[str] = field(default_factory=list)  # Important rules

    # Data quality
    data_filters: List[str] = field(default_factory=list)  # Filter conditions
    data_quality_checks: List[str] = field(default_factory=list)  # Validation logic
    null_handling: List[str] = field(default_factory=list)  # How nulls are handled

    # Performance
    estimated_row_reduction: Optional[float] = None  # % of rows filtered out
    heavy_operations: List[str] = field(default_factory=list)  # Expensive operations

    # AI analysis
    ai_logic_summary: Optional[str] = None  # Comprehensive AI analysis
    ai_optimization_suggestions: List[str] = field(default_factory=list)
    ai_potential_issues: List[str] = field(default_factory=list)
    ai_similar_scripts: List[str] = field(default_factory=list)  # Similar logic found elsewhere

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_modified: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "script_id": self.script_id,
            "script_name": self.script_name,
            "script_type": self.script_type,
            "script_path": self.script_path,
            "action_id": self.action_id,
            "workflow_id": self.workflow_id,
            "repository_id": self.repository_id,
            "raw_content": self.raw_content[:1000],  # Truncate for dict
            "lines_of_code": self.lines_of_code,
            "input_tables": self.input_tables,
            "output_tables": self.output_tables,
            "variables": self.variables,
            "transformations": [t.to_dict() for t in self.transformations],
            "column_lineages": [cl.to_dict() for cl in self.column_lineages],
            "business_purpose": self.business_purpose,
            "business_logic_summary": self.business_logic_summary,
            "ai_logic_summary": self.ai_logic_summary,
        }

    def get_transformation_summary(self) -> str:
        """Get summary of transformations"""
        if not self.transformations:
            return "No transformations"

        summary = []
        for trans_type, count in self.transformation_count_by_type.items():
            summary.append(f"- {trans_type}: {count}")

        return "\n".join(summary)

    def get_data_flow_summary(self) -> str:
        """Get summary of data flow"""
        parts = []

        if self.input_tables:
            parts.append(f"Inputs: {', '.join(self.input_tables[:5])}")

        if self.transformations:
            parts.append(f"→ {len(self.transformations)} transformations")

        if self.output_tables:
            parts.append(f"→ Outputs: {', '.join(self.output_tables[:5])}")

        return " ".join(parts)

    def get_searchable_content(self) -> str:
        """Generate rich searchable content"""
        parts = [
            f"Script: {self.script_name}",
            f"Type: {self.script_type}",
            f"Path: {self.script_path}",
            f"",
        ]

        if self.business_purpose:
            parts.append(f"Business Purpose:")
            parts.append(self.business_purpose)
            parts.append("")

        if self.input_tables:
            parts.append(f"Input Tables ({len(self.input_tables)}):")
            for tbl in self.input_tables:
                parts.append(f"  - {tbl}")
            parts.append("")

        if self.output_tables:
            parts.append(f"Output Tables ({len(self.output_tables)}):")
            for tbl in self.output_tables:
                parts.append(f"  - {tbl}")
            parts.append("")

        if self.transformations:
            parts.append(f"Transformations ({len(self.transformations)}):")
            for trans in self.transformations[:20]:  # First 20
                parts.append(f"  {trans.transformation_type.value}:")
                if trans.condition:
                    parts.append(f"    Condition: {trans.condition}")
                if trans.columns:
                    parts.append(f"    Columns: {', '.join(trans.columns[:5])}")
                if trans.business_meaning:
                    parts.append(f"    Meaning: {trans.business_meaning}")
                parts.append(f"    Code: {trans.code_snippet[:100]}")
            parts.append("")

        if self.column_lineages:
            parts.append(f"Column Lineage ({len(self.column_lineages)} mappings):")
            for lineage in self.column_lineages[:20]:
                if lineage.is_pass_through:
                    parts.append(f"  {lineage.source_table}.{lineage.source_column} → {lineage.target_table}.{lineage.target_column} (pass-through)")
                elif lineage.is_calculated:
                    parts.append(f"  {lineage.source_table}.{lineage.source_column} → {lineage.target_table}.{lineage.target_column}")
                    parts.append(f"    Calculation: {lineage.calculation_logic}")
                else:
                    parts.append(f"  {lineage.source_table}.{lineage.source_column} → {lineage.target_table}.{lineage.target_column}")
                    if lineage.transformation_logic:
                        parts.append(f"    Transform: {lineage.transformation_logic}")
            parts.append("")

        if self.data_filters:
            parts.append("Data Filters:")
            for filt in self.data_filters:
                parts.append(f"  - {filt}")
            parts.append("")

        if self.key_business_rules:
            parts.append("Key Business Rules:")
            for rule in self.key_business_rules:
                parts.append(f"  - {rule}")
            parts.append("")

        if self.business_logic_summary:
            parts.append("Business Logic:")
            parts.append(self.business_logic_summary)
            parts.append("")

        if self.ai_logic_summary:
            parts.append("AI Analysis:")
            parts.append(self.ai_logic_summary)
            parts.append("")

        if self.ai_optimization_suggestions:
            parts.append("Optimization Suggestions:")
            for sugg in self.ai_optimization_suggestions:
                parts.append(f"  - {sugg}")
            parts.append("")

        if self.ai_potential_issues:
            parts.append("Potential Issues:")
            for issue in self.ai_potential_issues:
                parts.append(f"  ⚠ {issue}")
            parts.append("")

        # Include actual code (truncated)
        parts.append("Code Preview:")
        parts.append("```")
        parts.append(self.raw_content[:500])  # First 500 chars
        parts.append("```")

        return "\n".join(parts)

    def __repr__(self):
        return f"<ScriptLogic {self.script_name} ({self.script_type}): {len(self.transformations)} transformations, {len(self.column_lineages)} lineages>"
