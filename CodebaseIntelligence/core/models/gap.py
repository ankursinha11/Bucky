from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime


class GapType(Enum):
    """Types of gaps identified"""
    MISSING_PROCESS = "missing_process"
    LOGIC_DIFFERENCE = "logic_difference"
    DATA_COVERAGE = "data_coverage"
    AGGREGATION_LEVEL = "aggregation_level"
    BUSINESS_RULE = "business_rule"
    MISSING_COLUMNS = "missing_columns"
    TRANSFORMATION_DIFFERENCE = "transformation_difference"
    MISSING_TABLE = "missing_table"
    SCHEMA_MISMATCH = "schema_mismatch"
    FILTER_DIFFERENCE = "filter_difference"
    JOIN_DIFFERENCE = "join_difference"
    SEQUENCE_DIFFERENCE = "sequence_difference"


class GapSeverity(Enum):
    """Severity of gaps"""
    CRITICAL = "critical"  # Missing core functionality
    HIGH = "high"  # Significant logic difference
    MEDIUM = "medium"  # Minor differences that may affect results
    LOW = "low"  # Cosmetic or non-functional differences
    INFO = "info"  # Informational, no action needed


@dataclass
class Gap:
    """
    Represents a gap identified between systems
    """
    gap_id: str
    gap_type: GapType
    severity: GapSeverity

    # Systems involved
    source_system: str  # e.g., "abinitio" or "hadoop"
    target_system: str  # e.g., "databricks"

    # Processes involved
    source_process_id: Optional[str] = None
    source_process_name: Optional[str] = None
    target_process_id: Optional[str] = None
    target_process_name: Optional[str] = None

    # Components involved
    source_component_id: Optional[str] = None
    source_component_name: Optional[str] = None
    target_component_id: Optional[str] = None
    target_component_name: Optional[str] = None

    # Gap details
    title: str = ""
    description: str = ""
    impact: Optional[str] = None

    # What's missing or different
    missing_functionality: Optional[str] = None
    source_logic: Optional[str] = None
    target_logic: Optional[str] = None
    difference_details: Optional[str] = None

    # Data-related gaps
    missing_tables: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    schema_differences: Dict[str, Any] = field(default_factory=dict)

    # Business context
    business_impact: Optional[str] = None
    affected_domain: Optional[str] = None  # e.g., "Lead Discovery", "Patient Matching"

    # Recommendations
    recommendation: Optional[str] = None
    remediation_steps: List[str] = field(default_factory=list)
    estimated_effort: Optional[str] = None  # "low", "medium", "high"

    # Evidence
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    code_references: List[str] = field(default_factory=list)

    # Confidence
    confidence_score: float = 1.0  # 0.0 to 1.0

    # Status
    status: str = "identified"  # identified, confirmed, false_positive, resolved, wont_fix
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Timestamps
    identified_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "gap_id": self.gap_id,
            "gap_type": self.gap_type.value,
            "severity": self.severity.value,
            "source_system": self.source_system,
            "target_system": self.target_system,
            "source_process_id": self.source_process_id,
            "source_process_name": self.source_process_name,
            "target_process_id": self.target_process_id,
            "target_process_name": self.target_process_name,
            "source_component_id": self.source_component_id,
            "source_component_name": self.source_component_name,
            "target_component_id": self.target_component_id,
            "target_component_name": self.target_component_name,
            "title": self.title,
            "description": self.description,
            "impact": self.impact,
            "missing_functionality": self.missing_functionality,
            "source_logic": self.source_logic,
            "target_logic": self.target_logic,
            "difference_details": self.difference_details,
            "missing_tables": self.missing_tables,
            "missing_columns": self.missing_columns,
            "schema_differences": self.schema_differences,
            "business_impact": self.business_impact,
            "affected_domain": self.affected_domain,
            "recommendation": self.recommendation,
            "remediation_steps": self.remediation_steps,
            "estimated_effort": self.estimated_effort,
            "evidence": self.evidence,
            "code_references": self.code_references,
            "confidence_score": self.confidence_score,
            "status": self.status,
            "assigned_to": self.assigned_to,
            "resolution_notes": self.resolution_notes,
            "metadata": self.metadata,
            "tags": self.tags,
            "identified_at": self.identified_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }

    def get_summary(self) -> str:
        """Get human-readable summary"""
        summary = f"GAP: {self.title}\n"
        summary += f"Type: {self.gap_type.value} | Severity: {self.severity.value}\n"
        summary += f"From {self.source_system} to {self.target_system}\n"

        if self.source_process_name:
            summary += f"Source: {self.source_process_name}\n"

        if self.target_process_name:
            summary += f"Target: {self.target_process_name}\n"
        else:
            summary += "Target: NOT FOUND\n"

        summary += f"\n{self.description}\n"

        if self.business_impact:
            summary += f"\nBusiness Impact: {self.business_impact}\n"

        if self.recommendation:
            summary += f"\nRecommendation: {self.recommendation}\n"

        return summary

    def __repr__(self):
        return f"<Gap {self.gap_type.value} [{self.severity.value}]: {self.title}>"
