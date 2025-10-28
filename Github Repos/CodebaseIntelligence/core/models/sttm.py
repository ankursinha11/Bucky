from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from .column_mapping import ColumnMapping


@dataclass
class STTMEntry:
    """
    Single entry in Source-to-Target Mapping report
    Enriched version of ColumnMapping for reporting
    """
    # IDs
    sttm_id: str
    mapping_id: str

    # Process context
    process_name: str
    system: str
    business_domain: Optional[str] = None

    # Source
    source_table: str = ""
    source_column: str = ""
    source_datatype: str = ""
    source_description: str = ""
    source_is_pk: str = "N"
    source_is_nullable: str = "Y"

    # Target
    target_table: str = ""
    target_column: str = ""
    target_datatype: str = ""
    target_description: str = ""
    target_is_pk: str = "N"
    target_is_nullable: str = "Y"

    # Mapping details
    processing_order: int = 0
    transformation_rule: str = ""
    business_rule: str = ""
    data_quality_rule: str = ""
    mapping_type: str = "direct"
    mapping_confidence: float = 1.0

    # Additional context
    notes: str = ""
    tags: str = ""

    @classmethod
    def from_column_mapping(cls, mapping: ColumnMapping, sttm_id: str) -> "STTMEntry":
        """Create STTM entry from ColumnMapping"""
        # Build transformation rule summary
        trans_rules = []
        for rule in mapping.transformation_rules:
            if rule.expression:
                trans_rules.append(f"{rule.rule_type}: {rule.expression}")
            else:
                trans_rules.append(rule.description)

        transformation_summary = "; ".join(trans_rules) if trans_rules else (
            mapping.transformation_summary or "Direct mapping"
        )

        return cls(
            sttm_id=sttm_id,
            mapping_id=mapping.mapping_id,
            process_name=mapping.process_name,
            system=mapping.system,
            business_domain=mapping.metadata.get("business_domain"),
            source_table=mapping.source_table or "",
            source_column=mapping.source_column,
            source_datatype=mapping.source_datatype_raw or (
                mapping.source_datatype.value if mapping.source_datatype else ""
            ),
            source_description=mapping.source_description or "",
            source_is_pk="Y" if mapping.is_primary_key else "N",
            source_is_nullable="Y" if mapping.is_nullable else "N",
            target_table=mapping.target_table or "",
            target_column=mapping.target_column,
            target_datatype=mapping.target_datatype_raw or (
                mapping.target_datatype.value if mapping.target_datatype else ""
            ),
            target_description=mapping.target_description or "",
            target_is_pk="Y" if mapping.is_primary_key else "N",
            target_is_nullable="Y" if mapping.is_nullable else "N",
            processing_order=mapping.processing_order or 0,
            transformation_rule=transformation_summary,
            business_rule=mapping.business_rule or "",
            data_quality_rule=mapping.data_quality_rule or "",
            mapping_type=mapping.mapping_type,
            mapping_confidence=mapping.mapping_confidence,
            notes=mapping.mapping_notes or "",
            tags=", ".join(mapping.tags),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export"""
        return {
            "STTM_ID": self.sttm_id,
            "Process_Name": self.process_name,
            "System": self.system,
            "Business_Domain": self.business_domain or "",
            "Source_Table": self.source_table,
            "Source_Column": self.source_column,
            "Source_Datatype": self.source_datatype,
            "Source_Description": self.source_description,
            "Source_Is_PK": self.source_is_pk,
            "Source_Is_Nullable": self.source_is_nullable,
            "Target_Table": self.target_table,
            "Target_Column": self.target_column,
            "Target_Datatype": self.target_datatype,
            "Target_Description": self.target_description,
            "Target_Is_PK": self.target_is_pk,
            "Target_Is_Nullable": self.target_is_nullable,
            "Processing_Order": self.processing_order,
            "Transformation_Rule": self.transformation_rule,
            "Business_Rule": self.business_rule,
            "Data_Quality_Rule": self.data_quality_rule,
            "Mapping_Type": self.mapping_type,
            "Mapping_Confidence": self.mapping_confidence,
            "Notes": self.notes,
            "Tags": self.tags,
        }


@dataclass
class STTMReport:
    """
    Complete STTM Report for a process or cross-system comparison
    """
    report_id: str
    report_name: str
    report_type: str  # "process", "cross_system", "gap_analysis"

    # Context
    source_system: Optional[str] = None
    target_system: Optional[str] = None
    process_names: List[str] = field(default_factory=list)

    # STTM entries
    entries: List[STTMEntry] = field(default_factory=list)
    total_mappings: int = 0

    # Statistics
    direct_mappings: int = 0
    derived_mappings: int = 0
    lookup_mappings: int = 0
    calculated_mappings: int = 0
    missing_mappings: int = 0

    # Tables involved
    source_tables: List[str] = field(default_factory=list)
    target_tables: List[str] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    generated_by: str = "Codebase Intelligence Platform"
    notes: Optional[str] = None

    def add_entry(self, entry: STTMEntry):
        """Add STTM entry and update statistics"""
        self.entries.append(entry)
        self.total_mappings += 1

        # Update statistics
        if entry.mapping_type == "direct":
            self.direct_mappings += 1
        elif entry.mapping_type == "derived":
            self.derived_mappings += 1
        elif entry.mapping_type == "lookup":
            self.lookup_mappings += 1
        elif entry.mapping_type == "calculated":
            self.calculated_mappings += 1

        # Track tables
        if entry.source_table and entry.source_table not in self.source_tables:
            self.source_tables.append(entry.source_table)
        if entry.target_table and entry.target_table not in self.target_tables:
            self.target_tables.append(entry.target_table)

    def to_dataframe(self):
        """Convert to pandas DataFrame for Excel export"""
        import pandas as pd
        return pd.DataFrame([entry.to_dict() for entry in self.entries])

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "report_id": self.report_id,
            "report_name": self.report_name,
            "report_type": self.report_type,
            "source_system": self.source_system,
            "target_system": self.target_system,
            "process_names": self.process_names,
            "total_mappings": self.total_mappings,
            "direct_mappings": self.direct_mappings,
            "derived_mappings": self.derived_mappings,
            "lookup_mappings": self.lookup_mappings,
            "calculated_mappings": self.calculated_mappings,
            "missing_mappings": self.missing_mappings,
            "source_tables_count": len(self.source_tables),
            "target_tables_count": len(self.target_tables),
            "generated_at": self.generated_at.isoformat(),
        }

    def __repr__(self):
        return f"<STTMReport {self.report_name}: {self.total_mappings} mappings>"
