"""
Gap Analyzer
Identifies gaps between different system implementations
"""

from typing import List, Dict, Any, Tuple, Optional
import hashlib
from loguru import logger

from core.models import Process, Component, Gap, GapType, GapSeverity


class GapAnalyzer:
    """Analyze gaps between Ab Initio, Hadoop, and Databricks implementations"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = self.config.get("threshold", 0.7)
        self.gaps: List[Gap] = []

    def analyze(
        self,
        source_processes: List[Process],
        target_processes: List[Process],
        source_components: List[Component],
        target_components: List[Component],
        process_mappings: Dict[str, str],  # source_id -> target_id
    ) -> List[Gap]:
        """
        Perform gap analysis
        process_mappings: dict of matched process pairs from matcher
        """
        logger.info(
            f"Analyzing gaps between {len(source_processes)} source and {len(target_processes)} target processes"
        )

        self.gaps = []

        # Gap Type 1: Missing Processes
        self._find_missing_processes(source_processes, target_processes, process_mappings)

        # Gap Type 2: Logic Differences in matched processes
        self._find_logic_differences(
            source_processes,
            target_processes,
            source_components,
            target_components,
            process_mappings,
        )

        # Gap Type 3: Data Coverage Gaps
        self._find_data_coverage_gaps(
            source_processes, target_processes, process_mappings
        )

        # Gap Type 4: Missing Tables/Columns
        self._find_missing_tables_columns(
            source_components, target_components, process_mappings
        )

        logger.info(f"Identified {len(self.gaps)} gaps")
        return self.gaps

    def _find_missing_processes(
        self,
        source_processes: List[Process],
        target_processes: List[Process],
        mappings: Dict[str, str],
    ):
        """Find processes that exist in source but not in target"""
        for source_proc in source_processes:
            if source_proc.id not in mappings:
                # Process not found in target
                gap = Gap(
                    gap_id=self._generate_gap_id("missing", source_proc.id),
                    gap_type=GapType.MISSING_PROCESS,
                    severity=GapSeverity.CRITICAL,
                    source_system=source_proc.system.value,
                    target_system="databricks",  # Assuming target
                    source_process_id=source_proc.id,
                    source_process_name=source_proc.name,
                    title=f"Missing Process: {source_proc.name}",
                    description=f"Process '{source_proc.name}' exists in {source_proc.system.value} but was not found in the target system.",
                    impact="Functionality gap - this process may not be replicated in the target system",
                    business_impact=source_proc.business_function,
                    affected_domain=source_proc.business_domain,
                    recommendation=f"Review and implement equivalent process in target system for: {source_proc.name}",
                    confidence_score=1.0,
                )
                self.gaps.append(gap)

    def _find_logic_differences(
        self,
        source_processes: List[Process],
        target_processes: List[Process],
        source_components: List[Component],
        target_components: List[Component],
        mappings: Dict[str, str],
    ):
        """Find logic differences in matched processes"""
        for source_proc_id, target_proc_id in mappings.items():
            source_proc = next(p for p in source_processes if p.id == source_proc_id)
            target_proc = next(p for p in target_processes if p.id == target_proc_id)

            # Compare component counts
            source_comps = [c for c in source_components if c.process_id == source_proc_id]
            target_comps = [c for c in target_components if c.process_id == target_proc_id]

            if len(source_comps) != len(target_comps):
                gap = Gap(
                    gap_id=self._generate_gap_id("logic", f"{source_proc_id}_{target_proc_id}"),
                    gap_type=GapType.LOGIC_DIFFERENCE,
                    severity=GapSeverity.HIGH,
                    source_system=source_proc.system.value,
                    target_system=target_proc.system.value,
                    source_process_id=source_proc.id,
                    source_process_name=source_proc.name,
                    target_process_id=target_proc.id,
                    target_process_name=target_proc.name,
                    title=f"Component Count Mismatch: {source_proc.name}",
                    description=f"Process has {len(source_comps)} components in source but {len(target_comps)} in target",
                    difference_details=f"Source components: {len(source_comps)}, Target components: {len(target_comps)}",
                    recommendation="Review component logic to ensure all transformations are replicated",
                )
                self.gaps.append(gap)

            # Check for aggregation level differences (e.g., patient account vs person ID)
            self._check_aggregation_differences(
                source_proc, target_proc, source_comps, target_comps
            )

    def _find_data_coverage_gaps(
        self,
        source_processes: List[Process],
        target_processes: List[Process],
        mappings: Dict[str, str],
    ):
        """Find data coverage gaps"""
        for source_proc_id, target_proc_id in mappings.items():
            source_proc = next(p for p in source_processes if p.id == source_proc_id)
            target_proc = next(p for p in target_processes if p.id == target_proc_id)

            # Compare tables involved
            source_tables = set(source_proc.tables_involved)
            target_tables = set(target_proc.tables_involved)

            missing_tables = source_tables - target_tables

            if missing_tables:
                gap = Gap(
                    gap_id=self._generate_gap_id("data", f"{source_proc_id}_{target_proc_id}"),
                    gap_type=GapType.DATA_COVERAGE,
                    severity=GapSeverity.HIGH,
                    source_system=source_proc.system.value,
                    target_system=target_proc.system.value,
                    source_process_id=source_proc.id,
                    source_process_name=source_proc.name,
                    target_process_id=target_proc.id,
                    target_process_name=target_proc.name,
                    title=f"Missing Tables in {target_proc.name}",
                    description=f"Tables used in source are not found in target: {', '.join(missing_tables)}",
                    missing_tables=list(missing_tables),
                    business_impact="Potential data coverage gap",
                    recommendation=f"Verify if tables {', '.join(missing_tables)} are mapped to different table names or need to be added",
                )
                self.gaps.append(gap)

    def _find_missing_tables_columns(
        self,
        source_components: List[Component],
        target_components: List[Component],
        process_mappings: Dict[str, str],
    ):
        """Find missing tables and columns at component level"""
        # Group components by process
        for source_proc_id, target_proc_id in process_mappings.items():
            source_comps = [c for c in source_components if c.process_id == source_proc_id]
            target_comps = [c for c in target_components if c.process_id == target_proc_id]

            # Check for missing columns in matched components
            for source_comp in source_comps:
                # Find equivalent target component (simplified matching)
                target_comp = self._find_equivalent_component(source_comp, target_comps)

                if target_comp:
                    self._compare_schemas(source_comp, target_comp)

    def _check_aggregation_differences(
        self,
        source_proc: Process,
        target_proc: Process,
        source_comps: List[Component],
        target_comps: List[Component],
    ):
        """
        Check for aggregation level differences
        Key issue: Ab Initio at patient account level, Databricks at person ID level
        """
        # Look for keywords indicating aggregation level
        source_desc = (source_proc.description or "") + " ".join(
            [c.business_description or "" for c in source_comps]
        )
        target_desc = (target_proc.description or "") + " ".join(
            [c.business_description or "" for c in target_comps]
        )

        source_has_patient_acct = "patient" in source_desc.lower() and "account" in source_desc.lower()
        source_has_person = "person" in source_desc.lower() or "personid" in source_desc.lower()

        target_has_patient_acct = "patient" in target_desc.lower() and "account" in target_desc.lower()
        target_has_person = "person" in target_desc.lower() or "personid" in target_desc.lower()

        # Check for level mismatch
        if source_has_patient_acct and target_has_person and not target_has_patient_acct:
            gap = Gap(
                gap_id=self._generate_gap_id("agg", f"{source_proc.id}_{target_proc.id}"),
                gap_type=GapType.AGGREGATION_LEVEL,
                severity=GapSeverity.HIGH,
                source_system=source_proc.system.value,
                target_system=target_proc.system.value,
                source_process_id=source_proc.id,
                source_process_name=source_proc.name,
                target_process_id=target_proc.id,
                target_process_name=target_proc.name,
                title=f"Aggregation Level Mismatch: {source_proc.name}",
                description="Source operates at patient account level while target operates at person ID level",
                difference_details="Source: patient account + payer ID level; Target: person ID level",
                business_impact="May result in different lead counts or coverage reporting",
                recommendation="Verify aggregation logic and ensure person-level aggregation in target produces equivalent results",
            )
            self.gaps.append(gap)

    def _find_equivalent_component(
        self, source_comp: Component, target_comps: List[Component]
    ) -> Optional[Component]:
        """Find equivalent component in target by matching names or I/O"""
        # Try exact name match
        for target_comp in target_comps:
            if source_comp.name.lower() == target_comp.name.lower():
                return target_comp

        # Try fuzzy name match
        source_name_lower = source_comp.name.lower()
        for target_comp in target_comps:
            if (
                source_name_lower in target_comp.name.lower()
                or target_comp.name.lower() in source_name_lower
            ):
                return target_comp

        # Try I/O matching
        for target_comp in target_comps:
            if self._components_have_similar_io(source_comp, target_comp):
                return target_comp

        return None

    def _components_have_similar_io(self, comp1: Component, comp2: Component) -> bool:
        """Check if two components have similar inputs/outputs"""
        # Check if output of comp1 matches input of comp2 or vice versa
        comp1_io = set(comp1.input_datasets + comp1.output_datasets + comp1.tables_read + comp1.tables_written)
        comp2_io = set(comp2.input_datasets + comp2.output_datasets + comp2.tables_read + comp2.tables_written)

        # Check for any overlap
        overlap = comp1_io & comp2_io
        return len(overlap) > 0

    def _compare_schemas(self, source_comp: Component, target_comp: Component):
        """Compare schemas of matched components"""
        source_schema = source_comp.output_schema
        target_schema = target_comp.output_schema

        if not source_schema or not target_schema:
            return

        source_fields = source_schema.get("fields", [])
        target_fields = target_schema.get("fields", [])

        source_field_names = {f.get("name", "").lower() for f in source_fields}
        target_field_names = {f.get("name", "").lower() for f in target_fields}

        missing_columns = source_field_names - target_field_names

        if missing_columns:
            gap = Gap(
                gap_id=self._generate_gap_id("col", f"{source_comp.id}_{target_comp.id}"),
                gap_type=GapType.MISSING_COLUMNS,
                severity=GapSeverity.MEDIUM,
                source_system=source_comp.system,
                target_system=target_comp.system,
                source_component_id=source_comp.id,
                source_component_name=source_comp.name,
                target_component_id=target_comp.id,
                target_component_name=target_comp.name,
                title=f"Missing Columns: {source_comp.name} -> {target_comp.name}",
                description=f"Columns in source not found in target: {', '.join(missing_columns)}",
                missing_columns=list(missing_columns),
                recommendation="Verify if columns are renamed or if they need to be added to target schema",
            )
            self.gaps.append(gap)

    def _generate_gap_id(self, gap_type: str, base: str) -> str:
        """Generate unique gap ID"""
        return hashlib.md5(f"{gap_type}_{base}".encode()).hexdigest()[:16]

    def get_summary(self) -> Dict[str, Any]:
        """Get gap analysis summary"""
        summary = {
            "total_gaps": len(self.gaps),
            "by_type": {},
            "by_severity": {},
            "critical_gaps": [],
        }

        for gap in self.gaps:
            # Count by type
            gap_type = gap.gap_type.value
            summary["by_type"][gap_type] = summary["by_type"].get(gap_type, 0) + 1

            # Count by severity
            severity = gap.severity.value
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1

            # Collect critical gaps
            if gap.severity == GapSeverity.CRITICAL:
                summary["critical_gaps"].append(
                    {"title": gap.title, "description": gap.description}
                )

        return summary
