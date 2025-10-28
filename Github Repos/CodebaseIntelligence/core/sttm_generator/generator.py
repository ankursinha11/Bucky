"""
STTM Generator
Generates Source-to-Target Mappings at column level from parsed components
"""

from typing import List, Dict, Any, Optional
import hashlib
from loguru import logger

from core.models import (
    Component,
    Process,
    ColumnMapping,
    TransformationRule,
    DataType,
    STTMEntry,
    STTMReport,
)


class STTMGenerator:
    """Generate STTM (Source-to-Target Mapping) from parsed components"""

    def __init__(self):
        self.column_mappings: List[ColumnMapping] = []

    def generate_from_process(self, process: Process, components: List[Component]) -> STTMReport:
        """Generate STTM report for a specific process"""
        logger.info(f"Generating STTM for process: {process.name}")

        report = STTMReport(
            report_id=f"sttm_{process.id}",
            report_name=f"STTM - {process.name}",
            report_type="process",
            source_system=process.system.value,
            process_names=[process.name],
        )

        # Filter components for this process
        process_components = [c for c in components if c.process_id == process.id]

        # Generate mappings for each component
        for component in process_components:
            mappings = self._generate_mappings_for_component(component, process)
            for mapping in mappings:
                entry = STTMEntry.from_column_mapping(
                    mapping, sttm_id=f"sttm_{len(report.entries) + 1:05d}"
                )
                report.add_entry(entry)

        logger.info(f"Generated {report.total_mappings} STTM entries for {process.name}")
        return report

    def generate_cross_system(
        self,
        source_processes: List[Process],
        target_processes: List[Process],
        all_components: List[Component],
        mappings: Dict[str, str],  # source_process_id -> target_process_id
    ) -> STTMReport:
        """Generate cross-system STTM report"""
        logger.info("Generating cross-system STTM report")

        report = STTMReport(
            report_id="sttm_cross_system",
            report_name="Cross-System STTM Report",
            report_type="cross_system",
            source_system=source_processes[0].system.value if source_processes else None,
            target_system=target_processes[0].system.value if target_processes else None,
        )

        # Generate mappings for each matched process pair
        for source_proc_id, target_proc_id in mappings.items():
            source_proc = next((p for p in source_processes if p.id == source_proc_id), None)
            target_proc = next((p for p in target_processes if p.id == target_proc_id), None)

            if source_proc and target_proc:
                cross_mappings = self._generate_cross_process_mappings(
                    source_proc, target_proc, all_components
                )

                for mapping in cross_mappings:
                    entry = STTMEntry.from_column_mapping(
                        mapping, sttm_id=f"sttm_{len(report.entries) + 1:05d}"
                    )
                    report.add_entry(entry)

        return report

    def _generate_mappings_for_component(
        self, component: Component, process: Process
    ) -> List[ColumnMapping]:
        """Generate column mappings for a single component"""
        mappings = []

        # Extract source and target schemas
        input_schema = component.input_schema
        output_schema = component.output_schema

        if not input_schema and not output_schema:
            # No schema info, create basic mapping
            mapping = self._create_basic_mapping(component, process)
            if mapping:
                mappings.append(mapping)
            return mappings

        # Generate field-level mappings
        if input_schema and output_schema:
            mappings.extend(
                self._map_schemas(component, process, input_schema, output_schema)
            )
        elif output_schema:
            # Only output schema (e.g., CREATE TABLE)
            mappings.extend(self._map_output_only(component, process, output_schema))

        return mappings

    def _create_basic_mapping(
        self, component: Component, process: Process
    ) -> Optional[ColumnMapping]:
        """Create basic mapping when no schema info available"""
        if not component.input_datasets and not component.output_datasets:
            return None

        mapping_id = self._generate_mapping_id(component.id, "basic")

        source_dataset = component.input_datasets[0] if component.input_datasets else ""
        target_dataset = component.output_datasets[0] if component.output_datasets else ""

        return ColumnMapping(
            mapping_id=mapping_id,
            process_id=process.id,
            process_name=process.name,
            component_id=component.id,
            component_name=component.name,
            system=component.system,
            source_dataset=source_dataset,
            source_column="*",
            target_dataset=target_dataset,
            target_column="*",
            transformation_summary=component.transformation_logic or "Component transformation",
            mapping_type="component_level",
            inferred=True,
        )

    def _map_schemas(
        self,
        component: Component,
        process: Process,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
    ) -> List[ColumnMapping]:
        """Map input schema fields to output schema fields"""
        mappings = []

        input_fields = input_schema.get("fields", [])
        output_fields = output_schema.get("fields", [])

        # Try to match fields by name
        matched_fields = self._match_fields(input_fields, output_fields)

        for input_field, output_field in matched_fields:
            mapping = self._create_field_mapping(
                component, process, input_field, output_field
            )
            mappings.append(mapping)

        return mappings

    def _map_output_only(
        self, component: Component, process: Process, output_schema: Dict[str, Any]
    ) -> List[ColumnMapping]:
        """Generate mappings for output-only schemas (e.g., table creation)"""
        mappings = []

        output_fields = output_schema.get("fields", [])
        for idx, field in enumerate(output_fields):
            mapping_id = self._generate_mapping_id(component.id, f"field_{idx}")

            target_table = output_schema.get("table") or (
                component.tables_written[0] if component.tables_written else None
            )

            mapping = ColumnMapping(
                mapping_id=mapping_id,
                process_id=process.id,
                process_name=process.name,
                component_id=component.id,
                component_name=component.name,
                system=component.system,
                target_table=target_table,
                target_column=field.get("name", ""),
                target_datatype=DataType.from_hive(field.get("type", "unknown")),
                target_datatype_raw=field.get("type", ""),
                processing_order=idx + 1,
                mapping_type="output_definition",
                inferred=True,
            )
            mappings.append(mapping)

        return mappings

    def _match_fields(
        self, input_fields: List[Dict], output_fields: List[Dict]
    ) -> List[tuple]:
        """Match input fields to output fields by name"""
        matches = []

        for out_field in output_fields:
            out_name = out_field.get("name", "").lower()

            # Find matching input field
            matched_input = None
            for in_field in input_fields:
                in_name = in_field.get("name", "").lower()
                if in_name == out_name:
                    matched_input = in_field
                    break

            if matched_input:
                matches.append((matched_input, out_field))
            else:
                # No match found - field is derived or new
                matches.append((None, out_field))

        return matches

    def _create_field_mapping(
        self,
        component: Component,
        process: Process,
        input_field: Optional[Dict],
        output_field: Dict,
    ) -> ColumnMapping:
        """Create column mapping for matched fields"""
        mapping_id = self._generate_mapping_id(
            component.id, output_field.get("name", "unknown")
        )

        source_column = input_field.get("name", "") if input_field else ""
        source_type = input_field.get("type", "") if input_field else ""

        target_column = output_field.get("name", "")
        target_type = output_field.get("type", "")

        # Determine if derived
        is_derived = input_field is None

        # Extract transformation if any
        transformation = self._extract_transformation(
            component, source_column, target_column
        )

        return ColumnMapping(
            mapping_id=mapping_id,
            process_id=process.id,
            process_name=process.name,
            component_id=component.id,
            component_name=component.name,
            system=component.system,
            source_column=source_column,
            source_datatype_raw=source_type,
            target_column=target_column,
            target_datatype_raw=target_type,
            is_derived=is_derived,
            transformation_summary=transformation,
            mapping_type="derived" if is_derived else "direct",
            inferred=True,
        )

    def _extract_transformation(
        self, component: Component, source_col: str, target_col: str
    ) -> str:
        """Extract transformation logic for a column"""
        # Look in transformation logic or source code
        if component.transformation_logic and target_col in component.transformation_logic:
            # Try to extract the relevant line
            lines = component.transformation_logic.split("\n")
            for line in lines:
                if target_col in line:
                    return line.strip()[:200]

        if component.transformation_logic:
            return component.transformation_logic[:200]

        return "Direct mapping" if source_col else "Derived column"

    def _generate_cross_process_mappings(
        self,
        source_process: Process,
        target_process: Process,
        all_components: List[Component],
    ) -> List[ColumnMapping]:
        """Generate mappings between source and target processes"""
        mappings = []

        source_comps = [c for c in all_components if c.process_id == source_process.id]
        target_comps = [c for c in all_components if c.process_id == target_process.id]

        # Match components by output/input datasets
        for source_comp in source_comps:
            for target_comp in target_comps:
                if self._components_connected(source_comp, target_comp):
                    # Generate mappings between these components
                    comp_mappings = self._map_component_pair(
                        source_comp, target_comp, source_process, target_process
                    )
                    mappings.extend(comp_mappings)

        return mappings

    def _components_connected(self, source_comp: Component, target_comp: Component) -> bool:
        """Check if two components are connected via datasets or tables"""
        # Check if source output matches target input
        for source_out in source_comp.output_datasets + source_comp.tables_written:
            if source_out in (target_comp.input_datasets + target_comp.tables_read):
                return True
        return False

    def _map_component_pair(
        self,
        source_comp: Component,
        target_comp: Component,
        source_process: Process,
        target_process: Process,
    ) -> List[ColumnMapping]:
        """Create mappings between component pair"""
        mappings = []

        # Get schemas
        source_schema = source_comp.output_schema
        target_schema = target_comp.input_schema or target_comp.output_schema

        if source_schema and target_schema:
            source_fields = source_schema.get("fields", [])
            target_fields = target_schema.get("fields", [])

            matched = self._match_fields(source_fields, target_fields)

            for src_field, tgt_field in matched:
                mapping_id = self._generate_mapping_id(
                    f"{source_comp.id}_{target_comp.id}",
                    tgt_field.get("name", "unknown") if tgt_field else "unknown",
                )

                mapping = ColumnMapping(
                    mapping_id=mapping_id,
                    process_id=f"{source_process.id}_to_{target_process.id}",
                    process_name=f"{source_process.name} -> {target_process.name}",
                    component_id=f"{source_comp.id}_to_{target_comp.id}",
                    component_name=f"{source_comp.name} -> {target_comp.name}",
                    system="cross-system",
                    source_system=source_comp.system,
                    source_column=src_field.get("name", "") if src_field else "",
                    source_datatype_raw=src_field.get("type", "") if src_field else "",
                    target_system=target_comp.system,
                    target_column=tgt_field.get("name", "") if tgt_field else "",
                    target_datatype_raw=tgt_field.get("type", "") if tgt_field else "",
                    is_derived=src_field is None,
                    mapping_type="cross_system",
                    inferred=True,
                )
                mappings.append(mapping)

        return mappings

    def _generate_mapping_id(self, base: str, suffix: str) -> str:
        """Generate unique mapping ID"""
        return hashlib.md5(f"{base}_{suffix}".encode()).hexdigest()[:16]
