"""
Ab Initio Parser - Main Entry Point
PRODUCTION VERSION with FAWN-based clean parameter extraction
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from core.models import Process, Component, ProcessType, ComponentType, SystemType
from .mp_file_parser import MPFileParser
from .patterns import get_patterns
from .graph_filter_config import is_graph_included, get_module_for_graph, INCLUDED_GRAPHS


class AbInitioParser:
    """
    Ab Initio Parser - Production Version

    Features:
    - FAWN-based clean parameter extraction
    - GraphFlow extraction with component names
    - Component analysis
    - Excel-ready output format
    - Graph filtering (only parse 36 critical graphs)
    """

    def __init__(self, enable_filter: bool = True):
        """
        Initialize Ab Initio parser

        Args:
            enable_filter: If True, only parse graphs in INCLUDED_GRAPHS list (default: True)
        """
        self.patterns = get_patterns()
        self.mp_parser = MPFileParser(self.patterns)
        self.raw_mp_data: List[Dict[str, Any]] = []  # Store raw MP data for Excel export
        self.enable_filter = enable_filter

    def parse_directory(self, abinitio_path: str) -> Dict[str, Any]:
        """
        Parse Ab Initio directory containing .mp files

        Args:
            abinitio_path: Path to Ab Initio files directory

        Returns:
            Dict with 'processes' and 'components' lists
        """
        logger.info(f"Parsing Ab Initio directory: {abinitio_path}")

        processes = []
        components = []
        self.raw_mp_data = []  # Reset

        # Find all .mp files
        mp_files = self._find_mp_files(abinitio_path)
        logger.info(f"Found {len(mp_files)} .mp files")

        # Filter graphs if enabled
        if self.enable_filter:
            filtered_files = []
            skipped_count = 0

            for mp_file in mp_files:
                file_name = Path(mp_file).stem
                if is_graph_included(file_name):
                    filtered_files.append(mp_file)
                else:
                    skipped_count += 1

            logger.info(f"✓ Graph filter ENABLED: {len(filtered_files)} graphs included, {skipped_count} skipped")
            logger.info(f"   Parsing only {len(INCLUDED_GRAPHS)} critical graphs")
            mp_files = filtered_files
        else:
            logger.info("Graph filter DISABLED: parsing all graphs")

        # Parse each .mp file
        for mp_file in mp_files:
            try:
                # Read file
                with open(mp_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Parse with FAWN approach
                mp_data = self.mp_parser.parse_mp_file(mp_file, content)

                # Store raw data for Excel export
                file_name = Path(mp_file).stem
                mp_data_with_file = {
                    **mp_data,
                    "file_name": file_name,
                    "file_path": str(mp_file),
                    "module": get_module_for_graph(file_name) if self.enable_filter else "Unknown",
                }
                self.raw_mp_data.append(mp_data_with_file)

                # Convert to Process and Component objects
                process, process_components = self._convert_to_models(mp_data, mp_file)

                if process:
                    processes.append(process)
                    components.extend(process_components)

            except Exception as e:
                logger.error(f"Error parsing {mp_file}: {e}")
                continue

        logger.info(f"✓ Parsed {len(processes)} processes, {len(components)} components")

        return {
            "processes": processes,
            "components": components,
            "raw_mp_data": self.raw_mp_data,  # Include for Excel export
            "summary": {
                "total_processes": len(processes),
                "total_components": len(components),
                "source_path": abinitio_path,
            }
        }

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a single .mp file

        Args:
            file_path: Path to .mp file

        Returns:
            Dict with 'processes' and 'components'
        """
        logger.info(f"Parsing single .mp file: {file_path}")

        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Parse
            mp_data = self.mp_parser.parse_mp_file(file_path, content)

            # Store raw data
            mp_data_with_file = {
                **mp_data,
                "file_name": Path(file_path).stem,
            }
            self.raw_mp_data = [mp_data_with_file]

            # Convert to models
            process, components = self._convert_to_models(mp_data, file_path)

            return {
                "processes": [process] if process else [],
                "components": components,
                "raw_mp_data": self.raw_mp_data,
            }

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return {"processes": [], "components": [], "raw_mp_data": []}

    def _find_mp_files(self, base_path: str) -> List[str]:
        """
        Find all .mp files recursively

        Args:
            base_path: Base directory to search

        Returns:
            List of .mp file paths
        """
        mp_files = []

        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.mp'):
                    mp_files.append(os.path.join(root, file))

        return sorted(mp_files)

    def _convert_to_models(self, mp_data: Dict[str, Any], file_path: str) -> tuple:
        """
        Convert parsed MP data to Process and Component objects

        Args:
            mp_data: Parsed MP data from MPFileParser
            file_path: Path to .mp file

        Returns:
            Tuple of (Process, List[Component])
        """
        graph_name = Path(file_path).stem

        # Generate unique hash from file path (cross-platform compatible)
        normalized_path = str(Path(file_path).as_posix()) if file_path else ''
        file_hash = hashlib.md5(normalized_path.encode()).hexdigest()[:8]

        # Create Process object with unique ID
        process = Process(
            id=f"abinitio_{graph_name}_{file_hash}",
            name=graph_name,
            system=SystemType.ABINITIO,
            process_type=ProcessType.GRAPH,
            file_path=file_path,
            description=f"Ab Initio Graph: {graph_name}",
            component_count=mp_data.get('component_count', 0),
        )

        # Convert graph parameters to process parameters dict
        graph_params_list = mp_data.get('graph_parameters', [])
        process.graph_parameters = {
            param['parameter_name']: param['parameter_value']
            for param in graph_params_list
        }

        # Create Component objects
        components = []
        component_ids = []

        for comp_data in mp_data.get('components', []):
            comp_id = f"{process.id}_{comp_data['component_id']}"

            # Map component type
            component_type = self._map_component_type(comp_data['component_type'])

            # Convert parameters list to dict
            parameters_dict = {
                param['parameter_name']: param['parameter_value']
                for param in comp_data.get('parameters', [])
            }

            component = Component(
                id=comp_id,
                name=comp_data.get('component_name', ''),
                component_type=component_type,
                system="abinitio",
                file_path=file_path,
                process_id=process.id,
                process_name=process.name,
                parameters=parameters_dict,
            )

            components.append(component)
            component_ids.append(comp_id)

        # Update process
        process.component_ids = component_ids
        process.component_count = len(components)

        return process, components

    def _map_component_type(self, comp_type_str: str) -> ComponentType:
        """
        Map Ab Initio component type to ComponentType enum

        Args:
            comp_type_str: Component type string from .mp file

        Returns:
            ComponentType enum value
        """
        type_mapping = {
            'Input_File': ComponentType.INPUT_FILE,
            'Output_File': ComponentType.OUTPUT_FILE,
            'Lookup_File': ComponentType.LOOKUP_FILE,
            'Transform': ComponentType.TRANSFORM,
            'Reformat': ComponentType.TRANSFORM,
            'Join': ComponentType.JOIN,
            'Filter': ComponentType.FILTER,
            'Aggregate': ComponentType.AGGREGATE,
            'Rollup': ComponentType.AGGREGATE,
            'Sort': ComponentType.SORT,
            'Dedup': ComponentType.FILTER,
            'Gather': ComponentType.AGGREGATE,
            'Partition': ComponentType.FILTER,
            'Scan': ComponentType.INPUT_FILE,
            'Run_Program': ComponentType.TRANSFORM,
        }

        return type_mapping.get(comp_type_str, ComponentType.UNKNOWN)

    def export_to_excel(self, output_path: str, enhanced_flows: List[Dict] = None):
        """
        Export parsed data to Excel with clean FAWN-style parameters

        Args:
            output_path: Path to output Excel file
            enhanced_flows: Enhanced GraphFlow from Autosys dependencies (optional)
        """
        import pandas as pd
        from openpyxl import load_workbook
        from openpyxl.styles import Font, PatternFill, Alignment

        logger.info(f"Exporting to Excel: {output_path}")

        # Excel row limit (leave some buffer)
        EXCEL_MAX_ROWS = 1000000

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Graph Parameters (CLEAN format) - ALWAYS CREATE
            param_data = []
            for mp_data in self.raw_mp_data:
                graph_name = mp_data.get("file_name", "Unknown")
                graph_params = mp_data.get("graph_parameters", [])

                for param in graph_params:
                    param_data.append({
                        "Graph": graph_name,
                        "Parameter": param.get("parameter_name", ""),
                        "Value": param.get("parameter_value", ""),
                    })

            # Always create sheet (even if empty)
            if param_data:
                df_params = pd.DataFrame(param_data)
                if len(df_params) > EXCEL_MAX_ROWS:
                    logger.warning(f"GraphParameters sheet too large ({len(df_params)} rows). Truncating to {EXCEL_MAX_ROWS} rows.")
                    df_params = df_params.head(EXCEL_MAX_ROWS)
            else:
                # Create empty sheet with headers
                df_params = pd.DataFrame(columns=["Graph", "Parameter", "Value"])
                logger.info("GraphParameters sheet is empty (no graph-level parameters found)")

            df_params.to_excel(writer, sheet_name='GraphParameters', index=False)

            # Sheet 2: Components & Fields (CLEAN format)
            component_data = []
            for mp_data in self.raw_mp_data:
                graph_name = mp_data.get("file_name", "Unknown")
                components = mp_data.get("components", [])

                for comp in components:
                    comp_name = comp.get("component_name", "")
                    comp_type = comp.get("component_type", "")
                    parameters = comp.get("parameters", [])

                    for param in parameters:
                        component_data.append({
                            "Graph": graph_name,
                            "Component": comp_name,
                            "Component_Type": comp_type,
                            "Field_Name": param.get("parameter_name", ""),
                            "Field_Value": param.get("parameter_value", ""),
                        })

                        # Stop if we hit the limit
                        if len(component_data) >= EXCEL_MAX_ROWS:
                            break
                    if len(component_data) >= EXCEL_MAX_ROWS:
                        break
                if len(component_data) >= EXCEL_MAX_ROWS:
                    break

            if component_data:
                df_components = pd.DataFrame(component_data)
                total_rows = len(df_components)
                if total_rows > EXCEL_MAX_ROWS:
                    logger.warning(f"Components&Fields sheet too large ({total_rows} rows). Truncating to {EXCEL_MAX_ROWS} rows.")
                    df_components = df_components.head(EXCEL_MAX_ROWS)
                df_components.to_excel(writer, sheet_name='Components&Fields', index=False)
                logger.info(f"   Components&Fields: {len(df_components)} rows exported")

            # Sheet 3: GraphFlow - ALWAYS CREATE
            flow_data = []

            # Priority 1: Use enhanced flows from Autosys (graph-to-graph dependencies)
            if enhanced_flows:
                logger.info(f"Using {len(enhanced_flows)} enhanced graph flows from Autosys dependencies")
                for flow in enhanced_flows:
                    flow_data.append({
                        "Source_Graph": flow.get("source_graph", ""),
                        "Target_Graph": flow.get("target_graph", ""),
                        "Source_Job": flow.get("source_job", ""),
                        "Target_Job": flow.get("target_job", ""),
                        "Dependency_Type": flow.get("condition_type", "success"),
                        "Source": "Autosys"
                    })

            # Priority 2: Add internal component flows from .mp files
            for mp_data in self.raw_mp_data:
                graph_name = mp_data.get("file_name", "Unknown")
                flows = mp_data.get("graph_flow", [])

                for flow in flows:
                    flow_data.append({
                        "Source_Graph": graph_name,
                        "Target_Graph": graph_name,
                        "Source_Job": flow.get("source_component_name", ""),
                        "Target_Job": flow.get("target_component_name", ""),
                        "Dependency_Type": "internal",
                        "Source": "MP_File"
                    })

            # Always create sheet (even if empty)
            if flow_data:
                df_flow = pd.DataFrame(flow_data)
                if len(df_flow) > EXCEL_MAX_ROWS:
                    logger.warning(f"GraphFlow sheet too large ({len(df_flow)} rows). Truncating to {EXCEL_MAX_ROWS} rows.")
                    df_flow = df_flow.head(EXCEL_MAX_ROWS)
            else:
                # Create empty sheet with headers
                df_flow = pd.DataFrame(columns=["Source_Graph", "Target_Graph", "Source_Job", "Target_Job", "Dependency_Type", "Source"])
                logger.info("GraphFlow sheet is empty (no flows found from Autosys or MP files)")

            df_flow.to_excel(writer, sheet_name='GraphFlow', index=False)

            # Sheet 4: Summary
            summary_data = []
            for mp_data in self.raw_mp_data:
                summary_data.append({
                    "Module": mp_data.get("module", "Unknown"),
                    "Graph_Name": mp_data.get("file_name", ""),
                    "Total_Components": mp_data.get("component_count", 0),
                    "Total_Parameters": len(mp_data.get("graph_parameters", [])),
                    "Total_Flows": mp_data.get("flow_count", 0),
                })

            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)

        logger.info(f"✓ Excel file created: {output_path}")
