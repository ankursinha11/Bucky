"""
Ab Initio MP File Parser - PRODUCTION VERSION
==============================================
FAWN-based parser with clean parameter extraction

Features:
- Clean parameter name/value extraction (FAWN approach)
- GraphFlow extraction with component ID mapping
- Component extraction
- DML parsing
- Transform logic extraction
"""

import re
from typing import List, Dict, Any, Optional
from loguru import logger


class MPFileParser:
    """
    Ab Initio .mp file parser with FAWN-style clean extraction
    """

    def __init__(self, patterns: Dict[str, Any]):
        """
        Initialize parser with patterns

        Args:
            patterns: Pattern dictionary for regex matching
        """
        self.patterns = patterns

    def extract_blocks(self, content: str) -> List[str]:
        """
        Extract all top-level blocks using bracket matching

        Args:
            content: Raw .mp file content

        Returns:
            List of block strings
        """
        blocks = []
        stack = []
        current_block = ''

        for char in content:
            if char == '{':
                if not stack:
                    current_block = ''
                stack.append(char)
                current_block += char
            elif char == '}':
                if stack:
                    stack.pop()
                current_block += char
                if not stack:
                    blocks.append(current_block.strip())
            elif stack:
                current_block += char

        return blocks

    def extract_component_name(self, block: str, component_type: str) -> str:
        """
        Extract component name from block

        Args:
            block: Component block string
            component_type: Type of component

        Returns:
            Component name or empty string
        """
        # Pattern: {ID|TYPE|NAME|...}
        pattern = rf'\{{\d+\|{re.escape(component_type)}\|([^|]+)\|'
        match = re.search(pattern, block)

        if match:
            return match.group(1).strip()

        return ""

    def extract_parameter_blocks(self, block: str) -> List[Dict[str, str]]:
        """
        Extract parameters with CLEAN name/value pairs (FAWN's approach)

        This extracts ONLY field 2 (name) and field 3 (value) from the pipe-separated format:
        {30001002|XXparameter|name|value|...}

        Args:
            block: Component block containing parameters

        Returns:
            List of clean parameter dictionaries with 'parameter_name' and 'parameter_value'
        """
        parameter_list = []

        # Look for XXparameter_set sections
        param_set_pattern = r'XXparameter_set\|@@@@\{\{(.*?)\}\}@'
        matches = re.finditer(param_set_pattern, block, re.DOTALL)

        for match in matches:
            parameter_section = match.group(1)

            # Extract individual parameters
            # Pattern: {ID|XXparameter|!?NAME|VALUE|ORDER|VERSION|TYPE|EXTRA}
            param_pattern = r'\{(\d+)\|XXparameter\|!?([^|]+)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^}]*)\}'

            for param_match in re.finditer(param_pattern, parameter_section):
                param_name = param_match.group(2).strip()  # Field 2: Clean name
                param_value = param_match.group(3).strip()  # Field 3: Clean value

                # Skip internal Ab Initio parameters
                if not param_name.startswith("_ab_"):
                    param_dict = {
                        "parameter_name": param_name,  # CLEAN
                        "parameter_value": param_value,  # CLEAN
                    }
                    parameter_list.append(param_dict)

        return parameter_list

    def extract_graph_parameters(self, blocks: List[str]) -> List[Dict[str, str]]:
        """
        Extract graph-level parameters

        Args:
            blocks: List of all blocks

        Returns:
            List of graph parameter dictionaries
        """
        graph_params = []

        for block in blocks:
            if 'XXgraph' in block:
                params = self.extract_parameter_blocks(block)
                graph_params.extend(params)
                break

        return graph_params

    def extract_components(self, blocks: List[str]) -> List[Dict[str, Any]]:
        """
        Extract all components from blocks

        Args:
            blocks: List of all blocks

        Returns:
            List of component dictionaries
        """
        components = []
        component_types = self.patterns.get("ALLOWED_COMPONENT_TYPES", [])

        for block in blocks:
            for comp_type in component_types:
                if comp_type in block:
                    # Extract component ID
                    id_match = re.search(r'\{(\d+)\|', block)
                    if id_match:
                        comp_id = id_match.group(1)

                        # Extract component name
                        comp_name = self.extract_component_name(block, comp_type)

                        # Extract parameters (CLEAN format!)
                        parameters = self.extract_parameter_blocks(block)

                        component_data = {
                            "component_id": comp_id,
                            "component_type": comp_type,
                            "component_name": comp_name or f"{comp_type}_{comp_id}",
                            "parameters": parameters,  # List of clean param dicts
                            "parameter_count": len(parameters),
                        }

                        components.append(component_data)
                    break

        return components

    def _build_component_id_map(self, blocks: List[str]) -> Dict[str, str]:
        """
        Build mapping from component IDs to component names for GraphFlow

        Args:
            blocks: List of all blocks

        Returns:
            Dictionary mapping ID -> Name
        """
        id_map = {}
        component_types = self.patterns.get("ALLOWED_COMPONENT_TYPES", [])

        for block in blocks:
            for comp_type in component_types:
                if comp_type in block:
                    # Extract ID
                    id_match = re.search(r'\{(\d+)\|', block)
                    if id_match:
                        comp_id = id_match.group(1)

                        # Extract name
                        name = self.extract_component_name(block, comp_type)

                        if comp_id and name:
                            id_map[comp_id] = name
                    break

        return id_map

    def extract_graph_flow(self, blocks: List[str]) -> List[Dict[str, str]]:
        """
        Extract GraphFlow (data lineage) connections with clean component names

        Args:
            blocks: List of all blocks

        Returns:
            List of flow dictionaries with source/target component names
        """
        flows = []

        # Build component ID -> Name mapping
        id_map = self._build_component_id_map(blocks)

        # Look for connection patterns in blocks
        # Pattern: component_id_source -> component_id_target
        connection_pattern = r'\{(\d+)\}.*?-+>.*?\{(\d+)\}'

        for block in blocks:
            matches = re.finditer(connection_pattern, block)

            for match in matches:
                source_id = match.group(1)
                target_id = match.group(2)

                # Map IDs to names
                source_name = id_map.get(source_id, f"component_{source_id}")
                target_name = id_map.get(target_id, f"component_{target_id}")

                flow = {
                    "source_component_id": source_id,
                    "source_component_name": source_name,
                    "target_component_id": target_id,
                    "target_component_name": target_name,
                }

                flows.append(flow)

        logger.debug(f"Extracted {len(flows)} graph flows with component names")

        return flows

    def parse_mp_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Parse complete .mp file

        Args:
            file_path: Path to .mp file
            content: File content

        Returns:
            Dictionary with all parsed data
        """
        logger.info(f"Parsing MP file with FAWN approach: {file_path}")

        # Extract all blocks
        blocks = self.extract_blocks(content)

        # Extract graph parameters (CLEAN format!)
        graph_parameters = self.extract_graph_parameters(blocks)

        # Extract components
        components = self.extract_components(blocks)

        # Extract graph flow
        graph_flow = self.extract_graph_flow(blocks)

        result = {
            "file_path": file_path,
            "total_blocks": len(blocks),
            "graph_parameters": graph_parameters,  # List of clean param dicts
            "components": components,  # Each has 'parameters' list
            "component_count": len(components),
            "graph_flow": graph_flow,  # Source/target with clean names
            "flow_count": len(graph_flow),
        }

        logger.info(f"✓ Parsed: {len(components)} components, "
                   f"{len(graph_parameters)} graph params, {len(graph_flow)} flows")

        return result
