"""
Ab Initio Graph Mapping Parser
Parses Excel files containing source-to-target mappings for Ab Initio graphs
Integrates with FAWN output for comprehensive understanding
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class AbInitioIO:
    """Represents an input or output for an Ab Initio graph"""
    component_id: str  # e.g., "110_IFIL"
    component_name: str  # e.g., "tempIPA"
    file_path: str  # e.g., "$AI_MFS_TEMP/${fileNamePrefix}ediGenPatientAccts.txt"
    description: str

    def to_dict(self) -> Dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "file_path": self.file_path,
            "description": self.description
        }


@dataclass
class AbInitioGraphMapping:
    """Represents a complete Ab Initio graph mapping"""
    graph_name: str
    summary: str
    inputs: List[AbInitioIO] = field(default_factory=list)
    outputs: List[AbInitioIO] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "graph_name": self.graph_name,
            "summary": self.summary,
            "inputs": [inp.to_dict() for inp in self.inputs],
            "outputs": [out.to_dict() for out in self.outputs],
            "transformations": self.transformations,
            "input_count": len(self.inputs),
            "output_count": len(self.outputs)
        }


class AbInitioMappingParser:
    """
    Parse Ab Initio graph mapping Excel files

    Supports multiple formats:
    - FAWN output format (detailed component analysis)
    - Custom mapping format (source-to-target)
    - Summary format (graph overview)
    """

    def __init__(self):
        """Initialize parser"""
        self.mappings: Dict[str, AbInitioGraphMapping] = {}

    def parse_excel_file(self, excel_path: str) -> Dict[str, AbInitioGraphMapping]:
        """
        Parse Ab Initio mapping Excel file

        Args:
            excel_path: Path to Excel file

        Returns:
            Dictionary of graph_name -> AbInitioGraphMapping
        """
        logger.info(f"Parsing Ab Initio mapping: {excel_path}")

        try:
            # Read Excel without assuming header structure
            df = pd.read_excel(excel_path, header=None)

            # Detect format and parse accordingly
            if self._is_summary_format(df):
                mappings = self._parse_summary_format(df)
            elif self._is_fawn_format(df):
                mappings = self._parse_fawn_format(df)
            else:
                mappings = self._parse_generic_format(df)

            self.mappings.update(mappings)
            logger.info(f"Parsed {len(mappings)} graph mappings")

            return mappings

        except Exception as e:
            logger.error(f"Failed to parse Excel: {e}")
            return {}

    def _is_summary_format(self, df: pd.DataFrame) -> bool:
        """Check if Excel is in summary format (graph name, summary, inputs, outputs)"""
        # Look for "Graph Name:" or "GRAPH SUMMARY" in first few rows
        first_rows = df.iloc[:5, 0].astype(str).str.lower()
        return any('graph' in row and ('name' in row or 'summary' in row) for row in first_rows)

    def _is_fawn_format(self, df: pd.DataFrame) -> bool:
        """Check if Excel is in FAWN output format"""
        # FAWN format has specific columns like Component ID, Component Type, etc.
        first_row = df.iloc[0].astype(str).str.lower()
        return any('component' in val for val in first_row)

    def _parse_summary_format(self, df: pd.DataFrame) -> Dict[str, AbInitioGraphMapping]:
        """Parse summary format Excel"""
        mappings = {}
        current_graph = None
        current_section = None

        for idx, row in df.iterrows():
            # Get first two columns (key-value pairs)
            key = str(row[0]) if pd.notna(row[0]) else ""
            value = str(row[1]) if pd.notna(row[1]) else ""

            key_lower = key.lower().strip()

            # Detect graph name
            if 'graph' in key_lower and 'name' in key_lower:
                if current_graph:
                    mappings[current_graph.graph_name] = current_graph

                current_graph = AbInitioGraphMapping(
                    graph_name=value.strip(),
                    summary=""
                )

            # Detect summary
            elif current_graph and 'summary' in key_lower:
                current_graph.summary = value.strip()

            # Detect section headers
            elif current_graph:
                if 'input' in key_lower:
                    current_section = 'inputs'
                elif 'output' in key_lower:
                    current_section = 'outputs'

                # Parse IO entry
                elif current_section and value:
                    io_entry = self._parse_io_entry(value)
                    if io_entry:
                        if current_section == 'inputs':
                            current_graph.inputs.append(io_entry)
                        elif current_section == 'outputs':
                            current_graph.outputs.append(io_entry)

        # Add last graph
        if current_graph:
            mappings[current_graph.graph_name] = current_graph

        return mappings

    def _parse_io_entry(self, text: str) -> Optional[AbInitioIO]:
        """
        Parse IO entry text

        Format: "110_IFIL tempIPA - $AI_MFS_TEMP/${fileNamePrefix}ediGenPatientAccts.txt - Description"
        """
        try:
            # Pattern: COMPONENT_ID NAME - PATH - DESCRIPTION
            pattern = r'(\d+_[A-Z]+)\s+(\w+)\s*-\s*([^\-]+)\s*-\s*(.+)'
            match = re.match(pattern, text)

            if match:
                return AbInitioIO(
                    component_id=match.group(1).strip(),
                    component_name=match.group(2).strip(),
                    file_path=match.group(3).strip(),
                    description=match.group(4).strip()
                )

            # Fallback: simpler parsing
            parts = text.split('-')
            if len(parts) >= 2:
                comp_parts = parts[0].strip().split()
                if len(comp_parts) >= 2:
                    return AbInitioIO(
                        component_id=comp_parts[0],
                        component_name=comp_parts[1] if len(comp_parts) > 1 else "",
                        file_path=parts[1].strip() if len(parts) > 1 else "",
                        description=parts[2].strip() if len(parts) > 2 else ""
                    )

        except Exception as e:
            logger.debug(f"Could not parse IO entry: {text[:100]}")

        return None

    def _parse_fawn_format(self, df: pd.DataFrame) -> Dict[str, AbInitioGraphMapping]:
        """Parse FAWN output format"""
        # TODO: Implement FAWN format parsing
        # FAWN outputs have component-level details
        logger.warning("FAWN format parsing not yet implemented")
        return {}

    def _parse_generic_format(self, df: pd.DataFrame) -> Dict[str, AbInitioGraphMapping]:
        """Parse generic/unknown format - best effort"""
        logger.warning("Unknown Excel format - using generic parsing")

        # Look for any table names, file paths, descriptions
        mappings = {}

        # Simple heuristic: treat as table where first row might be headers
        if df.shape[0] > 1:
            # Try treating first row as headers
            df_with_headers = pd.read_excel(df, header=0)

            # Look for columns that might contain graph info
            for idx, row in df_with_headers.iterrows():
                # Extract any useful information
                pass

        return mappings

    def get_graph_mapping(self, graph_name: str) -> Optional[AbInitioGraphMapping]:
        """Get mapping for a specific graph"""
        return self.mappings.get(graph_name)

    def list_graphs(self) -> List[str]:
        """List all parsed graphs"""
        return list(self.mappings.keys())

    def export_to_json(self, output_path: str):
        """Export all mappings to JSON"""
        import json

        output = {
            "total_graphs": len(self.mappings),
            "graphs": {name: mapping.to_dict() for name, mapping in self.mappings.items()}
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Exported {len(self.mappings)} graph mappings to {output_path}")

    def integrate_with_fawn(self, fawn_output_dir: str):
        """
        Integrate with FAWN output for enhanced understanding

        Args:
            fawn_output_dir: Directory containing FAWN JSON outputs
        """
        import json

        fawn_dir = Path(fawn_output_dir)

        if not fawn_dir.exists():
            logger.warning(f"FAWN output directory not found: {fawn_dir}")
            return

        # Read FAWN outputs
        for fawn_file in fawn_dir.glob("*.json"):
            try:
                with open(fawn_file, 'r') as f:
                    fawn_data = json.load(f)

                # Extract graph name
                graph_name = fawn_data.get('graph_name', fawn_file.stem)

                # If we have mapping for this graph, enrich it
                if graph_name in self.mappings:
                    mapping = self.mappings[graph_name]

                    # Add FAWN insights
                    if 'transformations' in fawn_data:
                        mapping.transformations = fawn_data['transformations']

                    logger.debug(f"Integrated FAWN data for {graph_name}")

            except Exception as e:
                logger.debug(f"Could not integrate FAWN file {fawn_file.name}: {e}")


# Testing function
def test_abinitio_mapping_parser():
    """Test the Ab Initio mapping parser"""
    import json

    print("=" * 60)
    print("AB INITIO MAPPING PARSER TEST")
    print("=" * 60)

    excel_file = "/Users/ankurshome/Desktop/Hadoop_Parser/CodebaseIntelligence/graph_1_source_to_target_mapping.xlsx"

    parser = AbInitioMappingParser()
    mappings = parser.parse_excel_file(excel_file)

    print(f"\n📊 Parsed {len(mappings)} graph mappings\n")

    for graph_name, mapping in mappings.items():
        print(f"📈 Graph: {graph_name}")
        print(f"   Summary: {mapping.summary[:100]}...")
        print(f"   Inputs: {len(mapping.inputs)}")
        for inp in mapping.inputs:
            print(f"      - {inp.component_id} {inp.component_name}: {inp.file_path}")
        print(f"   Outputs: {len(mapping.outputs)}")
        for out in mapping.outputs:
            print(f"      - {out.component_id} {out.component_name}: {out.file_path}")
        print()

    # Export to JSON
    output_file = "abinitio_graph_mappings.json"
    parser.export_to_json(output_file)

    print(f"✅ Mappings exported to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    test_abinitio_mapping_parser()
