"""
Enhanced Lineage Tracker
=========================
Advanced lineage tracking with proper script parsing and dependency graph construction

Features:
- Parses actual script logic (SQL, Pig, Shell, Python)
- Classifies scripts by operation type (Source, Transform, Consumer, Definition)
- Follows data locations (HDFS paths, table locations, file paths)
- Builds accurate dependency graphs
- Supports graph visualization

Author: STAG
Date: November 11, 2025
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
import re
import json
from loguru import logger


class ScriptType(Enum):
    """Classification of script by what it actually does"""
    SOURCE = "source"  # Writes raw data (downloads, generates, ingests)
    TRANSFORM = "transform"  # Reads data, transforms it, writes output
    CONSUMER = "consumer"  # Only reads data (reports, exports without write)
    DEFINITION = "definition"  # Metadata only (CREATE TABLE, schema definitions)
    ORCHESTRATOR = "orchestrator"  # Workflow/pipeline definition (Oozie, Airflow)
    UNKNOWN = "unknown"  # Cannot determine


class TransformationType(Enum):
    """Type of transformation operation"""
    FILTER = "filter"
    JOIN = "join"
    AGGREGATE = "aggregate"
    UNION = "union"
    DISTINCT = "distinct"
    SORT = "sort"
    PROJECTION = "projection"
    WINDOW = "window"
    CUSTOM = "custom"


@dataclass
class DataLocation:
    """Represents a data storage location"""
    location_type: str  # "hdfs_path", "table", "file", "hive_table"
    path: str  # Actual path or table name
    format: Optional[str] = None  # "parquet", "orc", "textfile", etc.
    is_external: bool = False  # External table or managed

    def __hash__(self):
        return hash((self.location_type, self.path))

    def __eq__(self, other):
        if not isinstance(other, DataLocation):
            return False
        return self.location_type == other.location_type and self.path == other.path


@dataclass
class ParsedScript:
    """Result of parsing a script"""
    script_path: str
    script_name: str
    script_type: ScriptType
    system: str  # hadoop, databricks, abinitio

    # Data flow
    reads_from: List[DataLocation] = field(default_factory=list)  # Input data locations
    writes_to: List[DataLocation] = field(default_factory=list)  # Output data locations

    # Logic
    transformations: List[TransformationType] = field(default_factory=list)
    business_logic: List[str] = field(default_factory=list)

    # Metadata
    column_mappings: Dict[str, str] = field(default_factory=dict)  # source_col -> target_col
    filters: List[str] = field(default_factory=list)
    joins: List[Dict[str, Any]] = field(default_factory=list)

    # Confidence
    confidence: float = 1.0
    parsing_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'script_path': self.script_path,
            'script_name': self.script_name,
            'script_type': self.script_type.value,
            'system': self.system,
            'reads_from': [{'type': loc.location_type, 'path': loc.path} for loc in self.reads_from],
            'writes_to': [{'type': loc.location_type, 'path': loc.path} for loc in self.writes_to],
            'transformations': [t.value for t in self.transformations],
            'business_logic': self.business_logic,
            'column_mappings': self.column_mappings,
            'filters': self.filters,
            'joins': self.joins,
            'confidence': self.confidence,
            'parsing_errors': self.parsing_errors
        }


@dataclass
class LineageNode:
    """Node in the lineage graph"""
    node_id: str  # Unique identifier
    node_type: str  # "data_location", "script", "table"
    name: str  # Display name
    details: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if not isinstance(other, LineageNode):
            return False
        return self.node_id == other.node_id


@dataclass
class LineageEdge:
    """Edge in the lineage graph"""
    source_id: str
    target_id: str
    edge_type: str  # "reads", "writes", "transforms"
    script: Optional[str] = None
    transformation: Optional[str] = None

    def __hash__(self):
        return hash((self.source_id, self.target_id, self.edge_type))


@dataclass
class LineageGraph:
    """Complete lineage dependency graph"""
    nodes: Dict[str, LineageNode] = field(default_factory=dict)  # node_id -> node
    edges: List[LineageEdge] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)  # Sources (no inputs)
    exit_points: List[str] = field(default_factory=list)  # Consumers (no outputs)

    def add_node(self, node: LineageNode):
        """Add node to graph"""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: LineageEdge):
        """Add edge to graph"""
        if edge not in self.edges:
            self.edges.append(edge)

    def get_upstream(self, node_id: str, max_depth: int = 10) -> List[str]:
        """Get all upstream dependencies"""
        visited = set()
        queue = [(node_id, 0)]
        upstream = []

        while queue:
            current_id, depth = queue.pop(0)
            if depth > max_depth or current_id in visited:
                continue

            visited.add(current_id)
            if current_id != node_id:
                upstream.append(current_id)

            # Find edges pointing to this node
            for edge in self.edges:
                if edge.target_id == current_id and edge.source_id not in visited:
                    queue.append((edge.source_id, depth + 1))

        return upstream

    def get_downstream(self, node_id: str, max_depth: int = 10) -> List[str]:
        """Get all downstream dependencies"""
        visited = set()
        queue = [(node_id, 0)]
        downstream = []

        while queue:
            current_id, depth = queue.pop(0)
            if depth > max_depth or current_id in visited:
                continue

            visited.add(current_id)
            if current_id != node_id:
                downstream.append(current_id)

            # Find edges coming from this node
            for edge in self.edges:
                if edge.source_id == current_id and edge.target_id not in visited:
                    queue.append((edge.target_id, depth + 1))

        return downstream

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'nodes': [
                {
                    'id': node_id,
                    'type': node.node_type,
                    'name': node.name,
                    'details': node.details
                }
                for node_id, node in self.nodes.items()
            ],
            'edges': [
                {
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'type': edge.edge_type,
                    'script': edge.script,
                    'transformation': edge.transformation
                }
                for edge in self.edges
            ],
            'entry_points': self.entry_points,
            'exit_points': self.exit_points
        }


class ScriptParser:
    """Parse different script types to extract data flow"""

    @staticmethod
    def parse_script(script_path: str, system: str) -> ParsedScript:
        """
        Parse a script file to extract data flow and logic

        Args:
            script_path: Path to script file
            system: System type (hadoop, databricks, abinitio)

        Returns:
            ParsedScript with extracted information
        """
        script_name = Path(script_path).name
        file_extension = Path(script_path).suffix.lower()

        logger.debug(f"Parsing {script_name} (type: {file_extension})")

        try:
            with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {script_path}: {e}")
            return ParsedScript(
                script_path=script_path,
                script_name=script_name,
                script_type=ScriptType.UNKNOWN,
                system=system,
                confidence=0.0,
                parsing_errors=[f"Read error: {e}"]
            )

        # Route to appropriate parser based on file type
        if file_extension in ['.sql', '.hql', '.hive']:
            return ScriptParser._parse_sql(script_path, script_name, content, system)
        elif file_extension == '.pig':
            return ScriptParser._parse_pig(script_path, script_name, content, system)
        elif file_extension == '.sh':
            return ScriptParser._parse_shell(script_path, script_name, content, system)
        elif file_extension == '.py':
            return ScriptParser._parse_python(script_path, script_name, content, system)
        elif file_extension == '.xml':
            return ScriptParser._parse_oozie_xml(script_path, script_name, content, system)
        else:
            return ParsedScript(
                script_path=script_path,
                script_name=script_name,
                script_type=ScriptType.UNKNOWN,
                system=system,
                confidence=0.5,
                parsing_errors=[f"Unknown file type: {file_extension}"]
            )

    @staticmethod
    def _parse_sql(script_path: str, script_name: str, content: str, system: str) -> ParsedScript:
        """Parse SQL/Hive script"""
        parsed = ParsedScript(
            script_path=script_path,
            script_name=script_name,
            system=system,
            script_type=ScriptType.UNKNOWN
        )

        content_upper = content.upper()

        # Classify script type
        has_create_table = 'CREATE' in content_upper and 'TABLE' in content_upper
        has_insert = 'INSERT' in content_upper
        has_select = 'SELECT' in content_upper
        has_external = 'EXTERNAL' in content_upper

        if has_create_table and not has_insert:
            # Just table definition
            parsed.script_type = ScriptType.DEFINITION

            # Extract table name
            create_match = re.search(r'CREATE\s+(?:EXTERNAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', content, re.IGNORECASE)
            if create_match:
                table_name = create_match.group(1)

                # Extract LOCATION if external
                location_match = re.search(r"LOCATION\s+'([^']+)'", content, re.IGNORECASE)
                if location_match:
                    location_path = location_match.group(1)
                    parsed.writes_to.append(DataLocation(
                        location_type="hive_table",
                        path=table_name,
                        is_external=has_external
                    ))
                    parsed.writes_to.append(DataLocation(
                        location_type="hdfs_path",
                        path=location_path
                    ))
                else:
                    parsed.writes_to.append(DataLocation(
                        location_type="hive_table",
                        path=table_name,
                        is_external=False
                    ))

        elif has_insert and has_select:
            # Transformation: INSERT INTO ... SELECT FROM ...
            parsed.script_type = ScriptType.TRANSFORM

            # Extract INSERT INTO target
            insert_matches = re.findall(r'INSERT\s+(?:INTO|OVERWRITE)\s+(?:TABLE\s+)?(\w+)', content, re.IGNORECASE)
            for table in insert_matches:
                parsed.writes_to.append(DataLocation(
                    location_type="hive_table",
                    path=table
                ))

            # Extract FROM sources
            from_matches = re.findall(r'FROM\s+(\w+)', content, re.IGNORECASE)
            for table in from_matches:
                parsed.reads_from.append(DataLocation(
                    location_type="hive_table",
                    path=table
                ))

            # Extract transformations
            if 'WHERE' in content_upper or 'FILTER' in content_upper:
                parsed.transformations.append(TransformationType.FILTER)
                # Extract filter conditions
                where_matches = re.findall(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|;|\n)', content, re.IGNORECASE | re.DOTALL)
                parsed.filters.extend(where_matches)

            if 'JOIN' in content_upper:
                parsed.transformations.append(TransformationType.JOIN)
                # Extract join conditions
                join_matches = re.findall(r'(\w+\s+)?JOIN\s+(\w+)(?:\s+\w+)?\s+ON\s+(.+?)(?:WHERE|GROUP|ORDER|;|\n)', content, re.IGNORECASE | re.DOTALL)
                for join_match in join_matches:
                    parsed.joins.append({
                        'type': join_match[0].strip() if join_match[0] else 'INNER',
                        'table': join_match[1],
                        'condition': join_match[2].strip()
                    })

            if 'GROUP BY' in content_upper:
                parsed.transformations.append(TransformationType.AGGREGATE)

            if 'DISTINCT' in content_upper:
                parsed.transformations.append(TransformationType.DISTINCT)

            if 'ORDER BY' in content_upper or 'SORT BY' in content_upper:
                parsed.transformations.append(TransformationType.SORT)

        elif has_select and not has_insert:
            # Read-only query (consumer)
            parsed.script_type = ScriptType.CONSUMER

            # Extract FROM sources
            from_matches = re.findall(r'FROM\s+(\w+)', content, re.IGNORECASE)
            for table in from_matches:
                parsed.reads_from.append(DataLocation(
                    location_type="hive_table",
                    path=table
                ))

        return parsed

    @staticmethod
    def _parse_pig(script_path: str, script_name: str, content: str, system: str) -> ParsedScript:
        """Parse Pig script"""
        parsed = ParsedScript(
            script_path=script_path,
            script_name=script_name,
            system=system,
            script_type=ScriptType.TRANSFORM  # Pig scripts are always transformations
        )

        # Extract LOAD statements (inputs)
        load_matches = re.findall(r"LOAD\s+'([^']+)'", content, re.IGNORECASE)
        for path in load_matches:
            parsed.reads_from.append(DataLocation(
                location_type="hdfs_path",
                path=path
            ))

        # Extract STORE statements (outputs)
        store_matches = re.findall(r"STORE\s+\w+\s+INTO\s+'([^']+)'", content, re.IGNORECASE)
        for path in store_matches:
            parsed.writes_to.append(DataLocation(
                location_type="hdfs_path",
                path=path
            ))

        # Extract transformations
        if re.search(r'\bFILTER\b', content, re.IGNORECASE):
            parsed.transformations.append(TransformationType.FILTER)

        if re.search(r'\bJOIN\b', content, re.IGNORECASE):
            parsed.transformations.append(TransformationType.JOIN)

        if re.search(r'\bGROUP\b', content, re.IGNORECASE):
            parsed.transformations.append(TransformationType.AGGREGATE)

        if re.search(r'\bDISTINCT\b', content, re.IGNORECASE):
            parsed.transformations.append(TransformationType.DISTINCT)

        return parsed

    @staticmethod
    def _parse_shell(script_path: str, script_name: str, content: str, system: str) -> ParsedScript:
        """Parse shell script"""
        parsed = ParsedScript(
            script_path=script_path,
            script_name=script_name,
            system=system,
            script_type=ScriptType.UNKNOWN
        )

        # Check for data writes (echo >> file, scp, hdfs dfs -put)
        has_write = bool(re.search(r'(echo.+?>>|scp.+?:|hdfs\s+dfs\s+-put|hadoop\s+fs\s+-put)', content))

        # Check for data reads (SELECT without INSERT, cat, hdfs dfs -get)
        has_hive_query = bool(re.search(r'hive\s+-e\s+"SELECT', content, re.IGNORECASE))
        has_insert = bool(re.search(r'INSERT\s+INTO', content, re.IGNORECASE))

        if has_write:
            # Writes data - could be source or transform
            if has_hive_query:
                # Runs query and writes somewhere - transform
                parsed.script_type = ScriptType.TRANSFORM
            else:
                # Just writes raw data - source
                parsed.script_type = ScriptType.SOURCE

            # Extract write locations
            write_matches = re.findall(r'echo.+?>>\s*([^\s;]+)', content)
            for path in write_matches:
                parsed.writes_to.append(DataLocation(
                    location_type="file",
                    path=path
                ))

        elif has_hive_query and not has_insert:
            # Read-only query - consumer
            parsed.script_type = ScriptType.CONSUMER

        # Extract Hive table references
        hive_tables = re.findall(r'(?:FROM|JOIN)\s+(\w+)', content, re.IGNORECASE)
        for table in hive_tables:
            parsed.reads_from.append(DataLocation(
                location_type="hive_table",
                path=table
            ))

        return parsed

    @staticmethod
    def _parse_python(script_path: str, script_name: str, content: str, system: str) -> ParsedScript:
        """Parse Python/PySpark script"""
        parsed = ParsedScript(
            script_path=script_path,
            script_name=script_name,
            system=system,
            script_type=ScriptType.TRANSFORM  # Assume transform by default
        )

        # Extract spark.read operations
        read_matches = re.findall(r'spark\.read\.(?:table|parquet|csv|json)\(["\']([^"\']+)["\']\)', content)
        for source in read_matches:
            parsed.reads_from.append(DataLocation(
                location_type="table" if 'spark.read.table' in content else "hdfs_path",
                path=source
            ))

        # Extract write operations
        write_matches = re.findall(r'\.(?:saveAsTable|write\.parquet|write\.csv)\(["\']([^"\']+)["\']\)', content)
        for target in write_matches:
            parsed.writes_to.append(DataLocation(
                location_type="table" if 'saveAsTable' in content else "hdfs_path",
                path=target
            ))

        # Extract transformations
        if re.search(r'\.filter\(|\.where\(', content):
            parsed.transformations.append(TransformationType.FILTER)

        if re.search(r'\.join\(', content):
            parsed.transformations.append(TransformationType.JOIN)

        if re.search(r'\.groupBy\(|\.agg\(', content):
            parsed.transformations.append(TransformationType.AGGREGATE)

        if re.search(r'\.distinct\(|\.dropDuplicates\(', content):
            parsed.transformations.append(TransformationType.DISTINCT)

        return parsed

    @staticmethod
    def _parse_oozie_xml(script_path: str, script_name: str, content: str, system: str) -> ParsedScript:
        """Parse Oozie workflow XML"""
        parsed = ParsedScript(
            script_path=script_path,
            script_name=script_name,
            system=system,
            script_type=ScriptType.ORCHESTRATOR
        )

        # Extract script references
        script_refs = re.findall(r'<script>([^<]+)</script>', content)
        parsed.business_logic = [f"Executes: {ref}" for ref in script_refs]

        return parsed


class EnhancedLineageTracker:
    """
    Enhanced lineage tracking with proper script parsing and graph construction
    """

    def __init__(self, indexer=None, ai_analyzer=None):
        """
        Initialize enhanced lineage tracker

        Args:
            indexer: MultiCollectionIndexer for finding scripts
            ai_analyzer: AIScriptAnalyzer for additional AI-powered analysis
        """
        self.indexer = indexer
        self.ai_analyzer = ai_analyzer
        self.parser = ScriptParser()

    def track_table_lineage(
        self,
        table_name: str,
        system: str,
        max_depth: int = 10
    ) -> LineageGraph:
        """
        Track complete lineage for a table

        Args:
            table_name: Table or entity name to track
            system: System (hadoop, databricks, abinitio)
            max_depth: Maximum depth to traverse

        Returns:
            LineageGraph with complete dependency graph
        """
        logger.info(f"🔍 Tracking lineage for: {table_name} in {system}")

        # Step 1: Find all scripts that mention this table
        scripts = self._find_related_scripts(table_name, system)
        logger.info(f"  📊 Found {len(scripts)} related scripts")

        # Step 2: Parse all scripts to understand data flow
        parsed_scripts = []
        for script_path in scripts:
            parsed = self.parser.parse_script(script_path, system)
            parsed_scripts.append(parsed)
            logger.debug(f"    - {parsed.script_name}: {parsed.script_type.value}")

        # Step 3: Build dependency graph
        graph = self._build_graph(parsed_scripts, table_name)
        logger.info(f"  ✓ Built graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

        return graph

    def _find_related_scripts(self, table_name: str, system: str) -> List[str]:
        """Find all scripts that reference the table"""
        if not self.indexer:
            logger.warning("No indexer available, cannot search for scripts")
            return []

        collection = f"{system}_collection"
        results = self.indexer.search_multi_collection(
            query=table_name,
            collections=[collection],
            top_k=100
        )

        script_paths = []
        if collection in results:
            for result in results[collection]:
                metadata = result.get('metadata', {})
                file_path = metadata.get('absolute_file_path', '')
                if file_path and Path(file_path).exists():
                    script_paths.append(file_path)

        return script_paths

    def _build_graph(
        self,
        parsed_scripts: List[ParsedScript],
        target_table: str
    ) -> LineageGraph:
        """Build lineage graph from parsed scripts"""
        graph = LineageGraph()

        # Create nodes for all data locations
        location_to_node = {}

        for script in parsed_scripts:
            # Add script node
            script_node_id = f"script:{script.script_name}"
            graph.add_node(LineageNode(
                node_id=script_node_id,
                node_type="script",
                name=script.script_name,
                details={
                    'path': script.script_path,
                    'type': script.script_type.value,
                    'transformations': [t.value for t in script.transformations],
                    'confidence': script.confidence
                }
            ))

            # Add nodes for input locations
            for loc in script.reads_from:
                loc_id = f"{loc.location_type}:{loc.path}"
                if loc_id not in graph.nodes:
                    graph.add_node(LineageNode(
                        node_id=loc_id,
                        node_type="data_location",
                        name=loc.path,
                        details={
                            'location_type': loc.location_type,
                            'is_external': loc.is_external
                        }
                    ))
                    location_to_node[loc.path] = loc_id

                # Add edge: location -> script (script reads from location)
                graph.add_edge(LineageEdge(
                    source_id=loc_id,
                    target_id=script_node_id,
                    edge_type="reads",
                    script=script.script_name
                ))

            # Add nodes for output locations
            for loc in script.writes_to:
                loc_id = f"{loc.location_type}:{loc.path}"
                if loc_id not in graph.nodes:
                    graph.add_node(LineageNode(
                        node_id=loc_id,
                        node_type="data_location",
                        name=loc.path,
                        details={
                            'location_type': loc.location_type,
                            'is_external': loc.is_external
                        }
                    ))
                    location_to_node[loc.path] = loc_id

                # Add edge: script -> location (script writes to location)
                graph.add_edge(LineageEdge(
                    source_id=script_node_id,
                    target_id=loc_id,
                    edge_type="writes",
                    script=script.script_name
                ))

        # Identify entry and exit points
        node_inputs = {edge.target_id for edge in graph.edges}
        node_outputs = {edge.source_id for edge in graph.edges}

        for node_id in graph.nodes:
            if node_id not in node_inputs and graph.nodes[node_id].node_type == "data_location":
                graph.entry_points.append(node_id)
            if node_id not in node_outputs and graph.nodes[node_id].node_type == "data_location":
                graph.exit_points.append(node_id)

        return graph
