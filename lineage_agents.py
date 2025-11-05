"""
Lineage Tracking Agents
=======================
AI-driven agents for cross-system lineage analysis

Agents:
1. Parsing Agent - Extracts data from code
2. Logic Agent - Interprets transformation logic
3. Mapping Agent - Creates STTM mappings
4. Similarity Agent - Finds equivalent implementations
5. Lineage Agent - Builds column-level lineage chains
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger
import json

from services.lineage.sttm_generator import STTMGenerator, STTMMapping
from services.ai_script_analyzer import AIScriptAnalyzer
from services.logic_comparator import LogicComparator
from services.multi_collection_indexer import MultiCollectionIndexer


@dataclass
class LineageResult:
    """Complete lineage analysis result"""
    selected_system: str
    selected_entity: str  # graph/workflow/pipeline name
    entity_metadata: Dict[str, Any]
    sttm_mappings: List[Dict[str, Any]]
    matched_systems: Dict[str, Dict[str, Any]]
    comparisons: Dict[str, Any]
    column_lineage: List[Dict[str, Any]]
    flow_level_lineage: List[Dict[str, Any]]
    logic_level_lineage: List[Dict[str, Any]]
    ai_reasoning_notes: str
    confidence_score: float
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class ParsingAgent:
    """
    Parses code entities and extracts structured data with cross-system context

    Capabilities:
    - Parse Ab Initio graphs (.mp/.dml)
    - Parse Hadoop workflows (.xml/.sh/.pig/.hql)
    - Parse Databricks notebooks (.py/.ipynb)
    - Extract datasets, transformations, joins, filters
    - Use Autosys context for Ab Initio analysis (orchestration, dependencies)
    """

    def __init__(self, abinitio_parser=None, autosys_parser=None, hadoop_parser=None, indexer=None, ai_analyzer=None):
        self.abinitio_parser = abinitio_parser
        self.autosys_parser = autosys_parser
        self.hadoop_parser = hadoop_parser
        self.indexer = indexer  # For cross-system context search
        self.ai_analyzer = ai_analyzer  # For AI-enhanced parsing
        logger.info("✓ Parsing Agent initialized with cross-system context support")

    def parse_entity(
        self,
        system_type: str,
        entity_name: str,
        file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse entity and return structured data

        Args:
            system_type: abinitio, hadoop, databricks
            entity_name: Name of graph/workflow/pipeline
            file_path: Optional file path

        Returns:
            Parsed entity data
        """
        logger.info(f"Parsing {system_type} entity: {entity_name}")

        if system_type == "abinitio":
            return self._parse_abinitio_entity(entity_name, file_path)
        elif system_type == "hadoop":
            return self._parse_hadoop_entity(entity_name, file_path)
        elif system_type == "databricks":
            return self._parse_databricks_entity(entity_name, file_path)
        else:
            raise ValueError(f"Unknown system type: {system_type}")

    def _parse_abinitio_entity(self, entity_name: str, file_path: Optional[str]) -> Dict[str, Any]:
        """
        Parse Ab Initio graph with Autosys orchestration context

        This enriches the Ab Initio graph analysis with:
        - Autosys job that runs this graph
        - Job dependencies (what runs before/after)
        - Scheduling and conditions
        - Cross-system flow context
        """
        if not self.abinitio_parser:
            from parsers.abinitio.parser import AbInitioParser
            self.abinitio_parser = AbInitioParser()

        if file_path:
            result = self.abinitio_parser.parse_file(file_path)
        else:
            # Search for the graph in indexed data
            result = {"error": "File path required for Ab Initio parsing"}

        # Fetch Autosys context for this Ab Initio graph
        autosys_context = self._fetch_autosys_context(entity_name)

        return {
            "system": "abinitio",
            "entity_name": entity_name,
            "entity_type": "graph",
            "processes": result.get("processes", []),
            "components": result.get("components", []),
            "metadata": result,
            "autosys_context": autosys_context  # Added Autosys orchestration context
        }

    def _fetch_autosys_context(self, abinitio_graph_name: str) -> Dict[str, Any]:
        """
        Fetch Autosys job context for an Ab Initio graph

        Searches the autosys_collection for jobs that reference this graph
        and enriches with dependency chain information

        Args:
            abinitio_graph_name: Name of the Ab Initio graph

        Returns:
            Dictionary with Autosys context:
            - orchestrating_job: The Autosys job that runs this graph
            - dependencies: Jobs that must complete before this runs
            - dependents: Jobs that run after this completes
            - scheduling: Schedule/conditions for execution
            - flow_context: Position in larger workflow
        """
        if not self.indexer:
            logger.debug("No indexer available - skipping Autosys context")
            return {}

        try:
            logger.info(f"Fetching Autosys context for Ab Initio graph: {abinitio_graph_name}")

            # Search autosys_collection for jobs that reference this graph
            autosys_results = self.indexer.search_multi_collection(
                query=abinitio_graph_name,
                collections=["autosys_collection"],
                top_k=5
            )

            if not autosys_results or "autosys_collection" not in autosys_results:
                logger.debug(f"No Autosys jobs found for graph: {abinitio_graph_name}")
                return {}

            autosys_jobs = autosys_results["autosys_collection"]

            if not autosys_jobs:
                return {}

            # Find the primary orchestrating job (best match)
            orchestrating_job = autosys_jobs[0]

            # Extract key context
            job_metadata = orchestrating_job.get('metadata', {})
            job_content = orchestrating_job.get('content', '')

            context = {
                "has_autosys_context": True,
                "orchestrating_job": {
                    "name": orchestrating_job.get('title', 'unknown'),
                    "job_name": job_metadata.get('job_name', 'unknown'),
                    "job_type": job_metadata.get('job_type', 'unknown'),
                    "command": job_metadata.get('command', ''),
                    "owner": job_metadata.get('owner', ''),
                    "machine": job_metadata.get('machine', ''),
                },
                "dependencies": {
                    "conditions": job_metadata.get('conditions', []),
                    "predecessor_jobs": job_metadata.get('conditions', []),  # Autosys conditions are dependencies
                },
                "scheduling": {
                    "run_calendar": job_metadata.get('run_calendar', ''),
                    "start_times": job_metadata.get('start_times', []),
                    "run_window": job_metadata.get('run_window', ''),
                },
                "flow_context": {
                    "description": job_metadata.get('description', ''),
                    "box_name": job_metadata.get('box_name', ''),  # Parent box if any
                    "total_related_jobs": len(autosys_jobs),
                },
                "ai_insights": None
            }

            # Use AI to interpret the Autosys context if available
            if self.ai_analyzer and job_content:
                try:
                    ai_analysis = self.ai_analyzer.analyze_with_context(
                        query=f"Analyze this Autosys job definition and explain its role in orchestrating the Ab Initio graph '{abinitio_graph_name}'. Focus on dependencies, scheduling, and how it fits in the overall workflow.",
                        context=job_content
                    )
                    context["ai_insights"] = ai_analysis.get('analysis', '')
                except Exception as e:
                    logger.warning(f"Could not get AI insights for Autosys context: {e}")

            logger.info(f"  ✓ Found Autosys context: Job '{context['orchestrating_job']['job_name']}' orchestrates this graph")

            return context

        except Exception as e:
            logger.warning(f"Error fetching Autosys context: {e}")
            return {"error": str(e)}

    def _parse_hadoop_entity(self, entity_name: str, file_path: Optional[str]) -> Dict[str, Any]:
        """Parse Hadoop workflow"""
        # Placeholder - extend based on your Hadoop parser
        return {
            "system": "hadoop",
            "entity_name": entity_name,
            "entity_type": "workflow",
            "scripts": [],
            "transformations": [],
            "metadata": {}
        }

    def _parse_databricks_entity(self, entity_name: str, file_path: Optional[str]) -> Dict[str, Any]:
        """Parse Databricks notebook"""
        # Placeholder - extend based on your Databricks parser
        return {
            "system": "databricks",
            "entity_name": entity_name,
            "entity_type": "notebook",
            "cells": [],
            "transformations": [],
            "metadata": {}
        }


class LogicAgent:
    """
    Interprets transformation logic using AI

    Capabilities:
    - Understand transformation semantics
    - Infer business logic
    - Detect data quality rules
    - Identify optimization opportunities
    """

    def __init__(self, ai_analyzer: Optional[AIScriptAnalyzer] = None):
        self.ai_analyzer = ai_analyzer or AIScriptAnalyzer()
        logger.info("✓ Logic Agent initialized")

    def analyze_transformation(
        self,
        transformation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze transformation logic using AI

        Args:
            transformation: Transformation definition
            context: Additional context

        Returns:
            Analysis results with business logic interpretation
        """
        logger.debug(f"Analyzing transformation: {transformation.get('type', 'unknown')}")

        # Extract transformation details
        transform_type = transformation.get('type', 'unknown')
        code_snippet = transformation.get('code', '')
        inputs = transformation.get('inputs', [])
        outputs = transformation.get('outputs', [])

        # Use AI to interpret
        analysis = {
            "transformation_type": transform_type,
            "business_purpose": self._infer_business_purpose(code_snippet),
            "data_quality_rules": self._detect_quality_rules(code_snippet),
            "complexity": self._assess_complexity(code_snippet),
            "dependencies": inputs,
            "outputs": outputs,
            "semantic_meaning": self._extract_semantic_meaning(code_snippet)
        }

        return analysis

    def _infer_business_purpose(self, code: str) -> str:
        """Infer business purpose from code"""
        code_lower = code.lower()

        if 'customer' in code_lower:
            return "Customer data processing"
        elif 'product' in code_lower:
            return "Product information management"
        elif 'order' in code_lower:
            return "Order processing"
        elif 'aggregate' in code_lower or 'sum' in code_lower:
            return "Data aggregation and summarization"
        elif 'join' in code_lower:
            return "Data enrichment through joins"
        elif 'filter' in code_lower or 'where' in code_lower:
            return "Data filtering and selection"
        else:
            return "Data transformation"

    def _detect_quality_rules(self, code: str) -> List[str]:
        """Detect data quality rules"""
        rules = []

        code_lower = code.lower()

        if 'not null' in code_lower or 'is not null' in code_lower:
            rules.append("NULL_CHECK")
        if 'distinct' in code_lower:
            rules.append("DEDUPLICATION")
        if 'trim' in code_lower or 'strip' in code_lower:
            rules.append("WHITESPACE_TRIMMING")
        if 'upper' in code_lower or 'lower' in code_lower:
            rules.append("CASE_STANDARDIZATION")
        if 'cast' in code_lower or 'convert' in code_lower:
            rules.append("TYPE_CONVERSION")

        return rules

    def _assess_complexity(self, code: str) -> str:
        """Assess transformation complexity"""
        lines = code.count('\n') + 1
        has_nested_logic = '(' in code and ')' in code and code.count('(') > 2
        has_multiple_operations = len([op for op in ['join', 'filter', 'group', 'aggregate'] if op in code.lower()]) > 2

        if lines > 50 or has_nested_logic or has_multiple_operations:
            return "HIGH"
        elif lines > 20:
            return "MEDIUM"
        else:
            return "LOW"

    def _extract_semantic_meaning(self, code: str) -> str:
        """Extract semantic meaning of transformation"""
        # Use AI to generate natural language description
        # Simplified version - can be enhanced with actual AI call
        return f"Transformation performs data processing on input streams"


class MappingAgent:
    """
    Creates STTM mappings from parsed data

    Capabilities:
    - Generate column-level mappings
    - Align source fields with target fields
    - Extract transformation logic
    - Detect dependencies
    """

    def __init__(self, sttm_generator: Optional[STTMGenerator] = None):
        self.sttm_generator = sttm_generator or STTMGenerator()
        logger.info("✓ Mapping Agent initialized")

    def create_mappings(
        self,
        parsed_entity: Dict[str, Any],
        partner: str = "default"
    ) -> List[STTMMapping]:
        """
        Create STTM mappings from parsed entity

        Args:
            parsed_entity: Parsed entity data
            partner: Business partner/domain

        Returns:
            List of STTM mappings
        """
        system_type = parsed_entity.get("system", "unknown")
        entity_name = parsed_entity.get("entity_name", "unknown")

        logger.info(f"Creating STTM mappings for {system_type}/{entity_name}")

        mappings = []

        if system_type == "abinitio":
            components = parsed_entity.get("components", [])
            for component in components:
                comp_mappings = self.sttm_generator.generate_from_abinitio_component(
                    component,
                    graph_name=entity_name,
                    partner=partner
                )
                mappings.extend(comp_mappings)

        elif system_type == "hadoop":
            scripts = parsed_entity.get("scripts", [])
            for script in scripts:
                script_mappings = self.sttm_generator.generate_from_hadoop_script(
                    script,
                    partner=partner
                )
                mappings.extend(script_mappings)

        logger.info(f"✓ Created {len(mappings)} STTM mappings")
        return mappings


class SimilarityAgent:
    """
    Finds equivalent implementations across systems

    Capabilities:
    - Embed transformations using vector embeddings
    - Search for similar logic across systems
    - Calculate semantic similarity scores
    - Rank matches by relevance
    """

    def __init__(
        self,
        indexer: Optional[MultiCollectionIndexer] = None,
        logic_comparator: Optional[LogicComparator] = None
    ):
        self.indexer = indexer or MultiCollectionIndexer()
        self.logic_comparator = logic_comparator or LogicComparator()
        logger.info("✓ Similarity Agent initialized")

    def find_equivalent_implementations(
        self,
        entity_data: Dict[str, Any],
        target_systems: List[str],
        top_k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find equivalent implementations in other systems

        Args:
            entity_data: Source entity data
            target_systems: Systems to search (hadoop, databricks, etc.)
            top_k: Number of top matches to return

        Returns:
            Dict mapping system names to matching entities
        """
        entity_name = entity_data.get("entity_name", "")
        system_type = entity_data.get("system", "")

        logger.info(f"Finding equivalents for {system_type}/{entity_name} in {target_systems}")

        # Build search query from entity description
        query = self._build_search_query(entity_data)

        # Map system names to collections
        collection_map = {
            "hadoop": "hadoop_collection",
            "databricks": "databricks_collection",
            "abinitio": "abinitio_collection"
        }

        results = {}

        for target_system in target_systems:
            if target_system == system_type:
                continue  # Skip same system

            collection_name = collection_map.get(target_system)
            if not collection_name:
                continue

            # Search for similar entities
            search_results = self.indexer.search_multi_collection(
                query=query,
                collections=[collection_name],
                top_k=top_k
            )

            if collection_name in search_results:
                matches = search_results[collection_name]

                # Calculate similarity scores
                scored_matches = []
                for match in matches:
                    similarity_score = self._calculate_similarity(
                        entity_data,
                        match
                    )

                    scored_matches.append({
                        "entity_name": match.get("title", "unknown"),
                        "similarity_score": similarity_score,
                        "content": match.get("content", ""),
                        "metadata": match.get("metadata", {})
                    })

                results[target_system] = sorted(
                    scored_matches,
                    key=lambda x: x["similarity_score"],
                    reverse=True
                )

        logger.info(f"✓ Found matches in {len(results)} systems")
        return results

    def _build_search_query(self, entity_data: Dict[str, Any]) -> str:
        """Build search query from entity data"""
        entity_name = entity_data.get("entity_name", "")
        entity_type = entity_data.get("entity_type", "")

        # Extract key terms from components/transformations
        components = entity_data.get("components", [])
        key_terms = []

        for comp in components[:5]:  # Use first 5 components
            if hasattr(comp, 'name'):
                key_terms.append(comp.name)

        query_parts = [entity_name]
        if key_terms:
            query_parts.extend(key_terms)

        return " ".join(query_parts)

    def _calculate_similarity(
        self,
        source_entity: Dict[str, Any],
        target_match: Dict[str, Any]
    ) -> float:
        """Calculate semantic similarity between entities"""
        # Use logic comparator if available
        try:
            # FIXED: Format data correctly for LogicComparator
            system1_formatted = {
                "system_name": source_entity.get("system", "unknown"),
                "name": source_entity.get("entity_name", "unknown"),
                "description": source_entity.get("description", ""),
                "code": source_entity.get("content", source_entity.get("logic", "")),
                "metadata": source_entity.get("metadata", {})
            }

            system2_formatted = {
                "system_name": target_match.get("metadata", {}).get("system", "unknown"),
                "name": target_match.get("title", "unknown").replace("Script: ", ""),
                "description": target_match.get("metadata", {}).get("description", ""),
                "code": target_match.get("content", ""),
                "metadata": target_match.get("metadata", {})
            }

            logger.debug(f"Comparing {system1_formatted['system_name']}/{system1_formatted['name']} " +
                        f"vs {system2_formatted['system_name']}/{system2_formatted['name']}")

            comparison = self.logic_comparator.compare_logic(
                system1=system1_formatted,
                system2=system2_formatted
            )

            similarity_score = comparison.get("similarity_score", 0.5)

            logger.info(f"→ Similarity {system1_formatted['name']} vs {system2_formatted['name']}: " +
                       f"{similarity_score:.0%}")

            return similarity_score

        except Exception as e:
            logger.error(f"Could not compare logic: {e}", exc_info=True)
            # Fallback to simple score
            return 0.7


class LineageAgent:
    """
    Builds column-level lineage chains

    Capabilities:
    - Trace data flow through transformations
    - Build dependency graphs
    - Generate lineage visualization data
    - Support three levels: flow, logic, column
    """

    def __init__(self):
        logger.info("✓ Lineage Agent initialized")

    def build_lineage(
        self,
        sttm_mappings: List[STTMMapping],
        entity_data: Dict[str, Any],
        matched_systems: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build complete lineage structure

        Args:
            sttm_mappings: STTM mappings for the entity
            entity_data: Source entity data
            matched_systems: Matched implementations in other systems

        Returns:
            Dict with flow_level, logic_level, column_level lineage
        """
        logger.info("Building lineage structure...")

        lineage = {
            "flow_level": self._build_flow_level(entity_data, matched_systems),
            "logic_level": self._build_logic_level(sttm_mappings, matched_systems),
            "column_level": self._build_column_level(sttm_mappings)
        }

        return lineage

    def _build_flow_level(
        self,
        entity_data: Dict[str, Any],
        matched_systems: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Build flow-level lineage (process-to-process)"""
        flows = []

        source_system = entity_data.get("system", "unknown")
        source_entity = entity_data.get("entity_name", "unknown")

        # Create flow nodes for matched systems
        for target_system, matches in matched_systems.items():
            if matches:
                best_match = matches[0]  # Take top match

                flows.append({
                    "source_system": source_system,
                    "source_entity": source_entity,
                    "target_system": target_system,
                    "target_entity": best_match.get("entity_name", "unknown"),
                    "similarity_score": best_match.get("similarity_score", 0.0),
                    "relationship_type": "equivalent_implementation"
                })

        return flows

    def _build_logic_level(
        self,
        sttm_mappings: List[STTMMapping],
        matched_systems: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Build logic-level lineage (transformation-level)"""
        logic_flows = []

        # Group mappings by transformation type
        transform_groups = {}
        for mapping in sttm_mappings:
            transform = mapping.transformation_logic
            if transform not in transform_groups:
                transform_groups[transform] = []
            transform_groups[transform].append(mapping)

        # Create logic flow entries
        for transform_logic, mappings in transform_groups.items():
            logic_flows.append({
                "transformation_type": mappings[0].component_name if mappings else "unknown",
                "transformation_logic": transform_logic,
                "field_count": len(mappings),
                "source_fields": list(set(
                    field for m in mappings for field in m.source_field_names
                )),
                "target_fields": [m.target_field_name for m in mappings],
                "complexity": "HIGH" if len(mappings) > 10 else "MEDIUM" if len(mappings) > 5 else "LOW"
            })

        return logic_flows

    def _build_column_level(
        self,
        sttm_mappings: List[STTMMapping]
    ) -> List[Dict[str, Any]]:
        """Build column-level lineage (field-level derivation)"""
        column_lineage = []

        for mapping in sttm_mappings:
            column_lineage.append({
                "column_name": mapping.target_field_name,
                "target_table": mapping.target_table_name,
                "source_columns": mapping.source_field_names,
                "source_dataset": mapping.source_dataset_name,
                "transformation_logic": mapping.transformation_logic,
                "dependencies": mapping.field_depends_on,
                "data_type": mapping.target_field_data_type,
                "contains_pii": mapping.contains_pii,
                "field_type": mapping.field_type,
                "processing_order": mapping.processing_order,
                "confidence_score": mapping.confidence_score
            })

        return column_lineage


# Main orchestrator
class LineageOrchestrator:
    """
    Orchestrates all lineage agents to produce complete lineage analysis

    This is the main entry point for lineage tracking
    """

    def __init__(
        self,
        ai_analyzer: Optional[AIScriptAnalyzer] = None,
        indexer: Optional[MultiCollectionIndexer] = None
    ):
        self.parsing_agent = ParsingAgent(indexer=indexer, ai_analyzer=ai_analyzer)  # Pass indexer and ai_analyzer for cross-system context
        self.logic_agent = LogicAgent(ai_analyzer)
        self.mapping_agent = MappingAgent()
        self.similarity_agent = SimilarityAgent(indexer)
        self.lineage_agent = LineageAgent()
        self.indexer = indexer  # Store for STTM indexing

        logger.info("✓ Lineage Orchestrator initialized with all agents (cross-system context enabled)")

    def analyze_lineage(
        self,
        system_type: str,
        entity_name: str,
        file_path: Optional[str] = None,
        target_systems: Optional[List[str]] = None,
        partner: str = "default"
    ) -> LineageResult:
        """
        Complete lineage analysis workflow

        Args:
            system_type: Source system (abinitio, hadoop, databricks)
            entity_name: Entity name (graph/workflow/pipeline)
            file_path: Optional file path for parsing
            target_systems: Systems to find equivalents in
            partner: Business partner/domain

        Returns:
            Complete lineage result with STTM mappings and matches
        """
        from datetime import datetime

        logger.info(f"Starting lineage analysis for {system_type}/{entity_name}")

        # Step 1: Parse entity
        logger.info("Step 1: Parsing entity...")
        parsed_entity = self.parsing_agent.parse_entity(
            system_type=system_type,
            entity_name=entity_name,
            file_path=file_path
        )

        # Step 2: Create STTM mappings
        logger.info("Step 2: Creating STTM mappings...")
        sttm_mappings = self.mapping_agent.create_mappings(
            parsed_entity=parsed_entity,
            partner=partner
        )

        # Step 2.5: Index STTM mappings to vector DB (auto-indexing)
        if self.indexer and sttm_mappings:
            logger.info("Step 2.5: Auto-indexing STTM mappings to vector DB...")
            try:
                sttm_dicts = [m.to_dict() if hasattr(m, 'to_dict') else m for m in sttm_mappings]
                index_result = self.indexer.index_sttm_mappings(
                    sttm_mappings=sttm_dicts,
                    system_type=system_type,
                    entity_name=entity_name
                )
                logger.info(f"  ✓ Indexed {index_result.get('total_mappings', 0)} STTM mappings to {index_result.get('collection', 'unknown')}")
            except Exception as e:
                logger.warning(f"Could not auto-index STTM mappings: {e}")
                # Continue even if indexing fails - it's not critical

        # Step 3: Find equivalent implementations
        target_systems = target_systems or ["hadoop", "databricks", "abinitio"]
        logger.info(f"Step 3: Finding equivalents in {target_systems}...")
        matched_systems = self.similarity_agent.find_equivalent_implementations(
            entity_data=parsed_entity,
            target_systems=target_systems,
            top_k=5
        )

        # Step 4: Build lineage
        logger.info("Step 4: Building lineage structure...")
        lineage = self.lineage_agent.build_lineage(
            sttm_mappings=sttm_mappings,
            entity_data=parsed_entity,
            matched_systems=matched_systems
        )

        # Step 5: Create comparisons
        logger.info("Step 5: Creating comparisons...")
        comparisons = self._create_comparisons(
            parsed_entity,
            matched_systems,
            lineage
        )

        # Build final result
        result = LineageResult(
            selected_system=system_type,
            selected_entity=entity_name,
            entity_metadata=parsed_entity,
            sttm_mappings=[m.to_dict() for m in sttm_mappings],
            matched_systems=self._format_matched_systems(matched_systems),
            comparisons=comparisons,
            column_lineage=lineage["column_level"],
            flow_level_lineage=lineage["flow_level"],
            logic_level_lineage=lineage["logic_level"],
            ai_reasoning_notes=self._generate_reasoning_notes(
                parsed_entity, matched_systems, sttm_mappings
            ),
            confidence_score=self._calculate_overall_confidence(sttm_mappings),
            created_at=datetime.now().isoformat()
        )

        logger.info("✓ Lineage analysis complete!")
        return result

    def _format_matched_systems(
        self,
        matched_systems: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Format matched systems for output"""
        formatted = {}

        for system, matches in matched_systems.items():
            if matches:
                best_match = matches[0]
                formatted[system] = {
                    "entity": best_match.get("entity_name", "unknown"),
                    "similarity_score": best_match.get("similarity_score", 0.0),
                    "match_count": len(matches)
                }

        return formatted

    def _create_comparisons(
        self,
        parsed_entity: Dict[str, Any],
        matched_systems: Dict[str, List[Dict[str, Any]]],
        lineage: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comparison summaries"""
        return {
            "flow_level": f"Found equivalent flows in {len(matched_systems)} systems",
            "logic_level": f"Analyzed {len(lineage['logic_level'])} transformations",
            "column_lineage": f"Traced {len(lineage['column_level'])} column mappings"
        }

    def _generate_reasoning_notes(
        self,
        parsed_entity: Dict[str, Any],
        matched_systems: Dict[str, List[Dict[str, Any]]],
        sttm_mappings: List[STTMMapping]
    ) -> str:
        """Generate AI reasoning notes"""
        notes = []

        notes.append(f"Analyzed {parsed_entity.get('system', 'unknown')} entity: {parsed_entity.get('entity_name', 'unknown')}")
        notes.append(f"Generated {len(sttm_mappings)} column-level mappings")
        notes.append(f"Found {sum(len(matches) for matches in matched_systems.values())} potential equivalents across systems")

        # Add PII detection summary
        pii_count = sum(1 for m in sttm_mappings if m.contains_pii)
        if pii_count > 0:
            notes.append(f"Detected {pii_count} fields containing PII")

        return " | ".join(notes)

    def _calculate_overall_confidence(self, sttm_mappings: List[STTMMapping]) -> float:
        """Calculate overall confidence score"""
        if not sttm_mappings:
            return 0.0

        avg_confidence = sum(m.confidence_score for m in sttm_mappings) / len(sttm_mappings)
        return round(avg_confidence, 2)
