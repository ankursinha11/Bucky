"""
Ab Initio Deep Parser
=====================
Deep parsing of Ab Initio graphs with component-level logic extraction

Features:
- Graph-level repository analysis (Tier 1)
- GraphFlow execution diagrams (Tier 2)
- Component-level transformation logic (Tier 3)
- Transform script extraction and parsing
- DML schema extraction
- Data lineage tracking through components
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

from core.models import (
    Repository, RepositoryType,
    WorkflowFlow, ActionNode, FlowEdge, ActionType,
    ScriptLogic, Transformation, ColumnLineage, TransformationType
)
from .parser import AbInitioParser


class DeepAbInitioParser:
    """
    Deep parser for Ab Initio with 3-tier architecture:
    - Tier 1: Repository (overall graph statistics and analysis)
    - Tier 2: WorkflowFlow (graph execution flow with component connections)
    - Tier 3: ScriptLogic (component transformations and data lineage)
    """

    def __init__(self, use_ai: bool = True):
        """
        Initialize deep parser

        Args:
            use_ai: Whether to use AI analysis (default: True)
        """
        self.base_parser = AbInitioParser()
        self.use_ai = use_ai
        self.ai_analyzer = None

        if use_ai:
            try:
                from services.ai_script_analyzer import AIScriptAnalyzer
                self.ai_analyzer = AIScriptAnalyzer()
            except Exception as e:
                logger.warning(f"AI analyzer not available: {e}")
                self.use_ai = False

    def parse_directory(self, abinitio_path: str) -> Dict[str, Any]:
        """
        Deep parse Ab Initio directory

        Args:
            abinitio_path: Path to Ab Initio files

        Returns:
            Dict with repository, workflow_flows, and script_logics
        """
        logger.info(f"🚀 DEEP PARSING Ab Initio: {abinitio_path}")

        # Step 1: Parse with base parser
        base_result = self.base_parser.parse_directory(abinitio_path)
        processes = base_result.get("processes", [])
        components = base_result.get("components", [])
        raw_mp_data = base_result.get("raw_mp_data", [])

        logger.info(f"Base parsing: {len(processes)} graphs, {len(components)} components")

        # Step 2: Create Tier 1 - Repository
        repository = self._create_repository(
            abinitio_path,
            processes,
            components
        )

        # Step 3: Create Tier 2 - WorkflowFlows (one per graph)
        workflow_flows = []
        for i, process in enumerate(processes):
            mp_data = raw_mp_data[i] if i < len(raw_mp_data) else {}
            workflow_flow = self._create_workflow_flow(process, mp_data)
            workflow_flows.append(workflow_flow)

        # Step 4: Create Tier 3 - ScriptLogics (component-level logic)
        script_logics = []
        for process in processes:
            # Get components for this process
            process_components = [c for c in components if c.process_id == process.id]

            for component in process_components:
                script_logic = self._create_script_logic(
                    component,
                    process,
                    abinitio_path
                )
                if script_logic:
                    script_logics.append(script_logic)

        logger.info(f"✓ Deep parsing complete:")
        logger.info(f"  - Repository: {repository.name}")
        logger.info(f"  - Workflows: {len(workflow_flows)}")
        logger.info(f"  - Component Logics: {len(script_logics)}")

        # Step 5: AI Analysis (if enabled)
        if self.use_ai and self.ai_analyzer:
            logger.info("Running AI analysis...")

            # Analyze repository
            try:
                repo_analysis = self.ai_analyzer.analyze_repository(repository)
                repository.ai_summary = repo_analysis.get("business_purpose", "")
                repository.ai_architecture = repo_analysis.get("architecture_summary", "")
            except Exception as e:
                logger.warning(f"Repository AI analysis failed: {e}")

            # Analyze workflows
            for workflow in workflow_flows:
                try:
                    workflow.ai_flow_summary = self.ai_analyzer.analyze_workflow_flow(workflow)
                except Exception as e:
                    logger.warning(f"Workflow AI analysis failed: {e}")

            # Analyze component logics
            for script_logic in script_logics:
                try:
                    self.ai_analyzer.analyze_script(script_logic)
                except Exception as e:
                    logger.warning(f"Script AI analysis failed: {e}")

        return {
            "repository": repository,
            "workflow_flows": workflow_flows,
            "script_logics": script_logics,
            "summary": {
                "total_graphs": len(processes),
                "total_components": len(components),
                "total_transformations": sum(
                    len(sl.transformations) for sl in script_logics
                ),
                "source_path": abinitio_path,
            }
        }

    def _create_repository(
        self,
        path: str,
        processes: List,
        components: List
    ) -> Repository:
        """
        Create Tier 1 Repository object

        Args:
            path: Repository path
            processes: List of Process objects
            components: List of Component objects

        Returns:
            Repository object
        """
        repo_name = Path(path).name
        repo_hash = hashlib.md5(str(Path(path).as_posix()).encode()).hexdigest()[:8]

        # Count component types
        transform_count = sum(1 for c in components if 'transform' in c.component_type.value.lower())
        input_count = sum(1 for c in components if 'input' in c.component_type.value.lower())
        output_count = sum(1 for c in components if 'output' in c.component_type.value.lower())

        # Identify business domains from graph/component names
        business_domains = self._extract_business_domains(processes, components)

        repository = Repository(
            id=f"abinitio_repo_{repo_name}_{repo_hash}",
            name=repo_name,
            repo_type=RepositoryType.ABINITIO,
            base_path=path,
            total_workflows=len(processes),
            total_scripts=len(components),  # Components are like scripts
            total_components=len(components),
            business_domains=business_domains,
            technologies=["Ab Initio GDE", "Co>Operating System", "Data Profiler"],
            description=f"Ab Initio repository with {len(processes)} graphs, {len(components)} components "
                       f"({transform_count} transforms, {input_count} inputs, {output_count} outputs)",
        )

        logger.info(f"Created Repository: {repository.name} with {len(processes)} graphs")

        return repository

    def _extract_business_domains(self, processes: List, components: List) -> List[str]:
        """
        Extract business domains from graph and component names

        Args:
            processes: List of processes
            components: List of components

        Returns:
            List of business domain strings
        """
        domains = set()

        # Common business domain keywords
        domain_keywords = [
            'customer', 'account', 'transaction', 'payment', 'order',
            'product', 'inventory', 'sales', 'finance', 'billing',
            'claims', 'policy', 'member', 'patient', 'provider',
            'enrollment', 'eligibility', 'demographics', 'address'
        ]

        # Check process names
        for process in processes:
            name_lower = process.name.lower()
            for keyword in domain_keywords:
                if keyword in name_lower:
                    domains.add(keyword.title())

        # Check component names
        for comp in components[:50]:  # Check first 50 components
            name_lower = comp.name.lower()
            for keyword in domain_keywords:
                if keyword in name_lower:
                    domains.add(keyword.title())

        return sorted(list(domains))

    def _create_workflow_flow(self, process, mp_data: Dict[str, Any]) -> WorkflowFlow:
        """
        Create Tier 2 WorkflowFlow object from graph

        Args:
            process: Process object
            mp_data: Raw MP data with graph_flow

        Returns:
            WorkflowFlow object
        """
        # Extract component actions
        actions = []
        components_data = mp_data.get("components", [])

        for comp in components_data:
            action_type = self._map_component_to_action_type(comp.get("component_type", ""))

            action = ActionNode(
                action_id=comp.get("component_id", ""),
                action_name=comp.get("component_name", ""),
                action_type=action_type,
                script_path=None,  # Ab Initio components are embedded
                script_content_id=f"{process.id}_{comp.get('component_id', '')}",
                input_tables=[],
                output_tables=[],
                parameters={}
            )
            actions.append(action)

        # Extract flow edges
        edges = []
        graph_flow = mp_data.get("graph_flow", [])

        for flow in graph_flow:
            edge = FlowEdge(
                edge_id=f"{flow.get('source_component_id', '')}_{flow.get('target_component_id', '')}",
                from_action=flow.get("source_component_id", ""),
                to_action=flow.get("target_component_id", ""),
                edge_type="data_flow"
            )
            edges.append(edge)

        # Generate flow diagrams
        flow_diagram_ascii = self._generate_ascii_flow(actions, edges)
        flow_diagram_mermaid = self._generate_mermaid_flow(actions, edges)

        workflow_flow = WorkflowFlow(
            workflow_id=process.id,
            workflow_name=process.name,
            workflow_type="abinitio_graph",
            actions=actions,
            edges=edges,
            flow_diagram_ascii=flow_diagram_ascii,
            flow_diagram_mermaid=flow_diagram_mermaid
        )

        logger.info(f"Created WorkflowFlow: {process.name} with {len(actions)} actions")

        return workflow_flow

    def _map_component_to_action_type(self, comp_type: str) -> ActionType:
        """
        Map Ab Initio component type to ActionType

        Args:
            comp_type: Component type string

        Returns:
            ActionType enum
        """
        type_mapping = {
            'Input_File': ActionType.INPUT,
            'Output_File': ActionType.OUTPUT,
            'Transform': ActionType.TRANSFORM,
            'Reformat': ActionType.TRANSFORM,
            'Join': ActionType.JOIN,
            'Filter': ActionType.FILTER,
            'Aggregate': ActionType.AGGREGATE,
            'Rollup': ActionType.AGGREGATE,
            'Sort': ActionType.SORT,
            'Lookup_File': ActionType.LOOKUP,
        }

        return type_mapping.get(comp_type, ActionType.SCRIPT)

    def _create_script_logic(
        self,
        component,
        process,
        base_path: str
    ) -> Optional[ScriptLogic]:
        """
        Create Tier 3 ScriptLogic from component

        Args:
            component: Component object
            process: Parent Process object
            base_path: Base repository path

        Returns:
            ScriptLogic object or None
        """
        script_id = component.id

        # Build content from component parameters
        raw_content = self._build_component_content(component)

        # Generate content hash
        content_hash = hashlib.md5(raw_content.encode()).hexdigest()

        # Get script path from component
        script_path = component.file_path or f"abinitio://{process.name}/{component.name}"

        # Create script logic
        script_logic = ScriptLogic(
            script_id=script_id,
            script_name=component.name,
            script_type=component.component_type.value,
            script_path=script_path,
            raw_content=raw_content,
            content_hash=content_hash,
            workflow_id=process.id,
        )

        # Extract transformations based on component type
        self._extract_transformations(component, script_logic)

        # Extract column lineage
        self._extract_column_lineage(component, script_logic)

        # Extract inputs/outputs
        self._extract_io(component, script_logic)

        return script_logic

    def _build_component_content(self, component) -> str:
        """
        Build readable content from component parameters

        Args:
            component: Component object

        Returns:
            Formatted content string
        """
        lines = [
            f"Component: {component.name}",
            f"Type: {component.component_type.value}",
            f"",
            "Parameters:"
        ]

        for param_name, param_value in component.parameters.items():
            lines.append(f"  {param_name}: {param_value}")

        return "\n".join(lines)

    def _extract_transformations(self, component, script_logic: ScriptLogic):
        """
        Extract transformations from component based on type

        Args:
            component: Component object
            script_logic: ScriptLogic to update
        """
        comp_type = component.component_type.value.lower()
        params = component.parameters

        trans_count = 0

        # Transform/Reformat components
        if 'transform' in comp_type or 'reformat' in comp_type:
            # Look for transform_expr or similar parameters
            for param_name, param_value in params.items():
                if 'expr' in param_name.lower() or 'formula' in param_name.lower():
                    trans_count += 1
                    trans = Transformation(
                        transformation_id=f"{script_logic.script_id}_transform_{trans_count}",
                        transformation_type=TransformationType.TRANSFORM,
                        code_snippet=f"{param_name} = {param_value}",
                    )
                    script_logic.transformations.append(trans)

        # Filter components
        elif 'filter' in comp_type:
            filter_expr = params.get('filter_expr', params.get('condition', ''))
            if filter_expr:
                trans = Transformation(
                    transformation_id=f"{script_logic.script_id}_filter",
                    transformation_type=TransformationType.FILTER,
                    code_snippet=f"FILTER: {filter_expr}",
                    condition=filter_expr,
                )
                script_logic.transformations.append(trans)

        # Join components
        elif 'join' in comp_type:
            join_key = params.get('join_key', params.get('key', ''))
            join_type = params.get('join_type', 'inner')

            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_join",
                transformation_type=TransformationType.JOIN,
                code_snippet=f"JOIN ON {join_key} ({join_type})",
                condition=f"key={join_key}",
            )
            script_logic.transformations.append(trans)

        # Aggregate/Rollup components
        elif 'aggregate' in comp_type or 'rollup' in comp_type:
            group_by = params.get('group_by', params.get('key', ''))
            agg_expr = params.get('aggregate', params.get('expression', ''))

            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_aggregate",
                transformation_type=TransformationType.AGGREGATE,
                code_snippet=f"GROUP BY {group_by} AGGREGATE {agg_expr}",
                condition=f"group_by={group_by}",
            )
            script_logic.transformations.append(trans)

        # Sort components
        elif 'sort' in comp_type:
            sort_key = params.get('sort_key', params.get('key', ''))

            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_sort",
                transformation_type=TransformationType.SORT,
                code_snippet=f"SORT BY {sort_key}",
            )
            script_logic.transformations.append(trans)

    def _extract_column_lineage(self, component, script_logic: ScriptLogic):
        """
        Extract column-level lineage from component

        Args:
            component: Component object
            script_logic: ScriptLogic to update
        """
        params = component.parameters

        # Look for DML or layout parameters
        dml_param = params.get('dml', params.get('layout', params.get('record_format', '')))

        if dml_param:
            # Parse field definitions
            fields = self._parse_dml_fields(dml_param)

            # Use component name as table name
            table_name = component.name

            for field in fields:
                lineage = ColumnLineage(
                    source_table=table_name,
                    source_column=field,
                    target_table=table_name,
                    target_column=field,
                    transformation_logic=f"Field: {field}",
                    is_pass_through=True
                )
                script_logic.column_lineages.append(lineage)

    def _parse_dml_fields(self, dml_content: str) -> List[str]:
        """
        Parse field names from DML content

        Args:
            dml_content: DML definition string

        Returns:
            List of field names
        """
        fields = []

        # Pattern for field definitions: field_name type(size)
        pattern = r'(\w+)\s+\w+\([^)]+\)'
        matches = re.findall(pattern, dml_content)

        fields.extend(matches)

        return fields

    def _extract_io(self, component, script_logic: ScriptLogic):
        """
        Extract input/output file paths

        Args:
            component: Component object
            script_logic: ScriptLogic to update
        """
        params = component.parameters
        comp_type = component.component_type.value.lower()

        # Input files
        if 'input' in comp_type:
            file_path = params.get('file', params.get('path', params.get('filename', '')))
            if file_path:
                script_logic.input_files.append(file_path)

        # Output files
        elif 'output' in comp_type:
            file_path = params.get('file', params.get('path', params.get('filename', '')))
            if file_path:
                script_logic.output_files.append(file_path)

    def _generate_ascii_flow(self, actions: List[ActionNode], edges: List[FlowEdge]) -> str:
        """
        Generate ASCII flow diagram

        Args:
            actions: List of ActionNode objects
            edges: List of FlowEdge objects

        Returns:
            ASCII diagram string
        """
        lines = ["Graph Flow:", "=" * 60]

        # Create adjacency map
        adj_map = {}
        for edge in edges:
            if edge.from_action not in adj_map:
                adj_map[edge.from_action] = []
            adj_map[edge.from_action].append(edge.to_action)

        # Create ID to action map
        action_map = {a.action_id: a for a in actions}

        # Find root nodes (no incoming edges)
        targets = {e.to_action for e in edges}
        roots = [a for a in actions if a.action_id not in targets]

        # Build flow recursively
        visited = set()

        def build_flow(action_id: str, indent: int = 0):
            if action_id in visited:
                return
            visited.add(action_id)

            action = action_map.get(action_id)
            if not action:
                return

            prefix = "  " * indent
            lines.append(f"{prefix}[{action.action_name}] ({action.action_type.value})")

            # Recurse to children
            children = adj_map.get(action_id, [])
            for child_id in children:
                lines.append(f"{prefix}  |")
                lines.append(f"{prefix}  v")
                build_flow(child_id, indent)

        for root in roots:
            build_flow(root.action_id)

        return "\n".join(lines)

    def _generate_mermaid_flow(self, actions: List[ActionNode], edges: List[FlowEdge]) -> str:
        """
        Generate Mermaid flow diagram

        Args:
            actions: List of ActionNode objects
            edges: List of FlowEdge objects

        Returns:
            Mermaid diagram string
        """
        lines = ["graph TD"]

        # Add nodes
        for action in actions:
            node_id = action.action_id.replace("-", "_")
            node_label = f"{action.action_name}<br/>[{action.action_type.value}]"
            lines.append(f"    {node_id}[{node_label}]")

        # Add edges
        for edge in edges:
            source = edge.from_action.replace("-", "_")
            target = edge.to_action.replace("-", "_")
            lines.append(f"    {source} --> {target}")

        return "\n".join(lines)
