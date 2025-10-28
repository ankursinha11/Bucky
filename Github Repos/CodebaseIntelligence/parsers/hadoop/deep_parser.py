"""
Deep Hadoop Parser - Integrates all script parsers and creates 3-tier structure
This is the MASTER parser that orchestrates everything!
"""

import os
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger

from core.models import Repository, RepositoryType, WorkflowFlow, ActionNode, FlowEdge, ActionType
from core.models.script_logic import ScriptLogic
from parsers.hadoop.parser import HadoopParser
from parsers.hadoop.pig_parser import PigScriptParser
from parsers.hadoop.spark_parser import SparkScriptParser
from services.ai_script_analyzer import AIScriptAnalyzer


class DeepHadoopParser:
    """
    Enhanced Hadoop parser with 3-tier deep parsing

    Tier 1: Repository level
    Tier 2: Workflow flow level
    Tier 3: Script logic level (with AI analysis!)
    """

    def __init__(self, use_ai: bool = True):
        """
        Initialize deep parser

        Args:
            use_ai: Whether to use AI for script analysis (default True)
        """
        self.base_parser = HadoopParser()
        self.pig_parser = PigScriptParser()
        self.spark_parser = SparkScriptParser()
        self.ai_analyzer = AIScriptAnalyzer() if use_ai else None

        logger.info("🚀 Deep Hadoop Parser initialized (3-tier + AI analysis)")

    def parse_directory(self, hadoop_path: str) -> Dict[str, Any]:
        """
        Parse Hadoop directory with DEEP analysis

        Returns all 3 tiers:
        - Repository (Tier 1)
        - Workflows with flows (Tier 2)
        - Scripts with deep logic (Tier 3)
        """
        logger.info(f"🔍 Deep parsing Hadoop repository: {hadoop_path}")

        # Parse with base parser first
        base_result = self.base_parser.parse_directory(hadoop_path)
        processes = base_result["processes"]
        components = base_result["components"]

        # Create Tier 1: Repository
        repository = self._create_repository(hadoop_path, processes, components)

        # Create Tier 2: Workflow Flows
        workflow_flows = self._create_workflow_flows(processes, components, hadoop_path, repository.id)

        # Create Tier 3: Script Logic (DEEP!)
        script_logics = self._parse_all_scripts(workflow_flows, hadoop_path)

        # AI Analysis (if enabled)
        if self.ai_analyzer and self.ai_analyzer.enabled:
            logger.info("🤖 Running AI analysis on scripts...")
            for script_logic in script_logics:
                self.ai_analyzer.analyze_script(script_logic)

            logger.info("🤖 Running AI analysis on workflows...")
            for workflow in workflow_flows:
                ai_summary = self.ai_analyzer.analyze_workflow_flow(workflow)
                if ai_summary:
                    workflow.ai_flow_summary = ai_summary

            logger.info("🤖 Running AI analysis on repository...")
            ai_repo_analysis = self.ai_analyzer.analyze_repository(repository)
            if ai_repo_analysis:
                repository.ai_summary = ai_repo_analysis.get("business_value")
                repository.ai_architecture = ai_repo_analysis.get("architecture_summary")

        logger.info(f"✅ Deep parsing complete!")
        logger.info(f"   Repository: {repository.name}")
        logger.info(f"   Workflows: {len(workflow_flows)}")
        logger.info(f"   Scripts analyzed: {len(script_logics)}")
        logger.info(f"   Total transformations: {sum(len(s.transformations) for s in script_logics)}")

        return {
            "repository": repository,
            "workflow_flows": workflow_flows,
            "script_logics": script_logics,
            "processes": processes,  # Keep for compatibility
            "components": components,  # Keep for compatibility
            "summary": {
                "repository_name": repository.name,
                "total_workflows": len(workflow_flows),
                "total_scripts": len(script_logics),
                "total_transformations": sum(len(s.transformations) for s in script_logics),
                "ai_enabled": self.ai_analyzer.enabled if self.ai_analyzer else False,
            }
        }

    def _create_repository(self, base_path: str, processes: List, components: List) -> Repository:
        """Create Tier 1: Repository"""

        repo_name = Path(base_path).name
        repo_id = f"repo_{repo_name}"

        # Count script types
        pig_scripts = sum(1 for c in components if c.component_type.value == "Pig_Script")
        spark_scripts = sum(1 for c in components if c.component_type.value == "Spark_Job")
        hive_scripts = sum(1 for c in components if c.component_type.value == "Hive_Query")
        shell_scripts = sum(1 for c in components if c.component_type.value == "Shell_Script")

        # Extract data sources/targets
        data_sources = []
        data_targets = []
        for comp in components:
            data_sources.extend(comp.input_datasets or [])
            data_targets.extend(comp.output_datasets or [])

        # Identify business domains from workflow names
        business_domains = set()
        for proc in processes:
            name_lower = proc.name.lower()
            if 'cdd' in name_lower:
                business_domains.add("CDD (Coverage Data Discovery)")
            if 'gmrn' in name_lower:
                business_domains.add("GMRN")
            if 'lead' in name_lower:
                business_domains.add("Lead Generation")
            if 'patient' in name_lower:
                business_domains.add("Patient Matching")

        repository = Repository(
            id=repo_id,
            name=repo_name,
            repo_type=RepositoryType.HADOOP,
            base_path=base_path,
            total_workflows=len(processes),
            total_scripts=len(components),
            pig_scripts=pig_scripts,
            spark_scripts=spark_scripts,
            hive_scripts=hive_scripts,
            python_scripts=0,
            shell_scripts=shell_scripts,
            workflow_ids=[p.id for p in processes],
            process_ids=[p.id for p in processes],
            data_sources=list(set(data_sources)),
            data_targets=list(set(data_targets)),
            business_domains=list(business_domains),
            technologies=["Pig", "Spark", "Hive", "Oozie"],
            description=f"Hadoop repository with {len(processes)} workflows and {len(components)} scripts",
        )

        return repository

    def _create_workflow_flows(self, processes: List, components: List, base_path: str, repo_id: str) -> List[WorkflowFlow]:
        """Create Tier 2: Workflow Flows"""

        workflow_flows = []

        for process in processes:
            # Create workflow flow
            workflow_flow = WorkflowFlow(
                workflow_id=process.id,
                workflow_name=process.name,
                workflow_type=process.process_type.value,
                repository_id=repo_id,
                file_path=process.file_path,
                overall_inputs=process.input_sources or [],
                overall_outputs=process.output_targets or [],
                business_domain=self._infer_business_domain(process.name),
            )

            # Create actions from components
            for i, comp_id in enumerate(process.component_ids):
                comp = next((c for c in components if c.id == comp_id), None)
                if not comp:
                    continue

                # Determine action type
                comp_type_str = comp.component_type.value
                if "Pig" in comp_type_str:
                    action_type = ActionType.PIG_SCRIPT
                elif "Spark" in comp_type_str:
                    action_type = ActionType.SPARK_JOB
                elif "Hive" in comp_type_str:
                    action_type = ActionType.HIVE_QUERY
                elif "Shell" in comp_type_str:
                    action_type = ActionType.SHELL_SCRIPT
                elif "Sqoop" in comp_type_str:
                    action_type = ActionType.SQOOP_IMPORT
                else:
                    action_type = ActionType.UNKNOWN

                # Extract script path from component
                script_path = self._find_script_path(comp, base_path)

                action = ActionNode(
                    action_id=comp.id,
                    action_name=comp.name,
                    action_type=action_type,
                    script_path=script_path,
                    input_tables=comp.tables_read or [],
                    output_tables=comp.tables_written or [],
                    input_paths=comp.input_datasets or [],
                    output_paths=comp.output_datasets or [],
                    parameters=comp.parameters or {},
                    execution_order=i + 1,
                    description=comp.description,
                )

                workflow_flow.actions.append(action)

                # Add to entry/exit points
                if i == 0:
                    workflow_flow.entry_actions.append(action.action_id)
                if i == len(process.component_ids) - 1:
                    workflow_flow.exit_actions.append(action.action_id)

            # Create edges (simplified - sequential flow)
            for i in range(len(workflow_flow.actions) - 1):
                edge = FlowEdge(
                    edge_id=f"{workflow_flow.workflow_id}_edge_{i}",
                    from_action=workflow_flow.actions[i].action_id,
                    to_action=workflow_flow.actions[i + 1].action_id,
                    edge_type="data",
                )
                workflow_flow.edges.append(edge)

            # Generate flow diagram
            workflow_flow.flow_diagram_ascii = self._generate_ascii_flow(workflow_flow)
            workflow_flow.flow_diagram_mermaid = self._generate_mermaid_flow(workflow_flow)

            workflow_flows.append(workflow_flow)

        return workflow_flows

    def _parse_all_scripts(self, workflow_flows: List[WorkflowFlow], base_path: str) -> List[ScriptLogic]:
        """Create Tier 3: Parse all scripts deeply"""

        script_logics = []

        for workflow in workflow_flows:
            for action in workflow.actions:
                if not action.script_path:
                    continue

                # Try to find script file
                script_full_path = None
                if os.path.exists(action.script_path):
                    script_full_path = action.script_path
                else:
                    # Try relative to base path
                    potential_path = os.path.join(base_path, action.script_path)
                    if os.path.exists(potential_path):
                        script_full_path = potential_path

                if not script_full_path:
                    logger.warning(f"Script not found: {action.script_path}")
                    continue

                # Parse based on type
                script_logic = None

                if action.action_type == ActionType.PIG_SCRIPT:
                    script_logic = self.pig_parser.parse_pig_script(
                        script_full_path,
                        workflow_id=workflow.workflow_id,
                        action_id=action.action_id
                    )

                elif action.action_type == ActionType.SPARK_JOB:
                    script_logic = self.spark_parser.parse_spark_script(
                        script_full_path,
                        workflow_id=workflow.workflow_id,
                        action_id=action.action_id
                    )

                # elif action.action_type == ActionType.HIVE_QUERY:
                #     script_logic = self.hive_parser.parse_hive_script(...)

                if script_logic:
                    # Link back to action
                    action.script_content_id = script_logic.script_id
                    script_logics.append(script_logic)

        return script_logics

    def _find_script_path(self, component, base_path: str) -> Optional[str]:
        """Find actual script file path"""
        # Look in component parameters for script reference
        params = component.parameters or {}

        # Common parameter names for scripts
        script_param_names = ['script', 'file', 'jar', 'python_file', 'sql_file']

        for param_name in script_param_names:
            if param_name in params:
                return params[param_name]

        # Try to infer from component name
        # e.g., component "es-consumer" might have script "es-consumer.pig"
        potential_scripts = [
            f"{component.name}.pig",
            f"{component.name}.py",
            f"{component.name}.hql",
            f"{component.name}.sql",
        ]

        for script_name in potential_scripts:
            # Look in common script directories
            for subdir in ['scripts', 'pig', 'spark', 'hive', 'sql', '']:
                potential_path = os.path.join(base_path, subdir, script_name)
                if os.path.exists(potential_path):
                    return potential_path

        return None

    def _infer_business_domain(self, workflow_name: str) -> Optional[str]:
        """Infer business domain from workflow name"""
        name_lower = workflow_name.lower()

        if 'cdd' in name_lower:
            return "CDD (Coverage Data Discovery)"
        elif 'gmrn' in name_lower:
            return "GMRN (Group Medical Record Number)"
        elif 'lead' in name_lower:
            return "Lead Generation"
        elif 'patient' in name_lower:
            return "Patient Matching"

        return None

    def _generate_ascii_flow(self, workflow: WorkflowFlow) -> str:
        """Generate ASCII flow diagram"""
        if not workflow.actions:
            return "No actions"

        lines = []
        for i, action in enumerate(workflow.actions[:10]):  # First 10
            if i > 0:
                lines.append("    ↓")
            lines.append(f"[{i+1}] {action.action_name} ({action.action_type.value})")

        if len(workflow.actions) > 10:
            lines.append(f"    ... and {len(workflow.actions) - 10} more actions")

        return "\n".join(lines)

    def _generate_mermaid_flow(self, workflow: WorkflowFlow) -> str:
        """Generate Mermaid.js flow diagram"""
        lines = ["graph TD"]

        for action in workflow.actions[:20]:  # First 20
            safe_id = action.action_id.replace("-", "_").replace(":", "_")
            label = f"{action.action_name}<br/>{action.action_type.value}"
            lines.append(f'    {safe_id}["{label}"]')

        for edge in workflow.edges[:20]:
            safe_from = edge.from_action.replace("-", "_").replace(":", "_")
            safe_to = edge.to_action.replace("-", "_").replace(":", "_")
            lines.append(f"    {safe_from} --> {safe_to}")

        return "\n".join(lines)
