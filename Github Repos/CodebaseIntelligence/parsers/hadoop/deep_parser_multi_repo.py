"""
Deep Hadoop Parser - Multi-Repository Support
Handles hadoop_repos with multiple app folders (app-cdd, app-globalmrn, etc.)
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from loguru import logger

from core.models import Repository, RepositoryType, WorkflowFlow, ActionNode, FlowEdge, ActionType
from core.models.script_logic import ScriptLogic
from parsers.hadoop.deep_parser import DeepHadoopParser


def _group_processes_by_app(processes: List, components: List, base_path: str) -> Dict[str, Dict]:
    """
    Group processes and components by application folder

    For structure like:
        hadoop_repos/
            app-cdd/
            app-globalmrn/
            app-data-ingestion/

    Returns dict mapping app_name -> {processes, components, path}
    """
    app_groups = defaultdict(lambda: {"processes": [], "components": set(), "path": None})

    base_path_obj = Path(base_path).resolve()

    for process in processes:
        if not process.file_path:
            continue

        # Get relative path from base
        try:
            file_path_obj = Path(process.file_path).resolve()
            rel_path = file_path_obj.relative_to(base_path_obj)

            # First part is app folder (e.g., "app-cdd", "app-globalmrn")
            parts = rel_path.parts
            if parts:
                app_name = parts[0]
                app_path = base_path_obj / app_name

                app_groups[app_name]["processes"].append(process)
                app_groups[app_name]["path"] = str(app_path)

                # Add components for this process
                for comp_id in process.component_ids:
                    app_groups[app_name]["components"].add(comp_id)
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not determine app for process {process.name}: {e}")
            # Put in "unknown" group
            app_groups["unknown"]["processes"].append(process)
            app_groups["unknown"]["path"] = base_path
            for comp_id in process.component_ids:
                app_groups["unknown"]["components"].add(comp_id)

    # Convert component IDs to actual Component objects
    comp_by_id = {c.id: c for c in components}

    for app_name in app_groups:
        comp_ids = app_groups[app_name]["components"]
        app_groups[app_name]["components"] = [
            comp_by_id[cid] for cid in comp_ids if cid in comp_by_id
        ]

    return dict(app_groups)


class DeepHadoopParserMultiRepo(DeepHadoopParser):
    """
    Enhanced Deep Hadoop Parser with multi-repository support

    Detects when hadoop_path contains multiple app folders and creates
    separate Repository objects for each app.
    """

    def parse_directory(self, hadoop_path: str) -> Dict[str, Any]:
        """
        Parse Hadoop directory - automatically detects multiple apps

        If hadoop_path contains app-* folders, creates separate repositories
        for each. Otherwise falls back to single repository behavior.
        """
        logger.info(f"🔍 Deep parsing Hadoop: {hadoop_path}")

        # Parse with base parser first
        base_result = self.base_parser.parse_directory(hadoop_path)
        processes = base_result["processes"]
        components = base_result["components"]

        logger.info(f"Found {len(processes)} processes, {len(components)} components")

        # Try to detect multiple apps
        app_groups = _group_processes_by_app(processes, components, hadoop_path)

        if len(app_groups) > 1 or (len(app_groups) == 1 and "unknown" not in app_groups):
            # Multiple apps detected!
            logger.info(f"🎯 Detected {len(app_groups)} applications: {list(app_groups.keys())}")
            return self._parse_multiple_apps(app_groups, hadoop_path)
        else:
            # Single repo - use parent class behavior
            logger.info("📦 Single repository detected")
            return super().parse_directory(hadoop_path)

    def _parse_multiple_apps(self, app_groups: Dict, base_path: str) -> Dict[str, Any]:
        """Parse multiple applications separately"""

        repositories = []
        all_workflow_flows = []
        all_script_logics = []
        all_processes = []
        all_components = []

        for app_name in sorted(app_groups.keys()):
            app_data = app_groups[app_name]
            app_processes = app_data["processes"]
            app_components = app_data["components"]
            app_path = app_data["path"]

            logger.info(f"📦 Processing app: {app_name} ({len(app_processes)} workflows, {len(app_components)} components)")

            # Create repository for this app
            repository = self._create_repository(app_path, app_processes, app_components, app_name)
            repositories.append(repository)

            # Create workflow flows
            workflow_flows = self._create_workflow_flows(
                app_processes, app_components, app_path, repository.id
            )
            all_workflow_flows.extend(workflow_flows)

            # Parse scripts
            script_logics = self._parse_all_scripts(workflow_flows, app_path)
            all_script_logics.extend(script_logics)

            # Track all for compatibility
            all_processes.extend(app_processes)
            all_components.extend(app_components)

            # AI Analysis (if enabled) - per repository
            if self.ai_analyzer and hasattr(self.ai_analyzer, 'enabled') and self.ai_analyzer.enabled:
                logger.info(f"🤖 Running AI analysis for {app_name}...")

                # Analyze scripts
                for script_logic in script_logics:
                    try:
                        self.ai_analyzer.analyze_script(script_logic)
                    except Exception as e:
                        logger.warning(f"Script AI analysis failed: {e}")

                # Analyze workflows
                for workflow in workflow_flows:
                    try:
                        ai_summary = self.ai_analyzer.analyze_workflow_flow(workflow)
                        if ai_summary:
                            workflow.ai_flow_summary = ai_summary
                    except Exception as e:
                        logger.warning(f"Workflow AI analysis failed: {e}")

                # Analyze repository
                try:
                    ai_repo_analysis = self.ai_analyzer.analyze_repository(repository)
                    if ai_repo_analysis:
                        repository.ai_summary = ai_repo_analysis.get("business_value", "")
                        repository.ai_architecture = ai_repo_analysis.get("architecture_summary", "")
                except Exception as e:
                    logger.warning(f"Repository AI analysis failed: {e}")

        logger.info(f"✅ Deep parsing complete!")
        logger.info(f"   Applications: {len(repositories)}")
        logger.info(f"   Total workflows: {len(all_workflow_flows)}")
        logger.info(f"   Total scripts: {len(all_script_logics)}")
        logger.info(f"   Total transformations: {sum(len(s.transformations) for s in all_script_logics)}")

        # Return structure compatible with deep_indexer
        # Use first repository as "main" but include all in metadata
        return {
            "repository": repositories[0] if repositories else None,
            "repositories": repositories,  # NEW: All repositories
            "workflow_flows": all_workflow_flows,
            "script_logics": all_script_logics,
            "processes": all_processes,
            "components": all_components,
            "summary": {
                "total_applications": len(repositories),
                "applications": [r.name for r in repositories],
                "total_workflows": len(all_workflow_flows),
                "total_scripts": len(all_script_logics),
                "total_transformations": sum(len(s.transformations) for s in all_script_logics),
                "ai_enabled": self.ai_analyzer.enabled if (self.ai_analyzer and hasattr(self.ai_analyzer, 'enabled')) else False,
            }
        }

    def _create_repository(self, base_path: str, processes: List, components: List, app_name: str = None) -> Repository:
        """
        Create repository - enhanced to use app_name if provided
        """
        if not app_name:
            app_name = Path(base_path).name

        repo_id = f"repo_{app_name}"

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

        # Identify business domains
        business_domains = set()
        for proc in processes:
            name_lower = proc.name.lower()
            if 'cdd' in name_lower:
                business_domains.add("CDD (Coverage Data Discovery)")
            if 'gmrn' in name_lower:
                business_domains.add("GMRN (Group Medical Record Number)")
            if 'lead' in name_lower:
                business_domains.add("Lead Generation")
            if 'patient' in name_lower:
                business_domains.add("Patient Matching")
            if 'ops' in name_lower or 'hint' in name_lower:
                business_domains.add("Operations & Hints")
            if 'coverage' in name_lower:
                business_domains.add("Coverage Helper")
            if 'ingest' in name_lower:
                business_domains.add("Data Ingestion")

        repository = Repository(
            id=repo_id,
            name=app_name,
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
            description=f"Hadoop application '{app_name}' with {len(processes)} workflows and {len(components)} scripts",
        )

        logger.info(f"Created Repository: {app_name} ({len(processes)} workflows, {len(components)} scripts)")

        return repository
