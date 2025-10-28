"""
Deep Databricks Parser - Multi-Repository Support
Handles databricks folder with multiple project/workspace folders
"""

import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from loguru import logger

from core.models import Repository, RepositoryType
from parsers.databricks.deep_parser import DeepDatabricksParser


def _group_processes_by_folder(processes: List, components: List, base_path: str) -> Dict[str, Dict]:
    """
    Group processes and components by top-level folder

    For structure like:
        databricks_notebooks/
            project-analytics/
            project-etl/
            workspace-finance/

    Returns dict mapping folder_name -> {processes, components, path}
    """
    folder_groups = defaultdict(lambda: {"processes": [], "components": set(), "path": None})

    base_path_obj = Path(base_path).resolve()

    for process in processes:
        if not process.file_path:
            continue

        # Get relative path from base
        try:
            file_path_obj = Path(process.file_path).resolve()
            rel_path = file_path_obj.relative_to(base_path_obj)

            # First part is folder (e.g., "project-analytics", "workspace-finance")
            parts = rel_path.parts
            if parts:
                folder_name = parts[0]
                folder_path = base_path_obj / folder_name

                folder_groups[folder_name]["processes"].append(process)
                folder_groups[folder_name]["path"] = str(folder_path)

                # Add components for this process
                for comp_id in process.component_ids:
                    folder_groups[folder_name]["components"].add(comp_id)
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not determine folder for process {process.name}: {e}")
            # Put in "unknown" group
            folder_groups["unknown"]["processes"].append(process)
            folder_groups["unknown"]["path"] = base_path
            for comp_id in process.component_ids:
                folder_groups["unknown"]["components"].add(comp_id)

    # Convert component IDs to actual Component objects
    comp_by_id = {c.id: c for c in components}

    for folder_name in folder_groups:
        comp_ids = folder_groups[folder_name]["components"]
        folder_groups[folder_name]["components"] = [
            comp_by_id[cid] for cid in comp_ids if cid in comp_by_id
        ]

    return dict(folder_groups)


class DeepDatabricksParserMultiRepo(DeepDatabricksParser):
    """
    Enhanced Deep Databricks Parser with multi-repository support

    Detects when databricks_path contains multiple workspace/project folders
    and creates separate Repository objects for each.
    """

    def parse_directory(self, databricks_path: str) -> Dict[str, Any]:
        """
        Parse Databricks directory - automatically detects multiple workspaces/projects

        If databricks_path contains multiple top-level folders, creates separate
        repositories for each. Otherwise falls back to single repository behavior.
        """
        logger.info(f"🔍 Deep parsing Databricks: {databricks_path}")

        # Parse with base parser first
        base_result = self.base_parser.parse_directory(databricks_path)
        processes = base_result["processes"]
        components = base_result["components"]

        logger.info(f"Found {len(processes)} notebooks, {len(components)} components")

        # Try to detect multiple workspaces/projects
        folder_groups = _group_processes_by_folder(processes, components, databricks_path)

        if len(folder_groups) > 1 or (len(folder_groups) == 1 and "unknown" not in folder_groups):
            # Multiple workspaces/projects detected!
            logger.info(f"🎯 Detected {len(folder_groups)} workspaces/projects: {list(folder_groups.keys())}")
            return self._parse_multiple_workspaces(folder_groups, databricks_path, processes, components)
        else:
            # Single workspace - use parent class behavior
            logger.info("📦 Single workspace detected")
            return super().parse_directory(databricks_path)

    def _parse_multiple_workspaces(
        self,
        folder_groups: Dict,
        base_path: str,
        all_processes: List,
        all_components: List
    ) -> Dict[str, Any]:
        """Parse multiple workspaces/projects separately"""

        repositories = []
        all_workflow_flows = []
        all_script_logics = []

        for folder_name in sorted(folder_groups.keys()):
            folder_data = folder_groups[folder_name]
            folder_processes = folder_data["processes"]
            folder_components = folder_data["components"]
            folder_path = folder_data["path"]

            logger.info(f"📦 Processing workspace: {folder_name} ({len(folder_processes)} notebooks, {len(folder_components)} components)")

            # Create repository for this workspace
            repository = self._create_repository(folder_path, folder_processes, folder_components)
            repositories.append(repository)

            # Create workflow flows
            workflow_flows = []
            for process in folder_processes:
                from core.models import WorkflowFlow
                workflow = WorkflowFlow(
                    workflow_id=process.id,
                    workflow_name=process.name,
                    workflow_type="notebook",
                    repository_id=repository.id,
                    file_path=process.file_path,
                )
                workflow_flows.append(workflow)

            all_workflow_flows.extend(workflow_flows)

            # Parse notebook cells
            script_logics = self._parse_notebooks(folder_processes, folder_path)
            all_script_logics.extend(script_logics)

            logger.info(f"  ✓ {folder_name}: {len(workflow_flows)} workflows, {len(script_logics)} scripts")

        logger.info(f"✅ Deep parsing complete!")
        logger.info(f"   Workspaces: {len(repositories)}")
        logger.info(f"   Total notebooks: {len(all_workflow_flows)}")
        logger.info(f"   Total scripts: {len(all_script_logics)}")

        # Return structure compatible with deep_indexer
        return {
            "repository": repositories[0] if repositories else None,
            "repositories": repositories,  # NEW: All repositories
            "workflow_flows": all_workflow_flows,
            "script_logics": all_script_logics,
            "processes": all_processes,
            "components": all_components,
            "summary": {
                "total_workspaces": len(repositories),
                "workspaces": [r.name for r in repositories],
                "total_notebooks": len(all_workflow_flows),
                "total_scripts": len(all_script_logics),
            }
        }

    def _create_repository(self, base_path: str, processes: List, components: List) -> Repository:
        """
        Create repository - uses folder name as repo name
        """
        repo_name = Path(base_path).name
        repo_id = f"repo_{repo_name}"

        # Count script types
        python_scripts = sum(1 for c in components if '.py' in (c.file_path or ''))
        sql_scripts = sum(1 for c in components if '.sql' in (c.file_path or ''))
        notebook_count = sum(1 for c in components if '.ipynb' in (c.file_path or ''))

        # Identify business domains from notebook names
        business_domains = set()
        for proc in processes:
            name_lower = proc.name.lower()
            if 'analytics' in name_lower or 'analysis' in name_lower:
                business_domains.add("Analytics")
            if 'etl' in name_lower or 'pipeline' in name_lower:
                business_domains.add("ETL / Data Pipeline")
            if 'ml' in name_lower or 'model' in name_lower:
                business_domains.add("Machine Learning")
            if 'reporting' in name_lower or 'report' in name_lower:
                business_domains.add("Reporting")
            if 'finance' in name_lower:
                business_domains.add("Finance")
            if 'customer' in name_lower:
                business_domains.add("Customer Analytics")
            if 'transform' in name_lower:
                business_domains.add("Data Transformation")

        repository = Repository(
            id=repo_id,
            name=repo_name,
            repo_type=RepositoryType.DATABRICKS,
            base_path=base_path,
            total_workflows=len(processes),
            total_notebooks=len(processes),
            total_scripts=len(components),
            python_scripts=python_scripts,
            sql_scripts=sql_scripts,
            workflow_ids=[p.id for p in processes],
            business_domains=list(business_domains),
            technologies=["PySpark", "SQL", "Delta Lake", "Databricks"],
            description=f"Databricks workspace '{repo_name}' with {len(processes)} notebooks and {len(components)} scripts",
        )

        logger.info(f"Created Repository: {repo_name} ({len(processes)} notebooks, {len(components)} scripts)")

        return repository
