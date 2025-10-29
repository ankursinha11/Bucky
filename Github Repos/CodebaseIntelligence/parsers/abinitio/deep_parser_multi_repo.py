"""
Ab Initio Multi-Repository Deep Parser
=======================================
Handles Ab Initio codebases with multiple project folders

Structure Example:
    abinitio_root/
    ├── blade/          ← Project 1
    │   └── mp/
    ├── edi/            ← Project 2
    │   └── mp/
    ├── pub_escan/      ← Project 3
    │   └── mp/
    ├── __blade/        ← Backup (skipped)
    └── __edi/          ← Backup (skipped)

Creates separate Repository objects for each project folder.
"""

import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from loguru import logger

from core.models import Repository
from .deep_parser import DeepAbInitioParser


class DeepAbInitioParserMultiRepo(DeepAbInitioParser):
    """
    Deep parser for Ab Initio with multi-project detection

    Automatically detects multiple project folders and creates
    separate repositories for each.

    Inherits from DeepAbInitioParser for backward compatibility.
    """

    def __init__(self, use_ai: bool = True):
        """
        Initialize multi-repo parser

        Args:
            use_ai: Whether to use AI analysis (default: True)
        """
        super().__init__(use_ai=use_ai)
        self.skip_prefixes = ['__', '.', 'tmp', 'temp', 'backup', 'archive']
        self.project_indicators = ['mp', 'dml', 'components', 'pset', 'plan']

    def parse_directory(self, abinitio_path: str) -> Dict[str, Any]:
        """
        Parse Ab Initio directory with multi-project detection

        Args:
            abinitio_path: Path to Ab Initio root directory

        Returns:
            Dict with repositories (list), workflow_flows, script_logics
        """
        logger.info(f"🔍 Detecting Ab Initio project structure: {abinitio_path}")

        # First, parse with base parser to get all processes/components
        base_result = self.base_parser.parse_directory(abinitio_path)
        processes = base_result.get("processes", [])
        components = base_result.get("components", [])
        raw_mp_data = base_result.get("raw_mp_data", [])

        logger.info(f"Base parsing: {len(processes)} graphs, {len(components)} components")

        # Try to detect multiple projects
        project_groups = self._group_by_projects(processes, components, abinitio_path)

        if len(project_groups) > 1:
            logger.info(f"✓ Detected {len(project_groups)} Ab Initio projects!")
            for project_name in project_groups.keys():
                logger.info(f"   - {project_name}")

            return self._parse_multiple_projects(project_groups, abinitio_path, raw_mp_data)
        else:
            # Single project - use parent class behavior
            logger.info("Single Ab Initio project detected - using standard deep parsing")
            return super().parse_directory(abinitio_path)

    def _group_by_projects(
        self,
        processes: List,
        components: List,
        base_path: str
    ) -> Dict[str, Dict]:
        """
        Group processes and components by project folder

        Detects project folders like:
            blade/, edi/, pub_escan/

        Skips backup folders like:
            __blade/, __edi/, .backup/

        Args:
            processes: List of Process objects
            components: List of Component objects
            base_path: Base directory path

        Returns:
            Dict mapping project_name -> {processes, components, path}
        """
        # Use list for components, NOT set!
        project_groups = defaultdict(lambda: {"processes": [], "components": [], "path": None})

        # Detect project folders
        detected_projects = self._detect_project_folders(base_path)

        if not detected_projects:
            # No clear project structure - treat as single project
            project_name = Path(base_path).name
            project_groups[project_name] = {
                "processes": list(processes),
                "components": list(components),
                "path": base_path
            }
            return dict(project_groups)

        # Group processes by project
        for process in processes:
            file_path = process.file_path
            project_name = self._identify_project_from_path(file_path, detected_projects, base_path)

            if project_name:
                # Use append for list
                project_groups[project_name]["processes"].append(process)
                if not project_groups[project_name]["path"]:
                    # Extract project directory path
                    project_groups[project_name]["path"] = self._extract_project_path(
                        file_path, project_name, base_path
                    )

        # Group components by project (match with process)
        # Build a process_id to project mapping for faster lookup
        process_to_project = {}
        for project_name, group in project_groups.items():
            for process in group["processes"]:
                process_to_project[process.id] = project_name

        # Now assign components to projects
        for component in components:
            process_id = component.process_id
            if process_id in process_to_project:
                project_name = process_to_project[process_id]
                # Use append for list, NOT add!
                project_groups[project_name]["components"].append(component)

        # Remove empty projects
        project_groups = {
            name: data for name, data in project_groups.items()
            if data["processes"] or data["components"]
        }

        return dict(project_groups)

    def _detect_project_folders(self, base_path: str) -> List[str]:
        """
        Detect project folders in base path

        A folder is considered a project if it:
        1. Contains typical Ab Initio structure (mp/, dml/, components/, etc.)
        2. Doesn't start with skip prefixes (__, ., tmp, etc.)

        Args:
            base_path: Base directory path

        Returns:
            List of project folder names
        """
        projects = []

        try:
            for entry in os.listdir(base_path):
                full_path = os.path.join(base_path, entry)

                # Must be a directory
                if not os.path.isdir(full_path):
                    continue

                # Skip backup/hidden folders
                if self._should_skip_folder(entry):
                    logger.debug(f"Skipping folder: {entry} (backup/hidden)")
                    continue

                # Check if it looks like an Ab Initio project
                if self._is_abinitio_project(full_path):
                    projects.append(entry)
                    logger.info(f"   Detected project: {entry}")

        except Exception as e:
            logger.warning(f"Error detecting projects: {e}")

        return sorted(projects)

    def _should_skip_folder(self, folder_name: str) -> bool:
        """
        Check if folder should be skipped

        Skips folders starting with:
        - __ (double underscore) - backups
        - . (dot) - hidden
        - tmp, temp, backup, archive

        Args:
            folder_name: Folder name

        Returns:
            True if should skip
        """
        for prefix in self.skip_prefixes:
            if folder_name.startswith(prefix):
                return True
        return False

    def _is_abinitio_project(self, folder_path: str) -> bool:
        """
        Check if folder is an Ab Initio project

        A folder is a project if it contains typical Ab Initio directories:
        - mp/ (graphs)
        - dml/ (data manipulation language)
        - components/
        - pset/ (parameter sets)
        - plan/

        Args:
            folder_path: Path to folder

        Returns:
            True if it's an Ab Initio project
        """
        try:
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

            # Check for Ab Initio indicators
            indicator_count = sum(1 for indicator in self.project_indicators if indicator in subfolders)

            # If has 2+ indicators, it's a project
            return indicator_count >= 2

        except Exception:
            return False

    def _identify_project_from_path(
        self,
        file_path: str,
        projects: List[str],
        base_path: str
    ) -> str:
        """
        Identify which project a file belongs to based on its path

        Args:
            file_path: Full file path
            projects: List of detected project names
            base_path: Base directory path

        Returns:
            Project name or None
        """
        # Normalize paths
        file_path = os.path.normpath(file_path)
        base_path = os.path.normpath(base_path)

        # Get relative path
        try:
            rel_path = os.path.relpath(file_path, base_path)
        except ValueError:
            return None

        # Split path and get first component
        path_parts = rel_path.split(os.sep)

        if path_parts and path_parts[0] in projects:
            return path_parts[0]

        return None

    def _extract_project_path(self, file_path: str, project_name: str, base_path: str) -> str:
        """
        Extract project directory path

        Args:
            file_path: File path
            project_name: Project name
            base_path: Base directory path

        Returns:
            Project directory path
        """
        return os.path.join(base_path, project_name)

    def _parse_multiple_projects(
        self,
        project_groups: Dict[str, Dict],
        base_path: str,
        raw_mp_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        Parse multiple Ab Initio projects

        Creates separate repositories for each project.

        Args:
            project_groups: Dict of project_name -> {processes, components, path}
            base_path: Base directory path
            raw_mp_data: Raw MP data from base parser

        Returns:
            Dict with repositories, workflow_flows, script_logics
        """
        logger.info(f"Creating {len(project_groups)} separate Ab Initio repositories...")

        all_repositories = []
        all_workflow_flows = []
        all_script_logics = []

        for project_name, group in project_groups.items():
            processes = group["processes"]
            components = group["components"]
            project_path = group["path"]

            logger.info(f"\n📂 Processing project: {project_name}")
            logger.info(f"   Path: {project_path}")
            logger.info(f"   Graphs: {len(processes)}")
            logger.info(f"   Components: {len(components)}")

            # Create Tier 1 - Repository
            repository = self._create_repository(
                project_path,
                processes,
                components
            )
            all_repositories.append(repository)

            # Create Tier 2 - WorkflowFlows
            for i, process in enumerate(processes):
                # Find matching raw MP data
                mp_data = {}
                for data in raw_mp_data:
                    if data.get("graph_name") == process.name or data.get("file_path") == process.file_path:
                        mp_data = data
                        break

                workflow_flow = self._create_workflow_flow(process, mp_data)
                all_workflow_flows.append(workflow_flow)

            # Create Tier 3 - ScriptLogics
            for process in processes:
                process_components = [c for c in components if c.process_id == process.id]

                for component in process_components:
                    script_logic = self._create_script_logic(
                        component,
                        process,
                        project_path
                    )
                    if script_logic:
                        all_script_logics.append(script_logic)

            logger.info(f"   ✓ Created repository: {repository.name}")

        logger.info(f"\n✅ Multi-project parsing complete!")
        logger.info(f"   Total repositories: {len(all_repositories)}")
        logger.info(f"   Total workflows: {len(all_workflow_flows)}")
        logger.info(f"   Total scripts: {len(all_script_logics)}")

        # AI Analysis (if enabled)
        if self.use_ai and self.ai_analyzer:
            logger.info("\n🤖 Running AI analysis on all projects...")

            for repository in all_repositories:
                try:
                    repo_analysis = self.ai_analyzer.analyze_repository(repository)
                    repository.ai_summary = repo_analysis.get("business_purpose", "")
                    repository.ai_architecture = repo_analysis.get("architecture_summary", "")
                except Exception as e:
                    logger.warning(f"Repository AI analysis failed for {repository.name}: {e}")

            for workflow in all_workflow_flows:
                try:
                    workflow.ai_flow_summary = self.ai_analyzer.analyze_workflow_flow(workflow)
                except Exception as e:
                    logger.warning(f"Workflow AI analysis failed: {e}")

            for script_logic in all_script_logics:
                try:
                    self.ai_analyzer.analyze_script(script_logic)
                except Exception as e:
                    logger.warning(f"Script AI analysis failed: {e}")

        return {
            "repositories": all_repositories,  # Multiple repositories
            "repository": all_repositories[0] if all_repositories else None,  # Backward compatibility
            "workflow_flows": all_workflow_flows,
            "script_logics": all_script_logics,
            "summary": {
                "total_projects": len(all_repositories),
                "total_graphs": sum(repo.total_workflows for repo in all_repositories),
                "total_components": sum(repo.total_components for repo in all_repositories),
                "total_transformations": sum(len(sl.transformations) for sl in all_script_logics),
                "source_path": base_path,
            }
        }
