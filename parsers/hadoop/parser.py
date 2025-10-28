"""
Hadoop Parser - Main Entry Point
Wraps OozieParser for integration with CodebaseIntelligence architecture
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from core.models import Process, Component, ProcessType, ComponentType, SystemType
from .oozie_parser import OozieParser


class HadoopParser:
    """
    Hadoop Parser - Production Version

    Features:
    - Parses Oozie workflows and coordinators
    - Sub-workflow resolution (recursive)
    - Variable resolution
    - Property extraction
    - Actual table name extraction
    """

    def __init__(self):
        """Initialize Hadoop parser"""
        self.oozie_parser = OozieParser()

    def parse_directory(self, hadoop_path: str) -> Dict[str, Any]:
        """
        Parse Hadoop repository containing multiple workflows

        Args:
            hadoop_path: Path to Hadoop repository (can contain multiple apps)

        Returns:
            Dict with 'processes' and 'components' lists
        """
        logger.info(f"Parsing Hadoop repository: {hadoop_path}")

        processes = []
        components = []

        # Find all workflow.xml files recursively
        workflow_files = self._find_workflow_files(hadoop_path)
        logger.info(f"Found {len(workflow_files)} workflow files")

        # Parse each workflow
        for workflow_file in workflow_files:
            try:
                workflow_data = self.oozie_parser.parse_workflow(workflow_file, hadoop_path)

                if workflow_data:
                    # Convert to Process and Component objects
                    process, process_components = self._convert_to_models(workflow_data)

                    if process:
                        processes.append(process)
                        components.extend(process_components)

            except Exception as e:
                logger.error(f"Error parsing {workflow_file}: {e}")
                continue

        logger.info(f"✓ Parsed {len(processes)} processes, {len(components)} components")

        return {
            "processes": processes,
            "components": components,
            "summary": {
                "total_processes": len(processes),
                "total_components": len(components),
                "source_path": hadoop_path,
            }
        }

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a single workflow.xml file

        Args:
            file_path: Path to workflow.xml file

        Returns:
            Dict with 'processes' and 'components'
        """
        logger.info(f"Parsing single workflow: {file_path}")

        try:
            workflow_data = self.oozie_parser.parse_workflow(file_path)

            if workflow_data:
                process, components = self._convert_to_models(workflow_data)

                return {
                    "processes": [process] if process else [],
                    "components": components,
                }
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

        return {"processes": [], "components": []}

    def _find_workflow_files(self, base_path: str) -> List[str]:
        """
        Find all Oozie workflow and coordinator XML files recursively

        Handles various naming patterns:
        - workflow.xml, *_workflow.xml (in oozie folders or workflows dir)
        - coordinator.xml, *_coordinator.xml
        - Any .xml file in 'oozie' directories

        Args:
            base_path: Base directory to search

        Returns:
            List of workflow/coordinator XML file paths
        """
        workflow_files = []

        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.xml'):
                    file_path = os.path.join(root, file)

                    # Include if:
                    # 1. File is named workflow.xml or *_workflow.xml
                    # 2. File is named coordinator.xml or *_coordinator.xml
                    # 3. File is in an 'oozie' directory
                    # 4. File is in a 'coordinators' directory
                    if (file == 'workflow.xml' or
                        file.endswith('_workflow.xml') or
                        file == 'coordinator.xml' or
                        file.endswith('_coordinator.xml') or
                        '/oozie/' in file_path or
                        '\\oozie\\' in file_path or  # Windows path
                        '/coordinators/' in file_path or
                        '\\coordinators\\' in file_path):  # Windows path
                        workflow_files.append(file_path)

        return sorted(workflow_files)

    def _convert_to_models(self, workflow_data: Dict[str, Any]) -> tuple:
        """
        Convert parsed workflow data to Process and Component objects

        Args:
            workflow_data: Parsed workflow data from OozieParser

        Returns:
            Tuple of (Process, List[Component])
        """
        workflow_name = workflow_data.get('name', 'unknown')
        workflow_path = workflow_data.get('file_path', '')
        is_coordinator = workflow_data.get('is_coordinator', False)

        # Generate unique hash from file path to ensure globally unique IDs
        file_hash = hashlib.md5(workflow_path.encode()).hexdigest()[:8]

        # Determine process type and description
        if is_coordinator:
            process_type = ProcessType.OOZIE_COORDINATOR
            description = f"Hadoop Oozie Coordinator: {workflow_name}"
            # For coordinators, extract schedule info
            schedule = workflow_data.get('coordinator_frequency', None)
        else:
            process_type = ProcessType.OOZIE_WORKFLOW
            description = f"Hadoop Oozie Workflow: {workflow_name}"
            schedule = None

        # Create Process object with unique ID
        process = Process(
            id=f"hadoop_{workflow_name}_{file_hash}",
            name=workflow_name,
            system=SystemType.HADOOP,
            process_type=process_type,
            file_path=workflow_path,
            repo_name=Path(workflow_path).parent.parent.name if workflow_path else '',
            description=description,
            schedule=schedule,
            input_sources=workflow_data.get('input_sources', []),
            output_targets=workflow_data.get('output_targets', []),
            parameters=workflow_data.get('parameters', {}),
        )

        # Create Component objects from actions
        components = []
        component_ids = []

        for idx, action in enumerate(workflow_data.get('actions', [])):
            # Include index to ensure uniqueness even if action names repeat
            component_id = f"{process.id}_{action['name']}_{idx}"

            # Map action type to ComponentType
            component_type = self._map_action_type(action['type'])

            component = Component(
                id=component_id,
                name=action['name'],
                component_type=component_type,
                system="hadoop",
                file_path=workflow_path,
                process_id=process.id,
                process_name=process.name,
                input_datasets=action.get('input_paths', []),
                output_datasets=action.get('output_paths', []),
                parameters=action.get('parameters', {}),
                transformation_logic=action.get('script_path', ''),
            )

            components.append(component)
            component_ids.append(component_id)

        # Update process with component info
        process.component_ids = component_ids
        process.component_count = len(components)

        return process, components

    def _map_action_type(self, action_type: str) -> ComponentType:
        """
        Map Oozie action type to ComponentType enum

        Args:
            action_type: Oozie action type string

        Returns:
            ComponentType enum value
        """
        type_mapping = {
            'spark': ComponentType.SPARK_JOB,
            'hive': ComponentType.HIVE_QUERY,
            'pig': ComponentType.PIG_SCRIPT,
            'sqoop': ComponentType.SQOOP_IMPORT,
            'shell': ComponentType.SHELL_SCRIPT,
            'sub-workflow': ComponentType.OOZIE_WORKFLOW,
            'coordinator': ComponentType.OOZIE_COORDINATOR,
            'java': ComponentType.SPARK_JOB,  # Treat Java as Spark
            'map-reduce': ComponentType.SPARK_JOB,
        }

        return type_mapping.get(action_type.lower(), ComponentType.UNKNOWN)
