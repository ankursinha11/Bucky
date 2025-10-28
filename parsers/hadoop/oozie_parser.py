"""
Hadoop Oozie Parser - PRODUCTION VERSION
========================================
Features:
- Sub-workflow resolution (follows nested workflows recursively)
- Variable resolution (${table}, ${dataset}, etc.)
- Property extraction and tracking
- Actual table name extraction from properties
- Coordinator parsing
- Complete action analysis
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from loguru import logger


class OozieParser:
    """Production Oozie workflow and coordinator parser"""

    def __init__(self):
        """Initialize parser"""
        self.namespaces = {
            '': 'uri:oozie:workflow:0.5',
            'workflow': 'uri:oozie:workflow:0.5',
            'coord': 'uri:oozie:coordinator:0.5',
        }

        # Cache for parsed workflows to avoid re-parsing
        self.parsed_workflows: Dict[str, Dict] = {}

        # Global properties from job.properties files
        self.global_properties: Dict[str, str] = {}

    def parse_workflow(self, workflow_path: str, base_path: str = None) -> Dict[str, Any]:
        """
        Parse Oozie workflow.xml with full sub-workflow resolution

        Args:
            workflow_path: Path to workflow.xml file
            base_path: Base repository path for resolving relative paths

        Returns:
            Dictionary with workflow data including sub-workflows
        """
        logger.info(f"Parsing workflow: {workflow_path}")

        # Check cache
        if workflow_path in self.parsed_workflows:
            logger.debug(f"Using cached workflow: {workflow_path}")
            return self.parsed_workflows[workflow_path]

        # Set base path
        if base_path is None:
            base_path = str(Path(workflow_path).parent.parent)

        # Load global properties if exists
        self._load_global_properties(workflow_path)

        # Parse XML
        try:
            tree = ET.parse(workflow_path)
            root = tree.getroot()
        except Exception as e:
            logger.error(f"Failed to parse workflow XML {workflow_path}: {e}")
            return {}

        # Extract workflow name
        workflow_name = root.get('name', Path(workflow_path).parent.name)

        # Parse workflow data
        workflow_data = {
            'name': workflow_name,
            'file_path': workflow_path,
            'actions': [],
            'sub_workflows': [],
            'input_sources': [],
            'output_targets': [],
            'parameters': {},
            'global_properties': self.global_properties.copy(),
        }

        # Extract parameters
        parameters_elem = root.find('.//{*}parameters')
        if parameters_elem is not None:
            for param in parameters_elem.findall('.//{*}property'):
                name = param.find('.//{*}name')
                value = param.find('.//{*}value')
                if name is not None and name.text:
                    workflow_data['parameters'][name.text] = value.text if value is not None else ''

        # Parse all actions
        for action in root.findall('.//{*}action'):
            action_data = self._parse_action(action, workflow_path, base_path)
            if action_data:
                workflow_data['actions'].append(action_data)

                # Collect I/O sources
                workflow_data['input_sources'].extend(action_data.get('input_paths', []))
                workflow_data['output_targets'].extend(action_data.get('output_paths', []))

        # Extract sub-workflows and follow them
        sub_workflows = self._extract_sub_workflows(root, workflow_path, base_path)
        workflow_data['sub_workflows'] = sub_workflows

        # Get actual I/O from sub-workflows
        sub_workflow_io = self.extract_actual_io_from_sub_workflows(workflow_data)
        workflow_data['input_sources'].extend(sub_workflow_io['input_sources'])
        workflow_data['output_targets'].extend(sub_workflow_io['output_targets'])

        # Deduplicate
        workflow_data['input_sources'] = list(set(workflow_data['input_sources']))
        workflow_data['output_targets'] = list(set(workflow_data['output_targets']))

        # Cache result
        self.parsed_workflows[workflow_path] = workflow_data

        logger.info(f"✓ Parsed workflow '{workflow_name}': {len(workflow_data['actions'])} actions, "
                   f"{len(workflow_data['sub_workflows'])} sub-workflows, "
                   f"{len(workflow_data['input_sources'])} inputs, {len(workflow_data['output_targets'])} outputs")

        return workflow_data

    def _load_global_properties(self, workflow_path: str):
        """Load global properties from job.properties file"""
        workflow_dir = Path(workflow_path).parent

        # Look for job.properties in workflow directory and parent directories
        search_dirs = [workflow_dir, workflow_dir.parent, workflow_dir.parent.parent]

        for search_dir in search_dirs:
            props_file = search_dir / 'job.properties'
            if props_file.exists():
                logger.debug(f"Loading properties from: {props_file}")
                try:
                    with open(props_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                self.global_properties[key.strip()] = value.strip()
                except Exception as e:
                    logger.warning(f"Failed to load properties from {props_file}: {e}")
                break

    def _parse_action(self, action_elem: ET.Element, workflow_path: str, base_path: str) -> Dict[str, Any]:
        """Parse a single action element"""
        action_name = action_elem.get('name', 'unknown')

        action_data = {
            'name': action_name,
            'type': 'unknown',
            'input_paths': [],
            'output_paths': [],
            'parameters': {},
            'script_path': '',
        }

        # Determine action type and extract details
        for action_type in ['spark', 'hive', 'pig', 'sqoop', 'shell', 'java', 'map-reduce', 'sub-workflow']:
            type_elem = action_elem.find(f'.//{{{self.namespaces["workflow"]}}}{action_type}')
            if type_elem is None:
                type_elem = action_elem.find(f'.//{action_type}')

            if type_elem is not None:
                action_data['type'] = action_type

                # Extract type-specific details
                if action_type == 'spark':
                    self._extract_spark_details(type_elem, action_data)
                elif action_type == 'hive':
                    self._extract_hive_details(type_elem, action_data)
                elif action_type == 'pig':
                    self._extract_pig_details(type_elem, action_data)
                elif action_type == 'sqoop':
                    self._extract_sqoop_details(type_elem, action_data)
                elif action_type == 'shell':
                    self._extract_shell_details(type_elem, action_data)
                elif action_type == 'sub-workflow':
                    # Sub-workflows handled separately
                    pass

                break

        # Resolve variables in paths
        action_data['input_paths'] = [self.resolve_variables(p) for p in action_data['input_paths']]
        action_data['output_paths'] = [self.resolve_variables(p) for p in action_data['output_paths']]

        return action_data

    def _extract_spark_details(self, spark_elem: ET.Element, action_data: Dict):
        """Extract Spark action details"""
        # Master
        master = spark_elem.find('.//{*}master')
        if master is not None and master.text:
            action_data['parameters']['master'] = master.text

        # Jar
        jar = spark_elem.find('.//{*}jar')
        if jar is not None and jar.text:
            action_data['script_path'] = jar.text

        # Main class
        main_class = spark_elem.find('.//{*}class')
        if main_class is not None and main_class.text:
            action_data['parameters']['main_class'] = main_class.text

        # Arguments - extract table names
        for arg in spark_elem.findall('.//{*}arg'):
            if arg.text:
                arg_text = arg.text.strip()
                # Look for table names (common patterns)
                if any(keyword in arg_text.lower() for keyword in ['table', 'input', 'source']):
                    action_data['input_paths'].append(arg_text)
                elif any(keyword in arg_text.lower() for keyword in ['output', 'target', 'dest']):
                    action_data['output_paths'].append(arg_text)

    def _extract_hive_details(self, hive_elem: ET.Element, action_data: Dict):
        """Extract Hive action details"""
        # Script
        script = hive_elem.find('.//{*}script')
        if script is not None and script.text:
            action_data['script_path'] = script.text

        # Query
        query = hive_elem.find('.//{*}query')
        if query is not None and query.text:
            # Extract table names from query
            tables = self._extract_tables_from_sql(query.text)
            action_data['input_paths'].extend(tables)

    def _extract_pig_details(self, pig_elem: ET.Element, action_data: Dict):
        """Extract Pig action details"""
        # Script
        script = pig_elem.find('.//{*}script')
        if script is not None and script.text:
            action_data['script_path'] = script.text

        # Parameters
        for param in pig_elem.findall('.//{*}param'):
            if param.text and '=' in param.text:
                key, value = param.text.split('=', 1)
                action_data['parameters'][key.strip()] = value.strip()

                # Check if parameter is a table name
                if any(keyword in key.lower() for keyword in ['table', 'input', 'source', 'output', 'target']):
                    if 'input' in key.lower() or 'source' in key.lower():
                        action_data['input_paths'].append(value.strip())
                    else:
                        action_data['output_paths'].append(value.strip())

    def _extract_sqoop_details(self, sqoop_elem: ET.Element, action_data: Dict):
        """Extract Sqoop action details"""
        # Command
        command = sqoop_elem.find('.//{*}command')
        if command is not None and command.text:
            # Extract table from Sqoop command
            if '--table' in command.text:
                match = re.search(r'--table\s+(\S+)', command.text)
                if match:
                    action_data['input_paths'].append(match.group(1))

            # Extract target directory
            if '--target-dir' in command.text:
                match = re.search(r'--target-dir\s+(\S+)', command.text)
                if match:
                    action_data['output_paths'].append(match.group(1))

    def _extract_shell_details(self, shell_elem: ET.Element, action_data: Dict):
        """Extract Shell action details"""
        # Exec (script name)
        exec_elem = shell_elem.find('.//{*}exec')
        if exec_elem is not None and exec_elem.text:
            action_data['script_path'] = exec_elem.text

        # Arguments
        for arg in shell_elem.findall('.//{*}argument'):
            if arg.text:
                arg_text = arg.text.strip()
                # Try to identify table names
                if any(keyword in arg_text.lower() for keyword in ['table', 'input', 'source']):
                    action_data['input_paths'].append(arg_text)
                elif any(keyword in arg_text.lower() for keyword in ['output', 'target']):
                    action_data['output_paths'].append(arg_text)

    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query"""
        tables = []

        # Simple regex patterns for common SQL patterns
        patterns = [
            r'FROM\s+([a-zA-Z0-9_]+)',
            r'JOIN\s+([a-zA-Z0-9_]+)',
            r'INTO\s+TABLE\s+([a-zA-Z0-9_]+)',
            r'INTO\s+([a-zA-Z0-9_]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            tables.extend(matches)

        return list(set(tables))

    def _extract_sub_workflows(self, root: ET.Element, workflow_path: str, base_path: str) -> List[Dict[str, Any]]:
        """
        Extract and recursively parse sub-workflows

        Returns:
            List of sub-workflow data dictionaries
        """
        sub_workflows = []

        for action in root.findall('.//{*}action'):
            sub_workflow_elem = action.find('.//{*}sub-workflow')
            if sub_workflow_elem is not None:
                sub_workflow_info = self._extract_sub_workflow_info(sub_workflow_elem, base_path)

                if sub_workflow_info['app_path']:
                    # Try to resolve and parse the sub-workflow
                    sub_workflow_path = self._resolve_sub_workflow_path(
                        sub_workflow_info['resolved_app_path'],
                        workflow_path,
                        base_path
                    )

                    if sub_workflow_path and os.path.exists(sub_workflow_path):
                        logger.debug(f"Following sub-workflow: {sub_workflow_path}")

                        # Recursively parse sub-workflow
                        sub_workflow_data = self.parse_workflow(sub_workflow_path, base_path)
                        sub_workflow_info['parsed_data'] = sub_workflow_data
                    else:
                        logger.warning(f"Sub-workflow not found: {sub_workflow_info['app_path']}")

                    sub_workflow_info['passed_properties'] = sub_workflow_info['resolved_properties']
                    sub_workflows.append(sub_workflow_info)

        return sub_workflows

    def _extract_sub_workflow_info(self, sub_workflow_elem: ET.Element, base_path: str) -> Dict[str, Any]:
        """Extract sub-workflow information including path and passed properties"""
        info = {
            'app_path': None,
            'properties': {},
            'resolved_properties': {},
        }

        # Extract app-path
        app_path_elem = sub_workflow_elem.find('.//{*}app-path')
        if app_path_elem is not None and app_path_elem.text:
            app_path = app_path_elem.text.strip()
            info['app_path'] = app_path
            info['resolved_app_path'] = self.resolve_variables(app_path)

        # Extract configuration properties passed to sub-workflow
        config_elem = sub_workflow_elem.find('.//{*}configuration')
        if config_elem is not None:
            for property_elem in config_elem.findall('.//{*}property'):
                name_elem = property_elem.find('.//{*}name')
                value_elem = property_elem.find('.//{*}value')

                if name_elem is not None and value_elem is not None:
                    prop_name = name_elem.text.strip() if name_elem.text else ''
                    prop_value = value_elem.text.strip() if value_elem.text else ''

                    info['properties'][prop_name] = prop_value
                    info['resolved_properties'][prop_name] = self.resolve_variables(prop_value)

        return info

    def _resolve_sub_workflow_path(self, app_path: str, parent_workflow_path: str, base_path: str) -> Optional[str]:
        """Resolve sub-workflow path to actual workflow.xml file"""
        if not app_path:
            return None

        # Try different resolution strategies
        candidates = [
            # Absolute path
            os.path.join(app_path, 'workflow.xml'),
            # Relative to base path
            os.path.join(base_path, app_path.lstrip('/'), 'workflow.xml'),
            # Relative to parent workflow
            os.path.join(Path(parent_workflow_path).parent.parent, app_path.lstrip('/'), 'workflow.xml'),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        return None

    def resolve_variables(self, text: str) -> str:
        """
        Resolve Oozie variables like ${table}, ${wf:user()}, etc.

        Args:
            text: Text containing variables

        Returns:
            Text with variables resolved
        """
        if not text or '${' not in text:
            return text

        resolved = text
        var_pattern = r'\$\{([^}]+)\}'

        for match in re.finditer(var_pattern, text):
            var_name = match.group(1)

            # Skip Oozie functions like wf:user(), wf:id(), etc.
            if ':' in var_name or '(' in var_name:
                resolved = resolved.replace(f'${{{var_name}}}', f'<{var_name}>')
                continue

            # Try to resolve from global properties
            if var_name in self.global_properties:
                resolved = resolved.replace(f'${{{var_name}}}', self.global_properties[var_name])

        return resolved

    def extract_actual_io_from_sub_workflows(self, workflow: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract actual input/output sources from sub-workflows

        This looks at properties passed to sub-workflows and extracts actual table names

        Args:
            workflow: Workflow data dictionary

        Returns:
            Dict with 'input_sources' and 'output_targets' lists
        """
        inputs = []
        outputs = []

        for sub_workflow in workflow.get('sub_workflows', []):
            passed_props = sub_workflow.get('passed_properties', {})

            # Common patterns for table/dataset names
            table_keys = ['table', 'tableName', 'dataset', 'datasetName', 'source', 'target',
                         'inputTable', 'outputTable', 'sourceTable', 'targetTable']

            for key in table_keys:
                if key in passed_props:
                    value = passed_props[key]
                    if value and not value.startswith('${'):
                        # This is an actual resolved table name!
                        if 'input' in key.lower() or 'source' in key.lower():
                            inputs.append(value)
                        elif 'output' in key.lower() or 'target' in key.lower():
                            outputs.append(value)
                        else:
                            # Default to input if unclear
                            inputs.append(value)

            # Also check parsed sub-workflow data
            if 'parsed_data' in sub_workflow:
                sub_data = sub_workflow['parsed_data']
                inputs.extend(sub_data.get('input_sources', []))
                outputs.extend(sub_data.get('output_targets', []))

        return {
            'input_sources': list(set(inputs)),
            'output_targets': list(set(outputs)),
        }
