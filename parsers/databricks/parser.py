"""
Databricks Parser - PRODUCTION VERSION
======================================
Features:
- Notebook parsing (.py, .sql, .scala, .ipynb)
- SQL query extraction
- PySpark code analysis
- Delta table tracking
- ADF pipeline integration
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from core.models import Process, Component, ProcessType, ComponentType, SystemType


class DatabricksParser:
    """
    Databricks Parser - Production Version

    Parses Databricks notebooks and extracts:
    - SQL queries and table references
    - PySpark DataFrame operations
    - Delta table operations
    - Notebook parameters
    """

    def __init__(self):
        """Initialize Databricks parser"""
        pass

    def parse_directory(self, databricks_path: str) -> Dict[str, Any]:
        """
        Parse Databricks directory containing notebooks

        Args:
            databricks_path: Path to Databricks notebooks directory

        Returns:
            Dict with 'processes' and 'components' lists
        """
        logger.info(f"Parsing Databricks directory: {databricks_path}")

        processes = []
        components = []

        # Find all notebook files
        notebook_files = self._find_notebook_files(databricks_path)
        logger.info(f"Found {len(notebook_files)} notebook files")

        # Parse each notebook
        for notebook_file in notebook_files:
            try:
                notebook_data = self._parse_notebook(notebook_file)

                if notebook_data:
                    # Convert to Process and Component objects
                    process, process_components = self._convert_to_models(notebook_data)

                    if process:
                        processes.append(process)
                        components.extend(process_components)

            except Exception as e:
                logger.error(f"Error parsing {notebook_file}: {e}")
                continue

        logger.info(f"✓ Parsed {len(processes)} processes, {len(components)} components")

        return {
            "processes": processes,
            "components": components,
            "summary": {
                "total_processes": len(processes),
                "total_components": len(components),
                "source_path": databricks_path,
            }
        }

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a single Databricks notebook

        Args:
            file_path: Path to notebook file

        Returns:
            Dict with 'processes' and 'components'
        """
        logger.info(f"Parsing single notebook: {file_path}")

        try:
            notebook_data = self._parse_notebook(file_path)

            if notebook_data:
                process, components = self._convert_to_models(notebook_data)

                return {
                    "processes": [process] if process else [],
                    "components": components,
                }
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

        return {"processes": [], "components": []}

    def _find_notebook_files(self, base_path: str) -> List[str]:
        """
        Find all Databricks notebook files

        Args:
            base_path: Base directory to search

        Returns:
            List of notebook file paths
        """
        notebook_files = []
        extensions = ['.py', '.sql', '.scala', '.ipynb', '.dbc']

        for root, dirs, files in os.walk(base_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    notebook_files.append(os.path.join(root, file))

        return sorted(notebook_files)

    def _parse_notebook(self, notebook_path: str) -> Dict[str, Any]:
        """
        Parse a single Databricks notebook

        Args:
            notebook_path: Path to notebook file

        Returns:
            Dictionary with notebook data
        """
        notebook_name = Path(notebook_path).stem
        file_ext = Path(notebook_path).suffix

        # Read content
        try:
            with open(notebook_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {notebook_path}: {e}")
            return {}

        # Parse based on file type
        if file_ext == '.ipynb':
            return self._parse_jupyter_notebook(notebook_path, content)
        elif file_ext == '.sql':
            return self._parse_sql_notebook(notebook_path, content)
        elif file_ext in ['.py', '.scala']:
            return self._parse_code_notebook(notebook_path, content, file_ext)
        else:
            return self._parse_generic_notebook(notebook_path, content)

    def _parse_jupyter_notebook(self, notebook_path: str, content: str) -> Dict[str, Any]:
        """Parse Jupyter notebook (.ipynb)"""
        try:
            notebook_json = json.loads(content)
            cells = notebook_json.get('cells', [])

            # Extract SQL queries and table references
            input_tables = []
            output_tables = []
            queries = []

            for cell in cells:
                cell_type = cell.get('cell_type', '')
                source = ''.join(cell.get('source', []))

                if cell_type == 'code':
                    # Extract table references
                    tables = self._extract_table_references(source)
                    input_tables.extend(tables['input_tables'])
                    output_tables.extend(tables['output_tables'])

                    # Extract queries
                    cell_queries = self._extract_sql_queries(source)
                    queries.extend(cell_queries)

            return {
                'name': Path(notebook_path).stem,
                'file_path': notebook_path,
                'notebook_type': 'jupyter',
                'input_tables': list(set(input_tables)),
                'output_tables': list(set(output_tables)),
                'queries': queries,
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Jupyter notebook JSON: {e}")
            return self._parse_generic_notebook(notebook_path, content)

    def _parse_sql_notebook(self, notebook_path: str, content: str) -> Dict[str, Any]:
        """Parse SQL notebook"""
        # Extract table references from SQL
        tables = self._extract_table_references(content)

        return {
            'name': Path(notebook_path).stem,
            'file_path': notebook_path,
            'notebook_type': 'sql',
            'input_tables': tables['input_tables'],
            'output_tables': tables['output_tables'],
            'queries': self._extract_sql_queries(content),
        }

    def _parse_code_notebook(self, notebook_path: str, content: str, file_ext: str) -> Dict[str, Any]:
        """Parse Python or Scala notebook"""
        language = 'python' if file_ext == '.py' else 'scala'

        # Extract table references
        tables = self._extract_table_references(content)

        return {
            'name': Path(notebook_path).stem,
            'file_path': notebook_path,
            'notebook_type': language,
            'input_tables': tables['input_tables'],
            'output_tables': tables['output_tables'],
            'queries': self._extract_sql_queries(content),
        }

    def _parse_generic_notebook(self, notebook_path: str, content: str) -> Dict[str, Any]:
        """Parse generic notebook"""
        tables = self._extract_table_references(content)

        return {
            'name': Path(notebook_path).stem,
            'file_path': notebook_path,
            'notebook_type': 'unknown',
            'input_tables': tables['input_tables'],
            'output_tables': tables['output_tables'],
            'queries': [],
        }

    def _extract_table_references(self, code: str) -> Dict[str, List[str]]:
        """
        Extract table references from code

        Args:
            code: Notebook code content

        Returns:
            Dict with 'input_tables' and 'output_tables' lists
        """
        input_tables = []
        output_tables = []

        # SQL patterns
        sql_patterns = {
            'input': [
                r'FROM\s+([a-zA-Z0-9_\.]+)',
                r'JOIN\s+([a-zA-Z0-9_\.]+)',
                r'\.table\(["\']([a-zA-Z0-9_\.]+)["\']\)',
                r'\.read\.table\(["\']([a-zA-Z0-9_\.]+)["\']\)',
            ],
            'output': [
                r'INTO\s+TABLE\s+([a-zA-Z0-9_\.]+)',
                r'INTO\s+([a-zA-Z0-9_\.]+)',
                r'CREATE\s+TABLE\s+([a-zA-Z0-9_\.]+)',
                r'\.saveAsTable\(["\']([a-zA-Z0-9_\.]+)["\']\)',
                r'\.insertInto\(["\']([a-zA-Z0-9_\.]+)["\']\)',
            ]
        }

        # Extract input tables
        for pattern in sql_patterns['input']:
            matches = re.findall(pattern, code, re.IGNORECASE)
            input_tables.extend(matches)

        # Extract output tables
        for pattern in sql_patterns['output']:
            matches = re.findall(pattern, code, re.IGNORECASE)
            output_tables.extend(matches)

        # Delta Lake patterns
        delta_read_pattern = r'delta\.`([^`]+)`'
        delta_matches = re.findall(delta_read_pattern, code)
        input_tables.extend(delta_matches)

        return {
            'input_tables': list(set(input_tables)),
            'output_tables': list(set(output_tables)),
        }

    def _extract_sql_queries(self, code: str) -> List[str]:
        """
        Extract SQL queries from code

        Args:
            code: Notebook code content

        Returns:
            List of SQL query strings
        """
        queries = []

        # Look for SQL in spark.sql() calls
        sql_pattern = r'spark\.sql\(["\']([^"\']+)["\']\)'
        matches = re.findall(sql_pattern, code, re.DOTALL)
        queries.extend(matches)

        # Look for %sql magic commands (Databricks notebooks)
        magic_pattern = r'%sql\s+(.*?)(?:%|\Z)'
        magic_matches = re.findall(magic_pattern, code, re.DOTALL)
        queries.extend(magic_matches)

        return queries

    def _convert_to_models(self, notebook_data: Dict[str, Any]) -> tuple:
        """
        Convert parsed notebook data to Process and Component objects

        Args:
            notebook_data: Parsed notebook data

        Returns:
            Tuple of (Process, List[Component])
        """
        notebook_name = notebook_data.get('name', 'unknown')
        notebook_path = notebook_data.get('file_path', '')

        # Create Process object
        process = Process(
            id=f"databricks_{notebook_name}",
            name=notebook_name,
            system=SystemType.DATABRICKS,
            process_type=ProcessType.NOTEBOOK,
            file_path=notebook_path,
            description=f"Databricks {notebook_data.get('notebook_type', 'Notebook')}: {notebook_name}",
            input_sources=notebook_data.get('input_tables', []),
            output_targets=notebook_data.get('output_tables', []),
        )

        # Create Component object (one per notebook for now)
        component = Component(
            id=f"{process.id}_main",
            name=notebook_name,
            component_type=ComponentType.NOTEBOOK,
            system="databricks",
            file_path=notebook_path,
            process_id=process.id,
            process_name=process.name,
            input_datasets=notebook_data.get('input_tables', []),
            output_datasets=notebook_data.get('output_tables', []),
            tables_read=notebook_data.get('input_tables', []),
            tables_written=notebook_data.get('output_tables', []),
        )

        # Update process
        process.component_ids = [component.id]
        process.component_count = 1

        return process, [component]
