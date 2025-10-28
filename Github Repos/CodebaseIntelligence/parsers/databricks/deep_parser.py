"""
Deep Databricks Parser - Cell-Level Analysis
Parses notebooks with cell-by-cell logic extraction
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List
from loguru import logger

from core.models import Repository, RepositoryType, WorkflowFlow, ActionNode, ActionType
from core.models.script_logic import ScriptLogic, Transformation, TransformationType
from parsers.databricks.parser import DatabricksParser


class DeepDatabricksParser:
    """Enhanced Databricks parser with cell-level deep parsing"""

    def __init__(self, use_ai: bool = True):
        self.base_parser = DatabricksParser()
        self.use_ai = use_ai

    def parse_directory(self, databricks_path: str) -> Dict:
        """Parse with cell-level analysis"""
        logger.info(f"🔍 Deep parsing Databricks: {databricks_path}")

        base_result = self.base_parser.parse_directory(databricks_path)
        processes = base_result["processes"]
        components = base_result["components"]

        # Create repository
        repository = self._create_repository(databricks_path, processes, components)

        # Create workflow flows
        workflow_flows = []
        for process in processes:
            workflow = WorkflowFlow(
                workflow_id=process.id,
                workflow_name=process.name,
                workflow_type="notebook",
                repository_id=repository.id,
                file_path=process.file_path,
            )
            workflow_flows.append(workflow)

        # Parse notebook cells
        script_logics = self._parse_notebooks(processes, databricks_path)

        logger.info(f"✓ Parsed {len(script_logics)} notebooks with cell-level analysis")

        return {
            "repository": repository,
            "workflow_flows": workflow_flows,
            "script_logics": script_logics,
            "processes": processes,
            "components": components,
        }

    def _create_repository(self, base_path: str, processes: List, components: List) -> Repository:
        """Create repository"""
        repo_name = Path(base_path).name
        return Repository(
            id=f"repo_{repo_name}",
            name=repo_name,
            repo_type=RepositoryType.DATABRICKS,
            base_path=base_path,
            total_workflows=len(processes),
            total_notebooks=len(processes),
            total_scripts=len(components),
            python_scripts=len([c for c in components if '.py' in (c.file_path or '')]),
            sql_scripts=len([c for c in components if '.sql' in (c.file_path or '')]),
            workflow_ids=[p.id for p in processes],
            technologies=["PySpark", "SQL", "Delta Lake"],
        )

    def _parse_notebooks(self, processes: List, base_path: str) -> List[ScriptLogic]:
        """Parse notebooks cell by cell"""
        script_logics = []

        for process in processes:
            if not process.file_path:
                continue

            if process.file_path.endswith('.ipynb'):
                script = self._parse_jupyter_notebook(process.file_path, process.id)
                if script:
                    script_logics.append(script)

        return script_logics

    def _parse_jupyter_notebook(self, notebook_path: str, workflow_id: str) -> ScriptLogic:
        """Parse Jupyter notebook cell by cell"""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
        except:
            return None

        script_name = Path(notebook_path).stem
        content_hash = hashlib.md5(str(notebook).encode()).hexdigest()
        script_id = f"databricks_{script_name}_{content_hash[:8]}"

        # Combine all cell content
        all_content = []
        for cell in notebook.get('cells', []):
            source = ''.join(cell.get('source', []))
            all_content.append(source)

        full_content = '\n\n'.join(all_content)

        script_logic = ScriptLogic(
            script_id=script_id,
            script_name=script_name,
            script_type="databricks_notebook",
            script_path=notebook_path,
            workflow_id=workflow_id,
            raw_content=full_content,
            content_hash=content_hash,
            lines_of_code=full_content.count('\n') + 1,
        )

        # Parse cell transformations
        cell_num = 0
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue

            cell_num += 1
            source = ''.join(cell.get('source', []))

            # Look for SQL magic
            if source.strip().startswith('%sql'):
                trans = Transformation(
                    transformation_id=f"{script_id}_cell_{cell_num}_sql",
                    transformation_type=TransformationType.UNKNOWN,
                    code_snippet=source[:200],
                    line_number=cell_num,
                )
                script_logic.transformations.append(trans)

            # Look for PySpark transformations
            if 'filter(' in source or 'where(' in source:
                trans = Transformation(
                    transformation_id=f"{script_id}_cell_{cell_num}_filter",
                    transformation_type=TransformationType.FILTER,
                    code_snippet=source[:200],
                    line_number=cell_num,
                )
                script_logic.transformations.append(trans)

            if 'join(' in source:
                trans = Transformation(
                    transformation_id=f"{script_id}_cell_{cell_num}_join",
                    transformation_type=TransformationType.JOIN,
                    code_snippet=source[:200],
                    line_number=cell_num,
                )
                script_logic.transformations.append(trans)

            if 'groupBy(' in source or 'agg(' in source:
                trans = Transformation(
                    transformation_id=f"{script_id}_cell_{cell_num}_group",
                    transformation_type=TransformationType.GROUP_BY,
                    code_snippet=source[:200],
                    line_number=cell_num,
                )
                script_logic.transformations.append(trans)

        return script_logic
