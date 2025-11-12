"""
Databricks Notebook Analyzer
Analyzes Databricks notebooks to extract pipeline structure, dependencies, and data flow
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class DatabricksNotebook:
    """Represents a single Databricks notebook"""
    path: str
    name: str
    pipeline: str  # e.g., "CDD/ie_prebdf"

    # Parameters
    parameters: List[str] = field(default_factory=list)

    # Dependencies
    imported_notebooks: List[str] = field(default_factory=list)
    imported_modules: List[str] = field(default_factory=list)

    # Data sources
    input_tables: List[str] = field(default_factory=list)
    output_tables: List[str] = field(default_factory=list)
    input_paths: List[str] = field(default_factory=list)
    output_paths: List[str] = field(default_factory=list)

    # Transformations
    key_operations: List[str] = field(default_factory=list)

    # Schema
    schema_definitions: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "path": self.path,
            "name": self.name,
            "pipeline": self.pipeline,
            "parameters": self.parameters,
            "imported_notebooks": self.imported_notebooks,
            "imported_modules": self.imported_modules,
            "input_tables": self.input_tables,
            "output_tables": self.output_tables,
            "input_paths": self.input_paths,
            "output_paths": self.output_paths,
            "key_operations": self.key_operations,
            "schema_count": len(self.schema_definitions)
        }


@dataclass
class DatabricksPipeline:
    """Represents a Databricks pipeline (group of related notebooks)"""
    name: str
    path: str
    notebooks: List[DatabricksNotebook] = field(default_factory=list)

    def get_execution_order(self) -> List[str]:
        """Infer execution order based on dependencies"""
        # Simple heuristic: notebooks with fewer dependencies run first
        sorted_notebooks = sorted(
            self.notebooks,
            key=lambda nb: len(nb.imported_notebooks)
        )
        return [nb.name for nb in sorted_notebooks]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "path": self.path,
            "notebook_count": len(self.notebooks),
            "notebooks": [nb.to_dict() for nb in self.notebooks],
            "execution_order": self.get_execution_order()
        }


class DatabricksAnalyzer:
    """
    Analyze Databricks repository to understand pipeline structure
    """

    def __init__(self, repo_path: str):
        """Initialize analyzer with repository path"""
        self.repo_path = Path(repo_path)
        self.notebooks: List[DatabricksNotebook] = []
        self.pipelines: Dict[str, DatabricksPipeline] = {}

    def analyze_repository(self) -> Dict[str, DatabricksPipeline]:
        """Analyze entire Databricks repository"""
        logger.info(f"Analyzing Databricks repository: {self.repo_path}")

        # Find all Python/SQL/notebook files
        python_files = list(self.repo_path.rglob("*.py"))
        sql_files = list(self.repo_path.rglob("*.sql"))
        notebook_files = list(self.repo_path.rglob("*.ipynb"))

        all_files = python_files + sql_files + notebook_files
        logger.info(f"Found {len(all_files)} files ({len(python_files)} py, {len(sql_files)} sql, {len(notebook_files)} ipynb)")

        # Analyze each file
        for file_path in all_files:
            try:
                notebook = self._analyze_notebook(file_path)
                if notebook:
                    self.notebooks.append(notebook)

                    # Group into pipelines
                    pipeline_name = notebook.pipeline
                    if pipeline_name not in self.pipelines:
                        self.pipelines[pipeline_name] = DatabricksPipeline(
                            name=pipeline_name,
                            path=str(file_path.parent)
                        )
                    self.pipelines[pipeline_name].notebooks.append(notebook)

            except Exception as e:
                logger.debug(f"Error analyzing {file_path.name}: {e}")

        logger.info(f"Analyzed {len(self.notebooks)} notebooks across {len(self.pipelines)} pipelines")
        return self.pipelines

    def _analyze_notebook(self, file_path: Path) -> Optional[DatabricksNotebook]:
        """Analyze a single notebook file"""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Determine pipeline from path
            relative_path = file_path.relative_to(self.repo_path)
            parts = list(relative_path.parts)
            if len(parts) >= 2:
                pipeline = f"{parts[0]}/{parts[1]}" if len(parts) > 1 else parts[0]
            else:
                pipeline = parts[0] if parts else "unknown"

            notebook = DatabricksNotebook(
                path=str(file_path),
                name=file_path.stem,
                pipeline=pipeline
            )

            # Extract parameters
            notebook.parameters = self._extract_parameters(content)

            # Extract imports
            notebook.imported_notebooks = self._extract_notebook_imports(content)
            notebook.imported_modules = self._extract_module_imports(content)

            # Extract data sources
            notebook.input_tables, notebook.output_tables = self._extract_tables(content)
            notebook.input_paths, notebook.output_paths = self._extract_file_paths(content)

            # Extract key operations
            notebook.key_operations = self._extract_operations(content)

            # Extract schema definitions
            notebook.schema_definitions = self._extract_schemas(content)

            return notebook

        except Exception as e:
            logger.debug(f"Failed to analyze {file_path.name}: {e}")
            return None

    def _extract_parameters(self, content: str) -> List[str]:
        """Extract dbutils.widgets parameters"""
        params = []

        # Pattern: dbutils.widgets.text("param_name", ...)
        widget_pattern = r'dbutils\.widgets\.(text|dropdown|combobox|multiselect)\s*\(\s*["\']([^"\']+)["\']'
        matches = re.findall(widget_pattern, content)

        for _, param_name in matches:
            if param_name not in params:
                params.append(param_name)

        return params

    def _extract_notebook_imports(self, content: str) -> List[str]:
        """Extract %run notebook imports"""
        imports = []

        # Pattern: %run "path/to/notebook" or %run /path/to/notebook
        run_pattern = r'%run\s+["\']?([^"\'\\n]+)["\']?'
        matches = re.findall(run_pattern, content)

        for path in matches:
            path = path.strip()
            if path and not path.startswith('#'):
                imports.append(path)

        return imports

    def _extract_module_imports(self, content: str) -> List[str]:
        """Extract Python module imports"""
        modules = []

        # Pattern: import module or from module import ...
        import_pattern = r'^(?:from\s+([^\s]+)|import\s+([^\s,]+))'

        for line in content.split('\n'):
            match = re.match(import_pattern, line.strip())
            if match:
                module = match.group(1) or match.group(2)
                if module and not module.startswith('.'):
                    modules.append(module.split('.')[0])  # Get root module

        return list(set(modules))

    def _extract_tables(self, content: str) -> Tuple[List[str], List[str]]:
        """Extract input and output tables"""
        input_tables = []
        output_tables = []

        # Input patterns
        read_patterns = [
            r'spark\.read\.table\(["\']([^"\']+)["\']',
            r'spark\.table\(["\']([^"\']+)["\']',
            r'FROM\s+([a-z_][a-z0-9_.]+)',  # SQL FROM
            r'JOIN\s+([a-z_][a-z0-9_.]+)',  # SQL JOIN
        ]

        for pattern in read_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            input_tables.extend(matches)

        # Output patterns
        write_patterns = [
            r'saveAsTable\(["\']([^"\']+)["\']',
            r'insertInto\(["\']([^"\']+)["\']',
            r'CREATE\s+(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-z_][a-z0-9_.]+)',  # SQL CREATE
            r'INSERT\s+(?:INTO|OVERWRITE)\s+(?:TABLE\s+)?([a-z_][a-z0-9_.]+)',  # SQL INSERT
        ]

        for pattern in write_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            output_tables.extend(matches)

        return list(set(input_tables)), list(set(output_tables))

    def _extract_file_paths(self, content: str) -> Tuple[List[str], List[str]]:
        """Extract input and output file paths"""
        input_paths = []
        output_paths = []

        # Input path patterns
        read_path_patterns = [
            r'spark\.read\.[a-z]+\(["\']([^"\']+\.(?:csv|parquet|json|txt|avro|delta))["\']',
            r'load\(["\']([^"\']+)["\']',
        ]

        for pattern in read_path_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            input_paths.extend(matches)

        # Output path patterns
        write_path_patterns = [
            r'write\.[a-z]+\(["\']([^"\']+)["\']',
            r'save\(["\']([^"\']+)["\']',
        ]

        for pattern in write_path_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            output_paths.extend(matches)

        return list(set(input_paths)), list(set(output_paths))

    def _extract_operations(self, content: str) -> List[str]:
        """Extract key transformation operations"""
        operations = []

        # Common Spark operations
        spark_ops = [
            'filter', 'select', 'groupBy', 'agg', 'join',
            'union', 'distinct', 'drop', 'withColumn',
            'repartition', 'coalesce', 'cache', 'persist'
        ]

        for op in spark_ops:
            # Check if operation is used
            pattern = rf'\.{op}\s*\('
            if re.search(pattern, content):
                operations.append(op)

        return operations

    def _extract_schemas(self, content: str) -> List[Dict]:
        """Extract schema definitions (StructType)"""
        schemas = []

        # Pattern: StructType([StructField(...)])
        schema_pattern = r'StructType\s*\(\s*\[(.*?)\]\s*\)'
        matches = re.findall(schema_pattern, content, re.DOTALL)

        for match in matches:
            # Extract field names
            field_pattern = r'StructField\s*\(\s*["\']([^"\']+)["\']'
            fields = re.findall(field_pattern, match)

            if fields:
                schemas.append({
                    "fields": fields,
                    "field_count": len(fields)
                })

        return schemas

    def get_pipeline_summary(self) -> Dict:
        """Get summary of all pipelines"""
        summary = {
            "total_pipelines": len(self.pipelines),
            "total_notebooks": len(self.notebooks),
            "pipelines": {}
        }

        for pipeline_name, pipeline in self.pipelines.items():
            summary["pipelines"][pipeline_name] = {
                "notebook_count": len(pipeline.notebooks),
                "total_parameters": sum(len(nb.parameters) for nb in pipeline.notebooks),
                "total_dependencies": sum(len(nb.imported_notebooks) for nb in pipeline.notebooks),
                "notebooks": [nb.name for nb in pipeline.notebooks]
            }

        return summary


# Testing function
def test_databricks_analyzer():
    """Test the analyzer"""
    import json

    print("=" * 60)
    print("DATABRICKS REPOSITORY ANALYSIS")
    print("=" * 60)

    repo_path = "/Users/ankurshome/Desktop/Hadoop_Parser/CodebaseIntelligence/Databricks_repo"

    analyzer = DatabricksAnalyzer(repo_path)
    pipelines = analyzer.analyze_repository()

    # Print summary
    summary = analyzer.get_pipeline_summary()

    print(f"\n📊 Summary:")
    print(f"   Total Pipelines: {summary['total_pipelines']}")
    print(f"   Total Notebooks: {summary['total_notebooks']}")

    print(f"\n📁 Pipelines:")
    for pipeline_name, pipeline_info in sorted(summary['pipelines'].items()):
        print(f"\n   {pipeline_name}:")
        print(f"      Notebooks: {pipeline_info['notebook_count']}")
        print(f"      Parameters: {pipeline_info['total_parameters']}")
        print(f"      Dependencies: {pipeline_info['total_dependencies']}")
        print(f"      Files: {', '.join(pipeline_info['notebooks'][:5])}" +
              (" ..." if len(pipeline_info['notebooks']) > 5 else ""))

    # Save detailed analysis
    output_file = "databricks_pipeline_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            "summary": summary,
            "pipelines": {name: pipeline.to_dict() for name, pipeline in pipelines.items()}
        }, f, indent=2)

    print(f"\n✅ Detailed analysis saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    test_databricks_analyzer()
