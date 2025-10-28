"""
Spark Script Parser - Deep Logic Extraction
Parses PySpark and Scala Spark scripts
"""

import re
import hashlib
import ast
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from core.models.script_logic import (
    ScriptLogic, Transformation, ColumnLineage, TransformationType
)


class SparkScriptParser:
    """Parse PySpark and Spark Scala scripts"""

    def parse_spark_script(self, script_path: str, workflow_id: str = None, action_id: str = None) -> ScriptLogic:
        """Parse Spark script"""
        script_name = Path(script_path).stem
        file_ext = Path(script_path).suffix

        try:
            with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {script_path}: {e}")
            return None

        # Determine language
        if file_ext == '.py':
            script_type = "pyspark"
        elif file_ext == '.scala':
            script_type = "scala_spark"
        else:
            script_type = "spark"

        # Generate ID
        normalized_path = str(Path(script_path).as_posix())
        content_hash = hashlib.md5(content.encode()).hexdigest()
        script_id = f"spark_{script_name}_{content_hash[:8]}"

        script_logic = ScriptLogic(
            script_id=script_id,
            script_name=script_name,
            script_type=script_type,
            script_path=script_path,
            action_id=action_id,
            workflow_id=workflow_id,
            raw_content=content,
            content_hash=content_hash,
            lines_of_code=content.count('\n') + 1,
        )

        if script_type == "pyspark":
            self._parse_pyspark(content, script_logic)
        else:
            self._parse_scala_spark(content, script_logic)

        logger.info(f"✓ Parsed Spark script: {script_name}")
        return script_logic

    def _parse_pyspark(self, content: str, script_logic: ScriptLogic):
        """Parse PySpark Python code"""

        # Extract inputs
        read_patterns = [
            r'\.read\.(?:parquet|csv|json|orc|avro|table)\(["\']([^"\']+)["\']',
            r'\.table\(["\']([^"\']+)["\']',
            r'spark\.read\(["\']([^"\']+)["\']',
        ]
        for pattern in read_patterns:
            for match in re.finditer(pattern, content):
                path = match.group(1)
                if path.startswith('/') or path.startswith('dbfs:'):
                    script_logic.input_files.append(path)
                else:
                    script_logic.input_tables.append(path)

        # Extract outputs
        write_patterns = [
            r'\.write\.(?:parquet|csv|json|orc|avro|saveAsTable)\(["\']([^"\']+)["\']',
            r'\.saveAsTable\(["\']([^"\']+)["\']',
        ]
        for pattern in write_patterns:
            for match in re.finditer(pattern, content):
                path = match.group(1)
                if path.startswith('/') or path.startswith('dbfs:'):
                    script_logic.output_files.append(path)
                else:
                    script_logic.output_tables.append(path)

        # Extract transformations
        trans_count = 0

        # FILTER/WHERE
        filter_pattern = r'\.(?:filter|where)\(["\']?([^"\'()]+)["\']?\)'
        for match in re.finditer(filter_pattern, content):
            condition = match.group(1).strip()
            trans_count += 1
            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_filter_{trans_count}",
                transformation_type=TransformationType.FILTER,
                code_snippet=match.group(0),
                condition=condition,
            )
            script_logic.transformations.append(trans)
            script_logic.data_filters.append(condition)

        # JOIN
        join_pattern = r'\.join\(([^,]+),\s*(?:on=)?([^,)]+)(?:,\s*how=["\']([^"\']+)["\'])?\)'
        for match in re.finditer(join_pattern, content):
            right_df = match.group(1).strip()
            join_cond = match.group(2).strip()
            join_how = match.group(3) if match.group(3) else "inner"

            trans_count += 1
            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_join_{trans_count}",
                transformation_type=TransformationType.JOIN if join_how == "inner" else TransformationType.LEFT_JOIN,
                code_snippet=match.group(0),
                join_type=join_how.upper() + " JOIN",
                condition=join_cond,
            )
            script_logic.transformations.append(trans)

        # GROUP BY
        groupby_pattern = r'\.groupBy\(([^)]+)\)\.agg\(([^)]+)\)'
        for match in re.finditer(groupby_pattern, content):
            group_cols = match.group(1).strip()
            agg_funcs = match.group(2).strip()

            trans_count += 1
            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_group_{trans_count}",
                transformation_type=TransformationType.GROUP_BY,
                code_snippet=match.group(0),
                group_by_columns=[col.strip().strip('"\'') for col in group_cols.split(',')],
            )
            script_logic.transformations.append(trans)

        # SELECT (projections)
        select_pattern = r'\.select\(([^)]+)\)'
        for match in re.finditer(select_pattern, content):
            columns = match.group(1).strip()
            trans_count += 1
            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_select_{trans_count}",
                transformation_type=TransformationType.TRANSFORM,
                code_snippet=match.group(0),
                columns=[col.strip().strip('"\'') for col in columns.split(',')],
            )
            script_logic.transformations.append(trans)

        # COUNT
        for trans in script_logic.transformations:
            trans_type = trans.transformation_type.value
            script_logic.transformation_count_by_type[trans_type] = \
                script_logic.transformation_count_by_type.get(trans_type, 0) + 1

    def _parse_scala_spark(self, content: str, script_logic: ScriptLogic):
        """Parse Scala Spark code (simplified)"""
        # Similar patterns to PySpark but with Scala syntax
        # This is a simplified version
        pass
