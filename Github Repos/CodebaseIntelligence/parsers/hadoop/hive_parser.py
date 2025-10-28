"""
Hive/SQL Parser - Deep Query Analysis
Parses Hive and SQL scripts to extract queries, transformations, and logic
"""

import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from core.models.script_logic import (
    ScriptLogic, Transformation, ColumnLineage, TransformationType
)


class HiveParser:
    """Parse Hive HQL and SQL scripts"""

    def parse_hive_script(self, script_path: str, workflow_id: str = None, action_id: str = None) -> ScriptLogic:
        """Parse Hive/SQL script"""
        script_name = Path(script_path).stem

        try:
            with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {script_path}: {e}")
            return None

        # Generate ID
        normalized_path = str(Path(script_path).as_posix())
        content_hash = hashlib.md5(content.encode()).hexdigest()
        script_id = f"hive_{script_name}_{content_hash[:8]}"

        script_logic = ScriptLogic(
            script_id=script_id,
            script_name=script_name,
            script_type="hive",
            script_path=script_path,
            action_id=action_id,
            workflow_id=workflow_id,
            raw_content=content,
            content_hash=content_hash,
            lines_of_code=content.count('\n') + 1,
        )

        self._parse_sql_content(content, script_logic)

        logger.info(f"✓ Parsed Hive script: {script_name}")
        return script_logic

    def _parse_sql_content(self, content: str, script_logic: ScriptLogic):
        """Parse SQL/HiveQL content"""

        # Extract input tables (FROM, JOIN)
        from_pattern = r'FROM\s+([a-zA-Z0-9_\.]+)'
        for match in re.finditer(from_pattern, content, re.IGNORECASE):
            table = match.group(1)
            if table.upper() not in ['SELECT', 'WHERE', 'GROUP']:
                script_logic.input_tables.append(table)

        join_pattern = r'JOIN\s+([a-zA-Z0-9_\.]+)'
        for match in re.finditer(join_pattern, content, re.IGNORECASE):
            table = match.group(1)
            script_logic.input_tables.append(table)

        # Extract output tables (INSERT, CREATE TABLE)
        insert_pattern = r'INSERT\s+(?:OVERWRITE\s+)?(?:TABLE\s+)?([a-zA-Z0-9_\.]+)'
        for match in re.finditer(insert_pattern, content, re.IGNORECASE):
            script_logic.output_tables.append(match.group(1))

        create_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_\.]+)'
        for match in re.finditer(create_pattern, content, re.IGNORECASE):
            script_logic.output_tables.append(match.group(1))

        # Extract transformations
        trans_count = 0

        # WHERE clauses (filters)
        where_pattern = r'WHERE\s+(.+?)(?:GROUP\s+BY|ORDER\s+BY|LIMIT|;|\n\n)'
        for match in re.finditer(where_pattern, content, re.IGNORECASE | re.DOTALL):
            condition = match.group(1).strip()
            trans_count += 1
            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_filter_{trans_count}",
                transformation_type=TransformationType.FILTER,
                code_snippet=f"WHERE {condition}",
                condition=condition,
            )
            script_logic.transformations.append(trans)
            script_logic.data_filters.append(condition)

        # JOINs
        join_full_pattern = r'((?:LEFT|RIGHT|INNER|FULL)?\s*(?:OUTER\s+)?JOIN)\s+([a-zA-Z0-9_\.]+)\s+(?:ON|USING)\s+(.+?)(?:WHERE|JOIN|GROUP|ORDER|;)'
        for match in re.finditer(join_full_pattern, content, re.IGNORECASE | re.DOTALL):
            join_type = match.group(1).strip()
            table = match.group(2)
            condition = match.group(3).strip()

            trans_count += 1
            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_join_{trans_count}",
                transformation_type=TransformationType.JOIN,
                code_snippet=match.group(0)[:200],
                join_type=join_type.upper(),
                condition=condition,
            )
            script_logic.transformations.append(trans)

        # GROUP BY
        groupby_pattern = r'GROUP\s+BY\s+([^H]+?)(?:HAVING|ORDER|LIMIT|;)'
        for match in re.finditer(groupby_pattern, content, re.IGNORECASE | re.DOTALL):
            group_cols = match.group(1).strip()
            trans_count += 1
            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_group_{trans_count}",
                transformation_type=TransformationType.GROUP_BY,
                code_snippet=f"GROUP BY {group_cols}",
                group_by_columns=[col.strip() for col in group_cols.split(',')],
            )
            script_logic.transformations.append(trans)

        # Aggregations in SELECT
        agg_pattern = r'(COUNT|SUM|AVG|MAX|MIN|COLLECT_LIST|COLLECT_SET)\s*\('
        for match in re.finditer(agg_pattern, content, re.IGNORECASE):
            func = match.group(1).upper()
            if func not in [t.code_snippet for t in script_logic.transformations]:
                trans_count += 1
                trans = Transformation(
                    transformation_id=f"{script_logic.script_id}_agg_{trans_count}",
                    transformation_type=TransformationType.AGGREGATE,
                    code_snippet=f"{func}(...)",
                    functions_used=[func],
                )
                script_logic.transformations.append(trans)

        # Count by type
        for trans in script_logic.transformations:
            trans_type = trans.transformation_type.value
            script_logic.transformation_count_by_type[trans_type] = \
                script_logic.transformation_count_by_type.get(trans_type, 0) + 1
