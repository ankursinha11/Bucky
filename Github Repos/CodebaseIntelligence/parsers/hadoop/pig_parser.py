"""
Pig Script Parser - Deep Logic Extraction
Parses Pig Latin scripts to extract transformations, data lineage, and business logic
"""

import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

from core.models.script_logic import (
    ScriptLogic, Transformation, ColumnLineage, TransformationType
)


class PigScriptParser:
    """
    Deep parser for Pig Latin scripts

    Extracts:
    - Input/output tables and paths
    - Transformations (FILTER, JOIN, GROUP BY, etc.)
    - Column-level lineage
    - Business logic patterns
    """

    def __init__(self):
        """Initialize Pig parser"""
        pass

    def parse_pig_script(self, script_path: str, workflow_id: str = None, action_id: str = None) -> ScriptLogic:
        """
        Parse a Pig script file

        Args:
            script_path: Path to .pig file
            workflow_id: Parent workflow ID
            action_id: Parent action ID

        Returns:
            ScriptLogic object with complete analysis
        """
        script_name = Path(script_path).stem

        # Read content
        try:
            with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {script_path}: {e}")
            return None

        # Generate unique ID
        normalized_path = str(Path(script_path).as_posix())
        content_hash = hashlib.md5(content.encode()).hexdigest()
        script_id = f"pig_{script_name}_{content_hash[:8]}"

        # Initialize ScriptLogic
        script_logic = ScriptLogic(
            script_id=script_id,
            script_name=script_name,
            script_type="pig",
            script_path=script_path,
            action_id=action_id,
            workflow_id=workflow_id,
            raw_content=content,
            content_hash=content_hash,
            lines_of_code=content.count('\n') + 1,
        )

        # Parse content
        self._extract_inputs_outputs(content, script_logic)
        self._extract_variables(content, script_logic)
        self._extract_transformations(content, script_logic)
        self._extract_column_lineage(content, script_logic)
        self._identify_business_patterns(content, script_logic)

        logger.info(f"✓ Parsed Pig script: {script_name} ({len(script_logic.transformations)} transformations)")

        return script_logic

    def _extract_inputs_outputs(self, content: str, script_logic: ScriptLogic):
        """Extract input and output tables/files"""

        # LOAD statements (inputs)
        load_pattern = r'LOAD\s+[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(load_pattern, content, re.IGNORECASE):
            path = match.group(1)
            script_logic.input_files.append(path)

        # STORE statements (outputs)
        store_pattern = r'STORE\s+\w+\s+INTO\s+[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(store_pattern, content, re.IGNORECASE):
            path = match.group(1)
            script_logic.output_files.append(path)

        # Hive/HCatalog tables (inputs/outputs)
        hcat_load_pattern = r'HCatLoader\(\s*[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(hcat_load_pattern, content):
            table = match.group(1)
            script_logic.input_tables.append(table)

        hcat_store_pattern = r'HCatStorer\(\s*[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(hcat_store_pattern, content):
            table = match.group(1)
            script_logic.output_tables.append(table)

    def _extract_variables(self, content: str, script_logic: ScriptLogic):
        """Extract variables, parameters, and constants"""

        # SET statements (configuration)
        set_pattern = r'SET\s+(\S+)\s+(.+);'
        for match in re.finditer(set_pattern, content, re.IGNORECASE):
            var_name = match.group(1)
            var_value = match.group(2).strip().strip("'\"")
            script_logic.variables[var_name] = var_value

        # DEFINE statements (UDFs, macros)
        define_pattern = r'DEFINE\s+(\w+)\s+(.+);'
        for match in re.finditer(define_pattern, content, re.IGNORECASE):
            name = match.group(1)
            definition = match.group(2).strip()
            script_logic.variables[f"DEFINE_{name}"] = definition

            # Track UDFs
            if '(' in definition:  # Likely a UDF
                script_logic.udfs_used.append(name)

    def _extract_transformations(self, content: str, script_logic: ScriptLogic):
        """Extract all transformations (FILTER, JOIN, GROUP BY, etc.)"""

        # Track line numbers
        lines = content.split('\n')

        # Counter for transformation IDs
        trans_count = {"filter": 0, "join": 0, "group": 0, "foreach": 0, "distinct": 0, "union": 0, "order": 0}

        # FILTER transformations
        filter_pattern = r'(\w+)\s*=\s*FILTER\s+(\w+)\s+BY\s+(.+?);'
        for match in re.finditer(filter_pattern, content, re.IGNORECASE):
            alias = match.group(1)
            source = match.group(2)
            condition = match.group(3).strip()

            # Find line number
            line_num = self._find_line_number(content, match.start())

            trans_count["filter"] += 1
            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_filter_{trans_count['filter']}",
                transformation_type=TransformationType.FILTER,
                code_snippet=match.group(0),
                line_number=line_num,
                condition=condition,
            )

            # Parse condition for columns
            trans.columns = self._extract_columns_from_condition(condition)

            script_logic.transformations.append(trans)
            script_logic.data_filters.append(condition)

        # JOIN transformations
        join_pattern = r'(\w+)\s*=\s*JOIN\s+(.+?)\s+BY\s+(.+?);'
        for match in re.finditer(join_pattern, content, re.IGNORECASE | re.DOTALL):
            alias = match.group(1)
            tables_part = match.group(2).strip()
            keys_part = match.group(3).strip()

            line_num = self._find_line_number(content, match.start())

            trans_count["join"] += 1
            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_join_{trans_count['join']}",
                transformation_type=TransformationType.JOIN,
                code_snippet=match.group(0),
                line_number=line_num,
            )

            # Detect join type (LEFT, RIGHT, FULL, INNER)
            if 'LEFT OUTER' in match.group(0).upper() or 'LEFT' in tables_part.upper():
                trans.join_type = "LEFT OUTER JOIN"
                trans.transformation_type = TransformationType.LEFT_JOIN
            elif 'FULL OUTER' in match.group(0).upper():
                trans.join_type = "FULL OUTER JOIN"
                trans.transformation_type = TransformationType.OUTER_JOIN
            else:
                trans.join_type = "INNER JOIN"
                trans.transformation_type = TransformationType.INNER_JOIN

            # Parse join keys
            trans.join_keys = self._parse_join_keys(keys_part)

            script_logic.transformations.append(trans)

        # GROUP BY transformations
        group_pattern = r'(\w+)\s*=\s*GROUP\s+(\w+)\s+BY\s+(.+?);'
        for match in re.finditer(group_pattern, content, re.IGNORECASE):
            alias = match.group(1)
            source = match.group(2)
            group_keys = match.group(3).strip()

            line_num = self._find_line_number(content, match.start())

            trans_count["group"] += 1
            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_group_{trans_count['group']}",
                transformation_type=TransformationType.GROUP_BY,
                code_snippet=match.group(0),
                line_number=line_num,
            )

            # Parse group by columns
            trans.group_by_columns = [col.strip() for col in group_keys.split(',')]

            script_logic.transformations.append(trans)

        # FOREACH (transformations/projections)
        foreach_pattern = r'(\w+)\s*=\s*FOREACH\s+(\w+)\s+GENERATE\s+(.+?);'
        for match in re.finditer(foreach_pattern, content, re.IGNORECASE | re.DOTALL):
            alias = match.group(1)
            source = match.group(2)
            projections = match.group(3).strip()

            line_num = self._find_line_number(content, match.start())

            trans_count["foreach"] += 1
            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_transform_{trans_count['foreach']}",
                transformation_type=TransformationType.TRANSFORM,
                code_snippet=match.group(0),
                line_number=line_num,
            )

            # Extract columns and functions
            trans.columns, trans.functions_used = self._parse_foreach_generate(projections)

            # Check for aggregations
            agg_functions = ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN']
            for func in agg_functions:
                if func in projections.upper():
                    trans.transformation_type = TransformationType.AGGREGATE
                    trans.aggregations.append({"function": func, "column": "detected"})

            script_logic.transformations.append(trans)

        # DISTINCT transformations
        distinct_pattern = r'(\w+)\s*=\s*DISTINCT\s+(\w+);'
        for match in re.finditer(distinct_pattern, content, re.IGNORECASE):
            alias = match.group(1)
            source = match.group(2)

            line_num = self._find_line_number(content, match.start())

            trans_count["distinct"] += 1
            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_distinct_{trans_count['distinct']}",
                transformation_type=TransformationType.DISTINCT,
                code_snippet=match.group(0),
                line_number=line_num,
            )

            script_logic.transformations.append(trans)

        # ORDER BY transformations
        order_pattern = r'(\w+)\s*=\s*ORDER\s+(\w+)\s+BY\s+(.+?);'
        for match in re.finditer(order_pattern, content, re.IGNORECASE):
            alias = match.group(1)
            source = match.group(2)
            order_cols = match.group(3).strip()

            line_num = self._find_line_number(content, match.start())

            trans_count["order"] += 1
            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_sort_{trans_count['order']}",
                transformation_type=TransformationType.SORT,
                code_snippet=match.group(0),
                line_number=line_num,
                columns=[col.strip() for col in order_cols.split(',')],
            )

            script_logic.transformations.append(trans)

        # UNION transformations
        union_pattern = r'(\w+)\s*=\s*UNION\s+(.+?);'
        for match in re.finditer(union_pattern, content, re.IGNORECASE):
            alias = match.group(1)
            sources = match.group(2).strip()

            line_num = self._find_line_number(content, match.start())

            trans_count["union"] += 1
            trans = Transformation(
                transformation_id=f"{script_logic.script_id}_union_{trans_count['union']}",
                transformation_type=TransformationType.UNION,
                code_snippet=match.group(0),
                line_number=line_num,
            )

            script_logic.transformations.append(trans)

        # Count transformations by type
        for trans in script_logic.transformations:
            trans_type = trans.transformation_type.value
            script_logic.transformation_count_by_type[trans_type] = \
                script_logic.transformation_count_by_type.get(trans_type, 0) + 1

    def _extract_column_lineage(self, content: str, script_logic: ScriptLogic):
        """Extract column-level data lineage"""

        # This is simplified - full lineage tracking requires complex analysis
        # For now, extract from FOREACH GENERATE statements

        foreach_pattern = r'(\w+)\s*=\s*FOREACH\s+(\w+)\s+GENERATE\s+(.+?);'
        for match in re.finditer(foreach_pattern, content, re.IGNORECASE | re.DOTALL):
            target_alias = match.group(1)
            source_alias = match.group(2)
            projections = match.group(3).strip()

            # Parse projections (simplified)
            proj_parts = [p.strip() for p in projections.split(',')]

            for proj in proj_parts:
                # Handle "column AS alias" patterns
                if ' AS ' in proj.upper():
                    parts = re.split(r'\s+AS\s+', proj, flags=re.IGNORECASE)
                    source_expr = parts[0].strip()
                    target_col = parts[1].strip()
                else:
                    source_expr = proj
                    target_col = proj

                # Simple case: direct column reference
                if '.' in source_expr and not '(' in source_expr:
                    # e.g., "table.column"
                    lineage = ColumnLineage(
                        source_table=source_alias,
                        source_column=source_expr.split('.')[-1],
                        target_table=target_alias,
                        target_column=target_col,
                        is_pass_through=True,
                    )
                    script_logic.column_lineages.append(lineage)

                # Calculated column
                elif any(op in source_expr for op in ['+', '-', '*', '/', '(', 'CASE', 'IF']):
                    lineage = ColumnLineage(
                        source_table=source_alias,
                        source_column="calculated",
                        target_table=target_alias,
                        target_column=target_col,
                        is_calculated=True,
                        calculation_logic=source_expr,
                    )
                    script_logic.column_lineages.append(lineage)

                # Aggregation
                elif any(func in source_expr.upper() for func in ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN']):
                    lineage = ColumnLineage(
                        source_table=source_alias,
                        source_column="aggregated",
                        target_table=target_alias,
                        target_column=target_col,
                        is_aggregated=True,
                        transformation_logic=source_expr,
                    )
                    script_logic.column_lineages.append(lineage)

    def _identify_business_patterns(self, content: str, script_logic: ScriptLogic):
        """Identify common business logic patterns"""

        content_upper = content.upper()

        # Data quality checks
        if 'IS NOT NULL' in content_upper:
            script_logic.data_quality_checks.append("NULL validation")
            script_logic.null_handling.append("Filters out NULL values")

        if 'ISEMPTY' in content_upper or 'IS NULL' in content_upper:
            script_logic.null_handling.append("Checks for empty/null values")

        # Business rules
        if 'AGE' in content_upper and ('> 18' in content or '>= 18' in content):
            script_logic.key_business_rules.append("Filters to adult patients (age >= 18)")

        if 'STATUS' in content_upper and 'ACTIVE' in content_upper:
            script_logic.key_business_rules.append("Filters to active records only")

        if 'DATE' in content_upper and ('CURRENT' in content_upper or 'TODAY' in content_upper):
            script_logic.key_business_rules.append("Uses current date for filtering")

        # Deduplication
        if 'DISTINCT' in content_upper:
            script_logic.data_quality_checks.append("Deduplication (DISTINCT)")

        # Heavy operations (for performance awareness)
        if 'JOIN' in content_upper:
            join_count = content_upper.count('JOIN')
            if join_count > 3:
                script_logic.heavy_operations.append(f"Multiple JOINs ({join_count} joins)")

        if 'GROUP BY' in content_upper and 'ALL' in content_upper:
            script_logic.heavy_operations.append("GROUP ALL (expensive operation)")

    # Helper methods

    def _find_line_number(self, content: str, position: int) -> int:
        """Find line number from string position"""
        return content[:position].count('\n') + 1

    def _extract_columns_from_condition(self, condition: str) -> List[str]:
        """Extract column names from a condition"""
        # Simple pattern: look for identifiers
        # This is simplified - full parsing would use a proper grammar
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b'
        matches = re.findall(pattern, condition)

        # Filter out SQL keywords and operators
        keywords = {'AND', 'OR', 'NOT', 'IS', 'NULL', 'TRUE', 'FALSE', 'IN', 'LIKE', 'BETWEEN'}
        columns = [m for m in matches if m.upper() not in keywords and not m.isdigit()]

        return list(set(columns))  # Unique columns

    def _parse_join_keys(self, keys_part: str) -> List[str]:
        """Parse join keys from BY clause"""
        # Handle both simple and complex join keys
        # e.g., "left_alias::col1, right_alias::col2"
        keys = []
        parts = keys_part.split(',')
        for part in parts:
            if '::' in part:
                keys.append(part.split('::')[-1].strip())
            else:
                keys.append(part.strip())
        return keys

    def _parse_foreach_generate(self, projections: str) -> Tuple[List[str], List[str]]:
        """Parse FOREACH GENERATE to extract columns and functions"""
        columns = []
        functions = []

        proj_parts = [p.strip() for p in projections.split(',')]

        for proj in proj_parts:
            # Extract function calls
            func_pattern = r'(\w+)\s*\('
            for match in re.finditer(func_pattern, proj):
                func_name = match.group(1)
                if func_name.upper() not in ['AS']:  # Exclude keywords
                    functions.append(func_name.upper())

            # Extract column references (simplified)
            col_pattern = r'(?:^|[\s\(])([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
            for match in re.finditer(col_pattern, proj):
                col = match.group(1)
                if col.upper() not in functions and col.upper() not in ['AS', 'AND', 'OR']:
                    columns.append(col)

        return list(set(columns)), list(set(functions))
