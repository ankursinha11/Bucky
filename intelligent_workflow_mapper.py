"""
Intelligent Workflow Mapper
Maps workflows across Hadoop (legacy), Databricks (current), and Ab Initio (running)
based on logic similarity, data flow, and transformations - NOT just names

Context:
- Hadoop: Legacy code (migration happened 2 years ago)
- Databricks: Current production (everything runs here now)
- Ab Initio: Still running (fully working)

Purpose:
- Validate Databricks migration from Hadoop is complete/correct
- Understand how Databricks correlates with Ab Initio
- Answer: "What Databricks pipeline replaced Hadoop workflow X?"
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class WorkflowSignature:
    """
    Signature of a workflow based on logic, not names

    This captures the ESSENCE of what a workflow does:
    - What data it reads (sources)
    - What transformations it applies
    - What data it writes (targets)
    - Business logic patterns
    """
    system: str  # hadoop, databricks, abinitio
    name: str
    file_path: str

    # Data flow
    source_tables: Set[str] = field(default_factory=set)
    target_tables: Set[str] = field(default_factory=set)
    source_file_patterns: Set[str] = field(default_factory=set)
    target_file_patterns: Set[str] = field(default_factory=set)

    # Logic patterns
    transformation_types: List[str] = field(default_factory=list)
    filter_patterns: List[str] = field(default_factory=list)
    join_patterns: List[str] = field(default_factory=list)
    aggregation_patterns: List[str] = field(default_factory=list)

    # Business concepts (extracted from code/comments)
    business_keywords: Set[str] = field(default_factory=set)

    # Column operations
    column_transformations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "system": self.system,
            "name": self.name,
            "file_path": self.file_path,
            "source_tables": list(self.source_tables),
            "target_tables": list(self.target_tables),
            "source_file_patterns": list(self.source_file_patterns),
            "target_file_patterns": list(self.target_file_patterns),
            "transformation_types": self.transformation_types,
            "filter_patterns": self.filter_patterns,
            "join_patterns": self.join_patterns,
            "aggregation_patterns": self.aggregation_patterns,
            "business_keywords": list(self.business_keywords),
            "column_transformations": self.column_transformations
        }


@dataclass
class WorkflowMapping:
    """
    Represents a mapping between workflows across systems

    Can be:
    - 1:1 (one Hadoop workflow → one Databricks pipeline)
    - N:1 (multiple Hadoop workflows → one Databricks pipeline)
    - 1:N (one Hadoop workflow → multiple Databricks notebooks)
    - N:M (complex mapping)
    """
    source_system: str
    target_system: str

    source_workflows: List[WorkflowSignature] = field(default_factory=list)
    target_workflows: List[WorkflowSignature] = field(default_factory=list)

    # Similarity scores
    data_flow_similarity: float = 0.0  # How similar are source→target mappings?
    logic_similarity: float = 0.0      # How similar are transformations?
    business_similarity: float = 0.0   # How similar are business concepts?
    overall_similarity: float = 0.0    # Weighted average

    # Mapping details
    mapping_type: str = "1:1"  # "1:1", "N:1", "1:N", "N:M"
    confidence: str = "unknown"  # "high", "medium", "low"

    # Analysis
    shared_tables: Set[str] = field(default_factory=set)
    missing_in_target: Set[str] = field(default_factory=set)
    added_in_target: Set[str] = field(default_factory=set)
    transformation_differences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "source_system": self.source_system,
            "target_system": self.target_system,
            "source_workflows": [w.name for w in self.source_workflows],
            "target_workflows": [w.name for w in self.target_workflows],
            "mapping_type": self.mapping_type,
            "confidence": self.confidence,
            "similarity_scores": {
                "data_flow": self.data_flow_similarity,
                "logic": self.logic_similarity,
                "business": self.business_similarity,
                "overall": self.overall_similarity
            },
            "analysis": {
                "shared_tables": list(self.shared_tables),
                "missing_in_target": list(self.missing_in_target),
                "added_in_target": list(self.added_in_target),
                "transformation_differences": self.transformation_differences
            }
        }


class IntelligentWorkflowMapper:
    """
    Intelligent workflow mapper using logic analysis

    Uses AI + pattern matching to map workflows across systems based on:
    1. Data flow (source tables → target tables)
    2. Transformation logic (filters, joins, aggregations)
    3. Business concepts (patient, eligibility, commercial, etc.)
    4. Column operations (what columns are transformed how)
    """

    def __init__(self, ai_client=None):
        """Initialize mapper"""
        self.ai_client = ai_client
        self.workflow_signatures: Dict[str, List[WorkflowSignature]] = {
            "hadoop": [],
            "databricks": [],
            "abinitio": []
        }
        self.mappings: List[WorkflowMapping] = []

    def extract_signature(
        self,
        system: str,
        name: str,
        file_path: str,
        code_content: str
    ) -> WorkflowSignature:
        """
        Extract workflow signature from code

        This is the CORE - extracts the essence of what a workflow does
        """
        signature = WorkflowSignature(
            system=system,
            name=name,
            file_path=file_path
        )

        # Extract based on system type
        if system == "hadoop":
            self._extract_hadoop_signature(signature, code_content)
        elif system == "databricks":
            self._extract_databricks_signature(signature, code_content)
        elif system == "abinitio":
            self._extract_abinitio_signature(signature, code_content)

        # Extract common business keywords
        signature.business_keywords = self._extract_business_keywords(code_content)

        return signature

    def _extract_hadoop_signature(self, sig: WorkflowSignature, code: str):
        """Extract signature from Hadoop Pig/Hive code"""
        code_upper = code.upper()

        # Source tables (LOAD, FROM)
        # Pig: LOAD 'path' or FROM table
        load_pattern = r"LOAD\s+'([^']+)'"
        from_pattern = r"FROM\s+([a-z_][a-z0-9_]*)"

        for match in re.findall(load_pattern, code, re.IGNORECASE):
            sig.source_file_patterns.add(match)

        for match in re.findall(from_pattern, code, re.IGNORECASE):
            sig.source_tables.add(match.lower())

        # Target tables (STORE, INSERT INTO)
        store_pattern = r"STORE\s+\w+\s+INTO\s+'([^']+)'"
        insert_pattern = r"INSERT\s+(?:INTO|OVERWRITE)\s+(?:TABLE\s+)?([a-z_][a-z0-9_]*)"

        for match in re.findall(store_pattern, code, re.IGNORECASE):
            sig.target_file_patterns.add(match)

        for match in re.findall(insert_pattern, code, re.IGNORECASE):
            sig.target_tables.add(match.lower())

        # Transformations
        if 'FILTER' in code_upper:
            sig.transformation_types.append('filter')
            # Extract filter conditions
            filter_conditions = re.findall(r'FILTER\s+\w+\s+BY\s+([^;]+)', code, re.IGNORECASE)
            sig.filter_patterns.extend(filter_conditions)

        if 'JOIN' in code_upper:
            sig.transformation_types.append('join')
            join_patterns = re.findall(r'JOIN\s+(\w+)\s+ON\s+([^;]+)', code, re.IGNORECASE)
            sig.join_patterns.extend([f"{table} ON {cond}" for table, cond in join_patterns])

        if 'GROUP' in code_upper:
            sig.transformation_types.append('group_by')
            group_patterns = re.findall(r'GROUP\s+\w+\s+BY\s+([^;]+)', code, re.IGNORECASE)
            sig.aggregation_patterns.extend(group_patterns)

        # Column transformations
        # Pig: FOREACH ... GENERATE
        foreach_pattern = r'FOREACH\s+\w+\s+GENERATE\s+([^;]+)'
        for match in re.findall(foreach_pattern, code, re.IGNORECASE):
            # Parse column transformations
            columns = [col.strip() for col in match.split(',')]
            for col in columns[:10]:  # Limit to first 10
                if 'AS' in col.upper():
                    parts = col.upper().split(' AS ')
                    if len(parts) == 2:
                        sig.column_transformations[parts[1].strip()] = parts[0].strip()

    def _extract_databricks_signature(self, sig: WorkflowSignature, code: str):
        """Extract signature from Databricks PySpark code"""
        code_lower = code.lower()

        # Source tables
        # spark.read.table("table_name")
        # spark.table("table_name")
        read_table_pattern = r'(?:read\.table|spark\.table)\s*\(\s*["\']([^"\']+)["\']\s*\)'
        for match in re.findall(read_table_pattern, code, re.IGNORECASE):
            sig.source_tables.add(match.lower())

        # spark.read.csv/parquet/etc
        read_file_pattern = r'read\.\w+\s*\(\s*["\']([^"\']+)["\']\s*\)'
        for match in re.findall(read_file_pattern, code, re.IGNORECASE):
            sig.source_file_patterns.add(match)

        # FROM in SQL
        from_pattern = r'FROM\s+([a-z_][a-z0-9_]*)'
        for match in re.findall(from_pattern, code, re.IGNORECASE):
            sig.source_tables.add(match.lower())

        # Target tables
        # saveAsTable("table_name")
        # insertInto("table_name")
        save_table_pattern = r'(?:saveAsTable|insertInto)\s*\(\s*["\']([^"\']+)["\']\s*\)'
        for match in re.findall(save_table_pattern, code, re.IGNORECASE):
            sig.target_tables.add(match.lower())

        # write.mode().save("path")
        write_pattern = r'write\.\w+\([^)]*\)\.save\s*\(\s*["\']([^"\']+)["\']\s*\)'
        for match in re.findall(write_pattern, code, re.IGNORECASE):
            sig.target_file_patterns.add(match)

        # Transformations
        if '.filter(' in code_lower or '.where(' in code_lower:
            sig.transformation_types.append('filter')
            # Extract filter conditions
            filter_pattern = r'\.(?:filter|where)\s*\(\s*["\']?([^)"\']+)["\']?\s*\)'
            sig.filter_patterns.extend(re.findall(filter_pattern, code, re.IGNORECASE))

        if '.join(' in code_lower:
            sig.transformation_types.append('join')
            # Extract join conditions
            join_pattern = r'\.join\s*\([^,]+,\s*([^)]+)\)'
            sig.join_patterns.extend(re.findall(join_pattern, code, re.IGNORECASE))

        if '.groupBy(' in code_lower or '.groupby(' in code_lower:
            sig.transformation_types.append('group_by')
            group_pattern = r'\.group[Bb]y\s*\(\s*([^)]+)\s*\)'
            sig.aggregation_patterns.extend(re.findall(group_pattern, code, re.IGNORECASE))

        # Column transformations
        # withColumn("new_col", transformation)
        withcol_pattern = r'withColumn\s*\(\s*["\']([^"\']+)["\']\s*,\s*([^)]+)\)'
        for col_name, transformation in re.findall(withcol_pattern, code, re.IGNORECASE):
            sig.column_transformations[col_name] = transformation[:100]  # Truncate long transformations

    def _extract_abinitio_signature(self, sig: WorkflowSignature, code: str):
        """Extract signature from Ab Initio graph data"""
        # Ab Initio signature comes from mapping Excel, not code
        # This will be populated when integrating with abinitio_mapping_parser
        pass

    def _extract_business_keywords(self, code: str) -> Set[str]:
        """Extract business domain keywords from code/comments"""
        # Common healthcare/insurance business terms
        business_terms = [
            'patient', 'eligibility', 'commercial', 'medicaid', 'medicare',
            'payer', 'claim', 'member', 'account', 'provider', 'hospital',
            'coverage', 'enrollment', 'benefit', 'policy', 'premium',
            'bdf', 'prebdf', 'postbdf', 'gmrn', 'mrn', 'lead', 'family',
            'demographic', 'address', 'phone', 'ssn', 'dob', 'gender',
            'ack', 'acknowledgment', 'swift', 'edi', '270', '271'
        ]

        code_lower = code.lower()
        found_terms = set()

        for term in business_terms:
            if term in code_lower:
                found_terms.add(term)

        return found_terms

    def calculate_similarity(
        self,
        workflow1: WorkflowSignature,
        workflow2: WorkflowSignature
    ) -> Tuple[float, float, float]:
        """
        Calculate similarity between two workflows

        Returns:
            (data_flow_similarity, logic_similarity, business_similarity)
        """
        # Data flow similarity (Jaccard similarity of tables)
        all_sources = workflow1.source_tables | workflow2.source_tables
        shared_sources = workflow1.source_tables & workflow2.source_tables

        all_targets = workflow1.target_tables | workflow2.target_tables
        shared_targets = workflow1.target_tables & workflow2.target_tables

        source_sim = len(shared_sources) / len(all_sources) if all_sources else 0
        target_sim = len(shared_targets) / len(all_targets) if all_targets else 0
        data_flow_similarity = (source_sim + target_sim) / 2

        # Logic similarity (transformation types overlap)
        all_transforms = set(workflow1.transformation_types + workflow2.transformation_types)
        shared_transforms = set(workflow1.transformation_types) & set(workflow2.transformation_types)
        logic_similarity = len(shared_transforms) / len(all_transforms) if all_transforms else 0

        # Business similarity (keyword overlap)
        all_keywords = workflow1.business_keywords | workflow2.business_keywords
        shared_keywords = workflow1.business_keywords & workflow2.business_keywords
        business_similarity = len(shared_keywords) / len(all_keywords) if all_keywords else 0

        return data_flow_similarity, logic_similarity, business_similarity

    def find_mappings(
        self,
        source_system: str,
        target_system: str,
        similarity_threshold: float = 0.3
    ) -> List[WorkflowMapping]:
        """
        Find mappings between source and target systems

        Uses similarity scoring to detect:
        - 1:1 mappings (one source → one target)
        - N:1 mappings (multiple sources → one target)
        - 1:N mappings (one source → multiple targets)
        - N:M mappings (complex relationships)
        """
        source_workflows = self.workflow_signatures.get(source_system, [])
        target_workflows = self.workflow_signatures.get(target_system, [])

        logger.info(f"Finding mappings: {source_system} ({len(source_workflows)}) → {target_system} ({len(target_workflows)})")

        # Calculate similarity matrix
        similarity_matrix = {}

        for source_wf in source_workflows:
            for target_wf in target_workflows:
                data_sim, logic_sim, business_sim = self.calculate_similarity(source_wf, target_wf)

                # Weighted average (data flow is most important)
                overall_sim = (
                    data_sim * 0.5 +
                    logic_sim * 0.3 +
                    business_sim * 0.2
                )

                if overall_sim >= similarity_threshold:
                    key = (source_wf.name, target_wf.name)
                    similarity_matrix[key] = {
                        'source': source_wf,
                        'target': target_wf,
                        'data_sim': data_sim,
                        'logic_sim': logic_sim,
                        'business_sim': business_sim,
                        'overall_sim': overall_sim
                    }

        logger.info(f"Found {len(similarity_matrix)} potential mappings above threshold {similarity_threshold}")

        # Group into workflow mappings
        mappings = self._group_into_mappings(similarity_matrix, source_system, target_system)

        self.mappings.extend(mappings)
        return mappings

    def _group_into_mappings(
        self,
        similarity_matrix: Dict,
        source_system: str,
        target_system: str
    ) -> List[WorkflowMapping]:
        """
        Group similar workflows into mappings

        Handles N:M relationships by clustering workflows that map to similar targets

        Strategy:
        1. Build graph of source→target relationships
        2. Detect clusters where multiple sources map to same target (N:1)
        3. Detect clusters where one source maps to multiple targets (1:N)
        4. Merge overlapping clusters for N:M cases
        """
        if not similarity_matrix:
            return []

        # Build mapping graph
        # source_name → {target_name: sim_data}
        source_to_targets = {}
        # target_name → {source_name: sim_data}
        target_to_sources = {}

        for (source_name, target_name), sim_data in similarity_matrix.items():
            if source_name not in source_to_targets:
                source_to_targets[source_name] = {}
            source_to_targets[source_name][target_name] = sim_data

            if target_name not in target_to_sources:
                target_to_sources[target_name] = {}
            target_to_sources[target_name][source_name] = sim_data

        logger.debug(f"Mapping graph: {len(source_to_targets)} sources → {len(target_to_sources)} targets")

        # Identify mapping patterns
        mappings = []
        processed_sources = set()
        processed_targets = set()

        # Strategy 1: Find N:1 patterns (multiple sources → one target)
        for target_name, sources_dict in target_to_sources.items():
            if target_name in processed_targets:
                continue

            if len(sources_dict) > 1:
                # Multiple sources map to this target
                logger.debug(f"Found N:1 pattern: {len(sources_dict)} sources → {target_name}")

                source_workflows = [sim_data['source'] for sim_data in sources_dict.values()]
                target_wf = list(sources_dict.values())[0]['target']

                # Calculate aggregate similarity
                avg_data_sim = sum(sd['data_sim'] for sd in sources_dict.values()) / len(sources_dict)
                avg_logic_sim = sum(sd['logic_sim'] for sd in sources_dict.values()) / len(sources_dict)
                avg_business_sim = sum(sd['business_sim'] for sd in sources_dict.values()) / len(sources_dict)
                avg_overall_sim = sum(sd['overall_sim'] for sd in sources_dict.values()) / len(sources_dict)

                mapping = WorkflowMapping(
                    source_system=source_system,
                    target_system=target_system,
                    source_workflows=source_workflows,
                    target_workflows=[target_wf],
                    data_flow_similarity=avg_data_sim,
                    logic_similarity=avg_logic_sim,
                    business_similarity=avg_business_sim,
                    overall_similarity=avg_overall_sim,
                    mapping_type=f"{len(source_workflows)}:1"
                )

                # Determine confidence
                if avg_overall_sim >= 0.7:
                    mapping.confidence = "high"
                elif avg_overall_sim >= 0.5:
                    mapping.confidence = "medium"
                else:
                    mapping.confidence = "low"

                # Analyze shared/missing tables across all sources
                all_source_tables = set()
                for src_wf in source_workflows:
                    all_source_tables.update(src_wf.target_tables)

                mapping.shared_tables = all_source_tables & target_wf.source_tables
                mapping.missing_in_target = all_source_tables - target_wf.target_tables
                mapping.added_in_target = target_wf.target_tables - all_source_tables

                # Document which sources were combined
                source_names = [sw.name for sw in source_workflows]
                mapping.transformation_differences.append(
                    f"Multiple source workflows combined: {', '.join(source_names)}"
                )

                mappings.append(mapping)
                processed_targets.add(target_name)
                processed_sources.update(sources_dict.keys())

        # Strategy 2: Find 1:N patterns (one source → multiple targets)
        for source_name, targets_dict in source_to_targets.items():
            if source_name in processed_sources:
                continue

            if len(targets_dict) > 1:
                # One source maps to multiple targets
                logger.debug(f"Found 1:N pattern: {source_name} → {len(targets_dict)} targets")

                source_wf = list(targets_dict.values())[0]['source']
                target_workflows = [sim_data['target'] for sim_data in targets_dict.values()]

                # Calculate aggregate similarity
                avg_data_sim = sum(sd['data_sim'] for sd in targets_dict.values()) / len(targets_dict)
                avg_logic_sim = sum(sd['logic_sim'] for sd in targets_dict.values()) / len(targets_dict)
                avg_business_sim = sum(sd['business_sim'] for sd in targets_dict.values()) / len(targets_dict)
                avg_overall_sim = sum(sd['overall_sim'] for sd in targets_dict.values()) / len(targets_dict)

                mapping = WorkflowMapping(
                    source_system=source_system,
                    target_system=target_system,
                    source_workflows=[source_wf],
                    target_workflows=target_workflows,
                    data_flow_similarity=avg_data_sim,
                    logic_similarity=avg_logic_sim,
                    business_similarity=avg_business_sim,
                    overall_similarity=avg_overall_sim,
                    mapping_type=f"1:{len(target_workflows)}"
                )

                # Determine confidence
                if avg_overall_sim >= 0.7:
                    mapping.confidence = "high"
                elif avg_overall_sim >= 0.5:
                    mapping.confidence = "medium"
                else:
                    mapping.confidence = "low"

                # Analyze shared/missing tables
                all_target_tables = set()
                for tgt_wf in target_workflows:
                    all_target_tables.update(tgt_wf.target_tables)

                mapping.shared_tables = source_wf.source_tables & set.union(*[tw.source_tables for tw in target_workflows])
                mapping.missing_in_target = source_wf.target_tables - all_target_tables
                mapping.added_in_target = all_target_tables - source_wf.target_tables

                # Document which targets were split into
                target_names = [tw.name for tw in target_workflows]
                mapping.transformation_differences.append(
                    f"Single source split into multiple targets: {', '.join(target_names)}"
                )

                mappings.append(mapping)
                processed_sources.add(source_name)
                processed_targets.update(targets_dict.keys())

        # Strategy 3: Remaining 1:1 mappings
        for (source_name, target_name), sim_data in similarity_matrix.items():
            if source_name in processed_sources or target_name in processed_targets:
                continue

            source_wf = sim_data['source']
            target_wf = sim_data['target']

            mapping = WorkflowMapping(
                source_system=source_system,
                target_system=target_system,
                source_workflows=[source_wf],
                target_workflows=[target_wf],
                data_flow_similarity=sim_data['data_sim'],
                logic_similarity=sim_data['logic_sim'],
                business_similarity=sim_data['business_sim'],
                overall_similarity=sim_data['overall_sim'],
                mapping_type="1:1"
            )

            # Determine confidence
            if mapping.overall_similarity >= 0.7:
                mapping.confidence = "high"
            elif mapping.overall_similarity >= 0.5:
                mapping.confidence = "medium"
            else:
                mapping.confidence = "low"

            # Analyze differences
            mapping.shared_tables = source_wf.source_tables & target_wf.source_tables
            mapping.missing_in_target = source_wf.target_tables - target_wf.target_tables
            mapping.added_in_target = target_wf.target_tables - source_wf.target_tables

            mappings.append(mapping)
            processed_sources.add(source_name)
            processed_targets.add(target_name)

        logger.info(f"Created {len(mappings)} mappings: "
                   f"{len([m for m in mappings if ':1' in m.mapping_type and m.mapping_type != '1:1'])} N:1, "
                   f"{len([m for m in mappings if '1:' in m.mapping_type and m.mapping_type != '1:1'])} 1:N, "
                   f"{len([m for m in mappings if m.mapping_type == '1:1'])} 1:1")

        return mappings

    def export_mappings(self, output_file: str):
        """Export mappings to JSON"""
        output = {
            "total_mappings": len(self.mappings),
            "mappings_by_confidence": {
                "high": len([m for m in self.mappings if m.confidence == "high"]),
                "medium": len([m for m in self.mappings if m.confidence == "medium"]),
                "low": len([m for m in self.mappings if m.confidence == "low"])
            },
            "mappings": [m.to_dict() for m in self.mappings]
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Exported {len(self.mappings)} mappings to {output_file}")


# Testing function
def test_intelligent_mapper():
    """Test the intelligent workflow mapper"""
    print("=" * 60)
    print("INTELLIGENT WORKFLOW MAPPER TEST")
    print("=" * 60)

    mapper = IntelligentWorkflowMapper()

    # Simulate Hadoop workflow signature
    hadoop_code = """
    -- Load patient accounts
    patient_accounts = LOAD '/data/patient_accounts.txt' USING PigStorage('|');

    -- Filter for commercial eligibility
    eligible = FILTER patient_accounts BY status == 'active' AND payer_type == 'commercial';

    -- Join with demographic data
    with_demo = JOIN eligible BY patient_id, demographics BY id;

    -- Store results
    STORE with_demo INTO '/output/commercial_eligibility.txt' USING PigStorage('|');
    """

    hadoop_sig = mapper.extract_signature(
        system="hadoop",
        name="ie_prebdf_commercial",
        file_path="/hadoop/app-cdd/pig/ie/commercial_eligibility.pig",
        code_content=hadoop_code
    )

    # Simulate Databricks workflow signature
    databricks_code = """
    # Read patient accounts
    patient_df = spark.read.table("patient_accounts")

    # Filter for active commercial patients
    eligible_df = patient_df.filter(
        (col("status") == "active") & (col("payer_type") == "commercial")
    )

    # Join with demographics
    demo_df = spark.table("demographics")
    result_df = eligible_df.join(demo_df, eligible_df.patient_id == demo_df.id)

    # Save results
    result_df.write.mode("overwrite").saveAsTable("commercial_eligibility")
    """

    databricks_sig = mapper.extract_signature(
        system="databricks",
        name="ie_prebdf_commercial_v2",
        file_path="/databricks/CDD/ie_prebdf/commercial_eligibility.py",
        code_content=databricks_code
    )

    # Add to mapper
    mapper.workflow_signatures["hadoop"].append(hadoop_sig)
    mapper.workflow_signatures["databricks"].append(databricks_sig)

    # Calculate similarity
    data_sim, logic_sim, business_sim = mapper.calculate_similarity(hadoop_sig, databricks_sig)

    print(f"\n📊 Similarity Analysis:")
    print(f"   Hadoop workflow: {hadoop_sig.name}")
    print(f"   Databricks workflow: {databricks_sig.name}")
    print(f"\n   Data Flow Similarity: {data_sim:.2%}")
    print(f"   Logic Similarity: {logic_sim:.2%}")
    print(f"   Business Similarity: {business_sim:.2%}")
    print(f"   Overall Similarity: {(data_sim*0.5 + logic_sim*0.3 + business_sim*0.2):.2%}")

    print(f"\n📋 Hadoop Signature:")
    print(f"   Sources: {hadoop_sig.source_file_patterns}")
    print(f"   Targets: {hadoop_sig.target_file_patterns}")
    print(f"   Transformations: {hadoop_sig.transformation_types}")
    print(f"   Business Keywords: {hadoop_sig.business_keywords}")

    print(f"\n📋 Databricks Signature:")
    print(f"   Sources: {databricks_sig.source_tables}")
    print(f"   Targets: {databricks_sig.target_tables}")
    print(f"   Transformations: {databricks_sig.transformation_types}")
    print(f"   Business Keywords: {databricks_sig.business_keywords}")

    # Find mappings
    mappings = mapper.find_mappings("hadoop", "databricks", similarity_threshold=0.3)

    print(f"\n🔗 Found {len(mappings)} mapping(s)")
    for mapping in mappings:
        print(f"\n   {mapping.source_workflows[0].name} → {mapping.target_workflows[0].name}")
        print(f"   Confidence: {mapping.confidence}")
        print(f"   Overall Similarity: {mapping.overall_similarity:.2%}")

    # Export
    mapper.export_mappings("intelligent_workflow_mappings.json")

    print("\n✅ Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_intelligent_mapper()
