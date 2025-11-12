"""
Workflow Integration System
Integrates existing analysis data with intelligent workflow mapper
Runs complete migration validation across all systems
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from services.intelligent_workflow_mapper import IntelligentWorkflowMapper, WorkflowSignature
from services.migration_validator import MigrationValidator


class WorkflowIntegration:
    """
    Integrates workflow analysis across all systems

    Workflow:
    1. Load existing Databricks analysis (databricks_pipeline_analysis.json)
    2. Scan Hadoop repository and extract signatures
    3. Load Ab Initio mappings
    4. Run intelligent mapper
    5. Generate validation reports
    """

    def __init__(self):
        """Initialize integration"""
        self.mapper = IntelligentWorkflowMapper()
        self.validator = MigrationValidator(self.mapper)

    def load_databricks_analysis(self, analysis_file: str = "databricks_pipeline_analysis.json") -> int:
        """
        Load existing Databricks analysis and convert to signatures

        Returns number of signatures loaded
        """
        logger.info(f"Loading Databricks analysis from {analysis_file}")

        try:
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)

            pipelines = analysis_data.get('pipelines', {})
            signatures_loaded = 0

            for pipeline_path, pipeline_data in pipelines.items():
                notebooks = pipeline_data.get('notebooks', [])

                for notebook in notebooks:
                    # Create signature from notebook data
                    signature = WorkflowSignature(
                        system="databricks",
                        name=notebook.get('name', ''),
                        file_path=notebook.get('path', '')
                    )

                    # Extract data flow
                    signature.source_tables = set(notebook.get('input_tables', []))
                    signature.target_tables = set(notebook.get('output_tables', []))
                    signature.source_file_patterns = set(notebook.get('input_paths', []))
                    signature.target_file_patterns = set(notebook.get('output_paths', []))

                    # Extract transformations
                    operations = notebook.get('key_operations', [])
                    if 'filter' in operations or 'where' in operations:
                        signature.transformation_types.append('filter')
                    if 'join' in operations:
                        signature.transformation_types.append('join')
                    if 'groupBy' in operations or 'group_by' in operations:
                        signature.transformation_types.append('group_by')

                    # Extract business keywords from name and path
                    text_to_analyze = f"{signature.name} {signature.file_path}".lower()
                    signature.business_keywords = self.mapper._extract_business_keywords(text_to_analyze)

                    # Add to mapper
                    self.mapper.workflow_signatures["databricks"].append(signature)
                    signatures_loaded += 1

            logger.info(f"Loaded {signatures_loaded} Databricks workflow signatures")
            return signatures_loaded

        except FileNotFoundError:
            logger.warning(f"Databricks analysis file not found: {analysis_file}")
            return 0
        except Exception as e:
            logger.error(f"Failed to load Databricks analysis: {e}")
            return 0

    def scan_hadoop_repository(
        self,
        repository_path: str,
        file_limit: Optional[int] = None
    ) -> int:
        """
        Scan Hadoop repository and extract workflow signatures

        Returns number of signatures extracted
        """
        logger.info(f"Scanning Hadoop repository: {repository_path}")

        repo_path = Path(repository_path)
        if not repo_path.exists():
            logger.error(f"Hadoop repository not found: {repository_path}")
            return 0

        # Find Pig, Hive, and Python files
        pig_files = list(repo_path.rglob("*.pig"))
        hql_files = list(repo_path.rglob("*.hql")) + list(repo_path.rglob("*.sql"))
        py_files = list(repo_path.rglob("*.py"))

        all_files = pig_files + hql_files + py_files

        if file_limit:
            all_files = all_files[:file_limit]

        logger.info(f"Found {len(pig_files)} Pig, {len(hql_files)} HQL, {len(py_files)} Python files")

        signatures_extracted = 0

        for file_path in all_files:
            try:
                # Read file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Skip empty files
                if len(content.strip()) < 100:
                    continue

                # Extract signature
                signature = self.mapper.extract_signature(
                    system="hadoop",
                    name=file_path.stem,
                    file_path=str(file_path),
                    code_content=content
                )

                # Only add if it has meaningful content
                if signature.source_tables or signature.target_tables or signature.source_file_patterns or signature.target_file_patterns:
                    self.mapper.workflow_signatures["hadoop"].append(signature)
                    signatures_extracted += 1

            except Exception as e:
                logger.debug(f"Could not process {file_path.name}: {e}")

        logger.info(f"Extracted {signatures_extracted} Hadoop workflow signatures")
        return signatures_extracted

    def load_abinitio_mappings(
        self,
        mapping_file: str = "abinitio_graph_mappings.json"
    ) -> int:
        """
        Load Ab Initio mappings and convert to signatures

        Returns number of signatures loaded
        """
        logger.info(f"Loading Ab Initio mappings from {mapping_file}")

        try:
            with open(mapping_file, 'r') as f:
                mappings_data = json.load(f)

            graphs = mappings_data.get('graphs', {})
            signatures_loaded = 0

            for graph_name, graph_data in graphs.items():
                # Create signature from graph data
                signature = WorkflowSignature(
                    system="abinitio",
                    name=graph_name,
                    file_path=graph_name  # No actual file path for Ab Initio
                )

                # Extract inputs
                for inp in graph_data.get('inputs', []):
                    file_path = inp.get('file_path', '')
                    if file_path:
                        signature.source_file_patterns.add(file_path)

                # Extract outputs
                for out in graph_data.get('outputs', []):
                    file_path = out.get('file_path', '')
                    if file_path:
                        signature.target_file_patterns.add(file_path)

                # Extract transformations
                transformations = graph_data.get('transformations', [])
                if transformations:
                    signature.transformation_types.extend(transformations)

                # Extract business keywords from graph name and summary
                text_to_analyze = f"{graph_name} {graph_data.get('summary', '')}".lower()
                signature.business_keywords = self.mapper._extract_business_keywords(text_to_analyze)

                # Add to mapper
                self.mapper.workflow_signatures["abinitio"].append(signature)
                signatures_loaded += 1

            logger.info(f"Loaded {signatures_loaded} Ab Initio workflow signatures")
            return signatures_loaded

        except FileNotFoundError:
            logger.warning(f"Ab Initio mappings file not found: {mapping_file}")
            return 0
        except Exception as e:
            logger.error(f"Failed to load Ab Initio mappings: {e}")
            return 0

    def run_complete_analysis(
        self,
        hadoop_repo_path: str,
        databricks_analysis_file: str = "databricks_pipeline_analysis.json",
        abinitio_mappings_file: str = "abinitio_graph_mappings.json",
        similarity_threshold: float = 0.3,
        output_dir: str = "./outputs"
    ):
        """
        Run complete workflow analysis and validation

        Steps:
        1. Load all workflow signatures
        2. Run Hadoop→Databricks mapping
        3. Run Databricks↔Ab Initio correlation
        4. Generate validation reports
        5. Export results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("COMPLETE WORKFLOW ANALYSIS")
        print("=" * 70)

        # Step 1: Load workflow signatures
        print("\n📂 STEP 1: Loading workflow signatures...")

        databricks_count = self.load_databricks_analysis(databricks_analysis_file)
        hadoop_count = self.scan_hadoop_repository(hadoop_repo_path, file_limit=100)  # Limit for testing
        abinitio_count = self.load_abinitio_mappings(abinitio_mappings_file)

        print(f"\n   Loaded signatures:")
        print(f"   - Databricks: {databricks_count}")
        print(f"   - Hadoop: {hadoop_count}")
        print(f"   - Ab Initio: {abinitio_count}")

        if hadoop_count == 0 or databricks_count == 0:
            print("\n⚠️ Insufficient data for analysis")
            return

        # Step 2: Hadoop→Databricks migration validation
        print("\n🔍 STEP 2: Validating Hadoop→Databricks migration...")

        hadoop_to_databricks_report = self.validator.validate_migration(
            "hadoop",
            "databricks",
            similarity_threshold=similarity_threshold
        )

        # Print summary
        self.validator.print_report_summary(hadoop_to_databricks_report)

        # Export report
        report_file = output_path / "hadoop_to_databricks_validation.json"
        self.validator.export_report(hadoop_to_databricks_report, str(report_file))

        # Step 3: Databricks↔Ab Initio correlation (if Ab Initio data available)
        if abinitio_count > 0:
            print("\n🔍 STEP 3: Analyzing Databricks↔Ab Initio correlation...")

            databricks_to_abinitio_report = self.validator.validate_migration(
                "databricks",
                "abinitio",
                similarity_threshold=similarity_threshold
            )

            # Print summary
            self.validator.print_report_summary(databricks_to_abinitio_report)

            # Export report
            report_file = output_path / "databricks_to_abinitio_correlation.json"
            self.validator.export_report(databricks_to_abinitio_report, str(report_file))

        # Step 4: Export all mappings
        print("\n📊 STEP 4: Exporting workflow mappings...")

        mappings_file = output_path / "workflow_mappings.json"
        self.mapper.export_mappings(str(mappings_file))

        print(f"\n✅ Analysis complete!")
        print(f"\n📁 Reports saved to: {output_dir}")
        print(f"   - hadoop_to_databricks_validation.json")
        if abinitio_count > 0:
            print(f"   - databricks_to_abinitio_correlation.json")
        print(f"   - workflow_mappings.json")

        print("\n" + "=" * 70)

    def get_mapping_for_workflow(
        self,
        workflow_name: str,
        source_system: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all mappings for a specific workflow

        Useful for answering: "What Databricks pipeline replaced Hadoop workflow X?"
        """
        matching_mappings = []

        for mapping in self.mapper.mappings:
            # Check source workflows
            for src_wf in mapping.source_workflows:
                if workflow_name.lower() in src_wf.name.lower():
                    if source_system is None or src_wf.system == source_system:
                        matching_mappings.append({
                            "source": [sw.name for sw in mapping.source_workflows],
                            "target": [tw.name for tw in mapping.target_workflows],
                            "source_system": mapping.source_system,
                            "target_system": mapping.target_system,
                            "mapping_type": mapping.mapping_type,
                            "confidence": mapping.confidence,
                            "similarity": mapping.overall_similarity
                        })
                        break

            # Check target workflows
            for tgt_wf in mapping.target_workflows:
                if workflow_name.lower() in tgt_wf.name.lower():
                    if source_system is None or tgt_wf.system == source_system:
                        matching_mappings.append({
                            "source": [sw.name for sw in mapping.source_workflows],
                            "target": [tw.name for tw in mapping.target_workflows],
                            "source_system": mapping.source_system,
                            "target_system": mapping.target_system,
                            "mapping_type": mapping.mapping_type,
                            "confidence": mapping.confidence,
                            "similarity": mapping.overall_similarity
                        })
                        break

        return matching_mappings


# Testing function
def test_workflow_integration():
    """Test workflow integration system"""
    print("=" * 60)
    print("WORKFLOW INTEGRATION TEST")
    print("=" * 60)

    integration = WorkflowIntegration()

    # Run complete analysis
    integration.run_complete_analysis(
        hadoop_repo_path="/Users/ankurshome/Desktop/Hadoop_Parser/CodebaseIntelligence/hadoop_repos/hadoop_repos",
        databricks_analysis_file="databricks_pipeline_analysis.json",
        abinitio_mappings_file="abinitio_graph_mappings.json",
        similarity_threshold=0.3,
        output_dir="./outputs"
    )

    print("\n✅ Integration test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_workflow_integration()
