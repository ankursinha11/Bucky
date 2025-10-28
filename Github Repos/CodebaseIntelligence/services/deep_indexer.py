"""
Deep Hierarchical Indexer
Indexes all 3 tiers for intelligent search
"""

from typing import List, Dict, Any
from loguru import logger

from core.models import Repository, WorkflowFlow
from core.models.script_logic import ScriptLogic
from services.local_search.local_search_client import LocalSearchClient


class DeepIndexer:
    """
    Index all 3 tiers of parsed data

    Tier 1: Repository documents (high-level overview)
    Tier 2: Workflow flow documents (mid-level flow)
    Tier 3: Script logic documents (deep logic + AI analysis)
    """

    def __init__(self, vector_db_path: str = "./outputs/vector_db"):
        """Initialize deep indexer"""
        self.search_client = LocalSearchClient(persist_directory=vector_db_path)
        self.indexed_count = 0

        logger.info("🗄️ Deep Hierarchical Indexer initialized")

    def index_deep_analysis(
        self,
        repository: Repository = None,
        workflow_flows: List[WorkflowFlow] = None,
        script_logics: List[ScriptLogic] = None,
    ) -> Dict[str, int]:
        """
        Index all tiers

        Args:
            repository: Tier 1 repository
            workflow_flows: Tier 2 workflows
            script_logics: Tier 3 scripts

        Returns:
            Dict with counts by tier
        """
        logger.info("📊 Indexing hierarchical structure...")

        documents = []
        counts = {
            "tier1_repository": 0,
            "tier2_workflows": 0,
            "tier3_scripts": 0,
            "tier3_transformations": 0,
            "tier3_lineage": 0,
        }

        # Index Tier 1: Repository
        if repository:
            repo_doc = self._create_repository_document(repository)
            documents.append(repo_doc)
            counts["tier1_repository"] = 1
            logger.info(f"  ✓ Tier 1 (Repository): {repository.name}")

        # Index Tier 2: Workflow Flows
        if workflow_flows:
            for workflow in workflow_flows:
                workflow_doc = self._create_workflow_document(workflow)
                documents.append(workflow_doc)

                # Also index flow diagram as separate document
                if workflow.flow_diagram_ascii or workflow.flow_diagram_mermaid:
                    flow_diagram_doc = self._create_flow_diagram_document(workflow)
                    documents.append(flow_diagram_doc)

            counts["tier2_workflows"] = len(workflow_flows)
            logger.info(f"  ✓ Tier 2 (Workflows): {len(workflow_flows)} workflows")

        # Index Tier 3: Script Logic
        if script_logics:
            for script in script_logics:
                # Main script document
                script_doc = self._create_script_document(script)
                documents.append(script_doc)

                # Index transformations separately for granular search
                for trans in script.transformations:
                    trans_doc = self._create_transformation_document(script, trans)
                    documents.append(trans_doc)
                    counts["tier3_transformations"] += 1

                # Index column lineages separately
                if script.column_lineages:
                    lineage_doc = self._create_lineage_document(script)
                    documents.append(lineage_doc)
                    counts["tier3_lineage"] += 1

            counts["tier3_scripts"] = len(script_logics)
            logger.info(f"  ✓ Tier 3 (Scripts): {len(script_logics)} scripts")
            logger.info(f"  ✓ Tier 3 (Transformations): {counts['tier3_transformations']} transformations")
            logger.info(f"  ✓ Tier 3 (Lineage): {counts['tier3_lineage']} lineage documents")

        # Index all documents
        if documents:
            logger.info(f"📤 Indexing {len(documents)} total documents...")
            self.search_client.index_documents(documents)
            self.indexed_count = len(documents)
            logger.info(f"✅ Indexed {self.indexed_count} documents successfully")

        return counts

    def _create_repository_document(self, repository: Repository) -> Dict[str, Any]:
        """Create Tier 1 repository document"""
        return {
            "id": repository.id,
            "content": repository.get_searchable_content(),
            "doc_type": "tier1_repository",
            "system": repository.repo_type.value,
            "title": f"Repository: {repository.name}",
            "metadata": {
                "tier": 1,
                "repository_name": repository.name,
                "total_workflows": repository.total_workflows,
                "total_scripts": repository.total_scripts,
                "pig_scripts": repository.pig_scripts,
                "spark_scripts": repository.spark_scripts,
                "business_domains": ",".join(repository.business_domains),
                "technologies": ",".join(repository.technologies),
                "ai_analyzed": bool(repository.ai_summary),
            }
        }

    def _create_workflow_document(self, workflow: WorkflowFlow) -> Dict[str, Any]:
        """Create Tier 2 workflow document"""
        return {
            "id": workflow.workflow_id,
            "content": workflow.get_searchable_content(),
            "doc_type": "tier2_workflow",
            "system": "hadoop",
            "title": f"Workflow: {workflow.workflow_name}",
            "metadata": {
                "tier": 2,
                "workflow_name": workflow.workflow_name,
                "workflow_type": workflow.workflow_type,
                "repository_id": workflow.repository_id,
                "total_actions": len(workflow.actions),
                "business_domain": workflow.business_domain or "",
                "ai_analyzed": bool(workflow.ai_flow_summary),
            }
        }

    def _create_flow_diagram_document(self, workflow: WorkflowFlow) -> Dict[str, Any]:
        """Create separate document for flow diagram (for visual search)"""
        content_parts = [
            f"Flow Diagram for Workflow: {workflow.workflow_name}",
            "",
        ]

        if workflow.flow_diagram_ascii:
            content_parts.append("ASCII Flow:")
            content_parts.append(workflow.flow_diagram_ascii)
            content_parts.append("")

        if workflow.flow_diagram_mermaid:
            content_parts.append("Mermaid Diagram:")
            content_parts.append(workflow.flow_diagram_mermaid)

        return {
            "id": f"{workflow.workflow_id}_flow_diagram",
            "content": "\n".join(content_parts),
            "doc_type": "tier2_flow_diagram",
            "system": "hadoop",
            "title": f"Flow: {workflow.workflow_name}",
            "metadata": {
                "tier": 2,
                "workflow_id": workflow.workflow_id,
                "workflow_name": workflow.workflow_name,
            }
        }

    def _create_script_document(self, script: ScriptLogic) -> Dict[str, Any]:
        """Create Tier 3 script document"""
        return {
            "id": script.script_id,
            "content": script.get_searchable_content(),
            "doc_type": "tier3_script",
            "system": script.script_type,
            "title": f"Script: {script.script_name}",
            "metadata": {
                "tier": 3,
                "script_name": script.script_name,
                "script_type": script.script_type,
                "workflow_id": script.workflow_id or "",
                "action_id": script.action_id or "",
                "total_transformations": len(script.transformations),
                "total_lineages": len(script.column_lineages),
                "ai_analyzed": bool(script.ai_logic_summary),
                "has_business_purpose": bool(script.business_purpose),
            }
        }

    def _create_transformation_document(self, script: ScriptLogic, transformation) -> Dict[str, Any]:
        """Create document for individual transformation (very granular!)"""
        content_parts = [
            f"Transformation: {transformation.transformation_type.value}",
            f"Script: {script.script_name}",
            f"",
            f"Code:",
            transformation.code_snippet,
            "",
        ]

        if transformation.condition:
            content_parts.append(f"Condition: {transformation.condition}")

        if transformation.columns:
            content_parts.append(f"Columns: {', '.join(transformation.columns)}")

        if transformation.business_meaning:
            content_parts.append(f"Business Meaning: {transformation.business_meaning}")

        if transformation.join_type:
            content_parts.append(f"Join Type: {transformation.join_type}")
            if transformation.join_keys:
                content_parts.append(f"Join Keys: {', '.join(transformation.join_keys)}")

        if transformation.group_by_columns:
            content_parts.append(f"Group By: {', '.join(transformation.group_by_columns)}")

        return {
            "id": transformation.transformation_id,
            "content": "\n".join(content_parts),
            "doc_type": "tier3_transformation",
            "system": script.script_type,
            "title": f"Transform: {transformation.transformation_type.value} in {script.script_name}",
            "metadata": {
                "tier": 3,
                "transformation_type": transformation.transformation_type.value,
                "script_id": script.script_id,
                "script_name": script.script_name,
                "workflow_id": script.workflow_id or "",
                "has_business_meaning": bool(transformation.business_meaning),
            }
        }

    def _create_lineage_document(self, script: ScriptLogic) -> Dict[str, Any]:
        """Create document for column lineages"""
        content_parts = [
            f"Column Lineage Map for: {script.script_name}",
            f"Script Type: {script.script_type}",
            "",
            f"Column Mappings ({len(script.column_lineages)}):",
            "",
        ]

        for lineage in script.column_lineages:
            arrow = "→"
            if lineage.is_pass_through:
                content_parts.append(f"  {lineage.source_table}.{lineage.source_column} {arrow} {lineage.target_table}.{lineage.target_column} (pass-through)")
            elif lineage.is_calculated:
                content_parts.append(f"  {lineage.source_table}.{lineage.source_column} {arrow} {lineage.target_table}.{lineage.target_column} (calculated)")
                if lineage.calculation_logic:
                    content_parts.append(f"    Calculation: {lineage.calculation_logic}")
            elif lineage.is_aggregated:
                content_parts.append(f"  {lineage.source_table}.{lineage.source_column} {arrow} {lineage.target_table}.{lineage.target_column} (aggregated)")
                if lineage.transformation_logic:
                    content_parts.append(f"    Transform: {lineage.transformation_logic}")
            else:
                content_parts.append(f"  {lineage.source_table}.{lineage.source_column} {arrow} {lineage.target_table}.{lineage.target_column}")

        return {
            "id": f"{script.script_id}_lineage",
            "content": "\n".join(content_parts),
            "doc_type": "tier3_lineage",
            "system": script.script_type,
            "title": f"Lineage: {script.script_name}",
            "metadata": {
                "tier": 3,
                "script_id": script.script_id,
                "script_name": script.script_name,
                "total_lineages": len(script.column_lineages),
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        search_stats = self.search_client.get_stats()
        return {
            "vector_db": search_stats,
            "indexed_by_this_session": self.indexed_count
        }
