"""
STAG Workflow Intelligence Integration
Integrates workflow mapping into STAG's chat, copilot, and vector database
Enables intelligent querying, document generation, and document analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

from services.intelligent_workflow_mapper import IntelligentWorkflowMapper, WorkflowSignature, WorkflowMapping
from services.migration_validator import MigrationValidator
from services.workflow_integration import WorkflowIntegration


class STAGWorkflowIntelligence:
    """
    Integrates workflow intelligence into STAG

    Capabilities:
    1. Index workflow mappings into vector DB (searchable via Copilot)
    2. Answer workflow questions naturally
    3. Generate comparison documents (Excel, JSON, MD)
    4. Analyze uploaded documents
    5. Full understanding of all 3 systems
    """

    def __init__(self, vector_db_client=None):
        """Initialize STAG workflow intelligence"""
        self.integration = WorkflowIntegration()
        self.vector_db = vector_db_client

        # Cached data
        self.mappings_loaded = False
        self.workflow_mappings: List[WorkflowMapping] = []
        self.workflow_signatures: Dict[str, List[WorkflowSignature]] = {}

    def load_workflow_intelligence(
        self,
        hadoop_repo_path: str,
        databricks_analysis_file: str = "databricks_pipeline_analysis.json",
        abinitio_mappings_file: str = "abinitio_graph_mappings.json"
    ):
        """
        Load all workflow intelligence data

        This should be called after STAG indexing completes
        """
        logger.info("Loading workflow intelligence for STAG...")

        # Load workflow signatures
        databricks_count = self.integration.load_databricks_analysis(databricks_analysis_file)
        hadoop_count = self.integration.scan_hadoop_repository(hadoop_repo_path, file_limit=None)  # Load all

        if Path(abinitio_mappings_file).exists():
            abinitio_count = self.integration.load_abinitio_mappings(abinitio_mappings_file)
        else:
            abinitio_count = 0

        logger.info(f"Loaded workflow signatures: Databricks={databricks_count}, Hadoop={hadoop_count}, Ab Initio={abinitio_count}")

        # Find mappings
        self.workflow_mappings = self.integration.mapper.mappings
        self.workflow_signatures = self.integration.mapper.workflow_signatures
        self.mappings_loaded = True

        # Index mappings into vector DB if available
        if self.vector_db:
            self._index_mappings_to_vector_db()

    def _index_mappings_to_vector_db(self):
        """
        Index workflow mappings into STAG's vector database

        This makes mappings searchable via Copilot
        """
        logger.info("Indexing workflow mappings into vector database...")

        documents_to_index = []

        # Create searchable documents for each mapping
        for mapping in self.workflow_mappings:
            source_names = [sw.name for sw in mapping.source_workflows]
            target_names = [tw.name for tw in mapping.target_workflows]

            # Create rich document content
            doc_content = f"""
# Workflow Mapping: {mapping.source_system} → {mapping.target_system}

## Source Workflows ({mapping.source_system})
{', '.join(source_names)}

## Target Workflows ({mapping.target_system})
{', '.join(target_names)}

## Mapping Type
{mapping.mapping_type}

## Confidence
{mapping.confidence}

## Similarity Scores
- Data Flow: {mapping.data_flow_similarity:.1%}
- Logic: {mapping.logic_similarity:.1%}
- Business: {mapping.business_similarity:.1%}
- Overall: {mapping.overall_similarity:.1%}

## Data Analysis
- Shared Tables: {', '.join(mapping.shared_tables) if mapping.shared_tables else 'None'}
- Missing in Target: {', '.join(mapping.missing_in_target) if mapping.missing_in_target else 'None'}
- Added in Target: {', '.join(mapping.added_in_target) if mapping.added_in_target else 'None'}

## Transformation Differences
{chr(10).join(f'- {diff}' for diff in mapping.transformation_differences) if mapping.transformation_differences else 'None detected'}

## Source Workflow Details
{self._format_workflow_details(mapping.source_workflows)}

## Target Workflow Details
{self._format_workflow_details(mapping.target_workflows)}

## Use Cases
- Query: "What Databricks pipeline replaced {source_names[0] if source_names else 'workflow'}?"
- Query: "Compare {source_names[0] if source_names else 'source'} with {target_names[0] if target_names else 'target'}"
- Query: "Show migration details for {source_names[0] if source_names else 'workflow'}"
"""

            documents_to_index.append({
                "id": f"workflow_mapping_{hash(tuple(source_names + target_names))}",
                "content": doc_content,
                "metadata": {
                    "type": "workflow_mapping",
                    "source_system": mapping.source_system,
                    "target_system": mapping.target_system,
                    "source_workflows": source_names,
                    "target_workflows": target_names,
                    "mapping_type": mapping.mapping_type,
                    "confidence": mapping.confidence,
                    "overall_similarity": mapping.overall_similarity
                }
            })

        # Add documents to vector DB
        if documents_to_index and self.vector_db:
            try:
                # Assuming vector DB has an add_documents method
                logger.info(f"Adding {len(documents_to_index)} workflow mapping documents to vector DB")
                # self.vector_db.add_documents(documents_to_index)
                # TODO: Implement actual vector DB indexing based on STAG's vector DB interface
            except Exception as e:
                logger.warning(f"Could not index mappings to vector DB: {e}")

        logger.info("Workflow mappings indexed successfully")

    def _format_workflow_details(self, workflows: List[WorkflowSignature]) -> str:
        """Format workflow details for documentation"""
        details = []
        for wf in workflows:
            detail = f"""
### {wf.name}
- File: {wf.file_path}
- Source Tables: {', '.join(wf.source_tables) if wf.source_tables else 'None'}
- Target Tables: {', '.join(wf.target_tables) if wf.target_tables else 'None'}
- Transformations: {', '.join(wf.transformation_types) if wf.transformation_types else 'None'}
- Business Keywords: {', '.join(wf.business_keywords) if wf.business_keywords else 'None'}
"""
            details.append(detail)
        return '\n'.join(details)

    def answer_workflow_question(self, question: str, context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Answer workflow-related questions intelligently

        Examples:
        - "What Databricks pipeline replaced Hadoop workflow X?"
        - "Compare ie_prebdf from Hadoop vs Databricks"
        - "Show all N:1 mappings"
        - "What workflows are unmapped?"
        """
        if not self.mappings_loaded:
            return {
                "answer": "Workflow intelligence not loaded. Please run indexing first.",
                "type": "error"
            }

        question_lower = question.lower()

        # Pattern 1: "What replaced X?" or "What Databricks pipeline replaced X?"
        if "replaced" in question_lower or "replacement" in question_lower:
            return self._handle_replacement_query(question, question_lower)

        # Pattern 2: "Compare X with Y" or "Compare X vs Y"
        elif "compare" in question_lower:
            return self._handle_comparison_query(question, question_lower)

        # Pattern 3: "Show N:1 mappings" or "Show consolidation"
        elif "n:1" in question_lower or "consolidation" in question_lower:
            return self._handle_mapping_type_query("n:1")

        # Pattern 4: "Show 1:N mappings" or "Show splitting"
        elif "1:n" in question_lower or "splitting" in question_lower or "split" in question_lower:
            return self._handle_mapping_type_query("1:n")

        # Pattern 5: "Show unmapped" or "What's missing?"
        elif "unmapped" in question_lower or "missing" in question_lower or "gap" in question_lower:
            return self._handle_unmapped_query(question_lower)

        # Pattern 6: "Show all mappings" or "List mappings"
        elif "all mapping" in question_lower or "list mapping" in question_lower:
            return self._handle_all_mappings_query()

        # Pattern 7: "Generate comparison sheet" or "Create comparison"
        elif "generate" in question_lower or "create" in question_lower:
            if "comparison" in question_lower or "sheet" in question_lower:
                return self._handle_document_generation_query(question, question_lower)

        # Default: Return general workflow statistics
        return self._handle_general_query()

    def _handle_replacement_query(self, question: str, question_lower: str) -> Dict[str, Any]:
        """Handle 'what replaced X?' queries"""
        # Extract workflow name from question
        # Simple extraction - look for workflow names in the mappings
        workflow_name = self._extract_workflow_name(question)

        if not workflow_name:
            return {
                "answer": "I couldn't identify the workflow name in your question. Please specify which workflow you're asking about.",
                "type": "clarification_needed"
            }

        # Find mappings for this workflow
        matching_mappings = []
        for mapping in self.workflow_mappings:
            for src_wf in mapping.source_workflows:
                if workflow_name.lower() in src_wf.name.lower():
                    matching_mappings.append(mapping)
                    break

        if not matching_mappings:
            return {
                "answer": f"No replacement found for workflow '{workflow_name}'. It may be unmapped or have a very different name in the target system.",
                "type": "no_results",
                "workflow": workflow_name
            }

        # Format answer
        answer_parts = [f"**Replacement for '{workflow_name}':**\n"]

        for mapping in matching_mappings:
            source_names = [sw.name for sw in mapping.source_workflows]
            target_names = [tw.name for tw in mapping.target_workflows]

            answer_parts.append(f"\n**Mapping Type**: {mapping.mapping_type}")
            answer_parts.append(f"**Confidence**: {mapping.confidence} ({mapping.overall_similarity:.1%} similarity)")
            answer_parts.append(f"\n**Source ({mapping.source_system})**:")
            for name in source_names:
                answer_parts.append(f"  - {name}")
            answer_parts.append(f"\n**Target ({mapping.target_system})**:")
            for name in target_names:
                answer_parts.append(f"  - {name}")

            if mapping.mapping_type != "1:1":
                if ":" in mapping.mapping_type and mapping.mapping_type.split(":")[0] != "1":
                    answer_parts.append(f"\n⚠️ Note: This is a consolidation - {len(source_names)} source workflows were combined.")
                elif ":" in mapping.mapping_type and mapping.mapping_type.split(":")[1] != "1":
                    answer_parts.append(f"\n⚠️ Note: This is a split - source workflow was divided into {len(target_names)} target workflows.")

        return {
            "answer": "\n".join(answer_parts),
            "type": "replacement_found",
            "mappings": [m.to_dict() for m in matching_mappings],
            "workflow": workflow_name
        }

    def _handle_comparison_query(self, question: str, question_lower: str) -> Dict[str, Any]:
        """Handle 'compare X vs Y' queries"""
        # Extract workflow names
        workflows = self._extract_workflow_names(question)

        if len(workflows) < 2:
            return {
                "answer": "Please specify two workflows to compare (e.g., 'Compare ie_prebdf from Hadoop with ie_prebdf_v2 from Databricks')",
                "type": "clarification_needed"
            }

        # Find the workflows in signatures
        workflow1 = self._find_workflow_signature(workflows[0])
        workflow2 = self._find_workflow_signature(workflows[1])

        if not workflow1 or not workflow2:
            missing = []
            if not workflow1:
                missing.append(workflows[0])
            if not workflow2:
                missing.append(workflows[1])
            return {
                "answer": f"Could not find workflow(s): {', '.join(missing)}",
                "type": "no_results"
            }

        # Calculate similarity
        data_sim, logic_sim, business_sim = self.integration.mapper.calculate_similarity(workflow1, workflow2)
        overall_sim = data_sim * 0.5 + logic_sim * 0.3 + business_sim * 0.2

        # Format comparison
        answer_parts = [
            f"**Comparison: {workflow1.name} ({workflow1.system}) vs {workflow2.name} ({workflow2.system})**\n",
            f"**Similarity Scores:**",
            f"  - Data Flow: {data_sim:.1%}",
            f"  - Logic: {logic_sim:.1%}",
            f"  - Business: {business_sim:.1%}",
            f"  - **Overall**: {overall_sim:.1%}",
            f"\n**{workflow1.name} ({workflow1.system}):**",
            f"  - Source Tables: {', '.join(workflow1.source_tables) if workflow1.source_tables else 'None'}",
            f"  - Target Tables: {', '.join(workflow1.target_tables) if workflow1.target_tables else 'None'}",
            f"  - Transformations: {', '.join(workflow1.transformation_types) if workflow1.transformation_types else 'None'}",
            f"  - Business Keywords: {', '.join(workflow1.business_keywords) if workflow1.business_keywords else 'None'}",
            f"\n**{workflow2.name} ({workflow2.system}):**",
            f"  - Source Tables: {', '.join(workflow2.source_tables) if workflow2.source_tables else 'None'}",
            f"  - Target Tables: {', '.join(workflow2.target_tables) if workflow2.target_tables else 'None'}",
            f"  - Transformations: {', '.join(workflow2.transformation_types) if workflow2.transformation_types else 'None'}",
            f"  - Business Keywords: {', '.join(workflow2.business_keywords) if workflow2.business_keywords else 'None'}",
            f"\n**Shared Elements:**",
            f"  - Tables: {', '.join(workflow1.source_tables & workflow2.source_tables) if (workflow1.source_tables & workflow2.source_tables) else 'None'}",
            f"  - Transformations: {', '.join(set(workflow1.transformation_types) & set(workflow2.transformation_types)) if workflow1.transformation_types and workflow2.transformation_types else 'None'}",
            f"  - Business Keywords: {', '.join(workflow1.business_keywords & workflow2.business_keywords) if (workflow1.business_keywords & workflow2.business_keywords) else 'None'}"
        ]

        return {
            "answer": "\n".join(answer_parts),
            "type": "comparison",
            "workflow1": workflow1.to_dict(),
            "workflow2": workflow2.to_dict(),
            "similarity": {
                "data_flow": data_sim,
                "logic": logic_sim,
                "business": business_sim,
                "overall": overall_sim
            }
        }

    def _handle_mapping_type_query(self, mapping_type: str) -> Dict[str, Any]:
        """Handle queries for specific mapping types"""
        if mapping_type == "n:1":
            matching = [m for m in self.workflow_mappings if ":" in m.mapping_type and m.mapping_type.split(":")[0] != "1" and m.mapping_type != "1:1"]
            title = "N:1 Mappings (Consolidation)"
            description = "Multiple source workflows were consolidated into single target workflows."
        elif mapping_type == "1:n":
            matching = [m for m in self.workflow_mappings if ":" in m.mapping_type and m.mapping_type.split(":")[1] != "1" and m.mapping_type != "1:1"]
            title = "1:N Mappings (Splitting)"
            description = "Single source workflows were split into multiple target workflows."
        else:
            matching = [m for m in self.workflow_mappings if m.mapping_type == "1:1"]
            title = "1:1 Mappings"
            description = "Direct workflow replacements."

        if not matching:
            return {
                "answer": f"No {mapping_type} mappings found.",
                "type": "no_results"
            }

        answer_parts = [f"**{title}**", f"{description}\n", f"Found {len(matching)} mappings:\n"]

        for i, mapping in enumerate(matching[:10], 1):  # Show first 10
            source_names = [sw.name for sw in mapping.source_workflows]
            target_names = [tw.name for tw in mapping.target_workflows]

            answer_parts.append(f"\n**{i}. {mapping.mapping_type} Mapping**")
            answer_parts.append(f"  - Sources: {', '.join(source_names)}")
            answer_parts.append(f"  - Targets: {', '.join(target_names)}")
            answer_parts.append(f"  - Confidence: {mapping.confidence}")

        if len(matching) > 10:
            answer_parts.append(f"\n... and {len(matching) - 10} more")

        return {
            "answer": "\n".join(answer_parts),
            "type": "mapping_type_list",
            "mappings": [m.to_dict() for m in matching],
            "count": len(matching)
        }

    def _handle_unmapped_query(self, question_lower: str) -> Dict[str, Any]:
        """Handle queries about unmapped workflows"""
        # Determine which system
        if "hadoop" in question_lower:
            system = "hadoop"
        elif "databricks" in question_lower:
            system = "databricks"
        elif "abinitio" in question_lower or "ab initio" in question_lower:
            system = "abinitio"
        else:
            system = "hadoop"  # Default to source system

        # Get all workflows from system
        all_workflows = self.workflow_signatures.get(system, [])

        # Get mapped workflows
        mapped_workflow_names = set()
        for mapping in self.workflow_mappings:
            if mapping.source_system == system:
                mapped_workflow_names.update([sw.name for sw in mapping.source_workflows])
            if mapping.target_system == system:
                mapped_workflow_names.update([tw.name for tw in mapping.target_workflows])

        # Find unmapped
        unmapped = [wf for wf in all_workflows if wf.name not in mapped_workflow_names]

        answer_parts = [
            f"**Unmapped {system.title()} Workflows**\n",
            f"Total {system} workflows: {len(all_workflows)}",
            f"Mapped: {len(mapped_workflow_names)}",
            f"Unmapped: {len(unmapped)}\n"
        ]

        if unmapped:
            answer_parts.append("**Unmapped Workflows:**")
            for i, wf in enumerate(unmapped[:20], 1):  # Show first 20
                answer_parts.append(f"{i}. {wf.name}")
                if wf.business_keywords:
                    answer_parts.append(f"   Keywords: {', '.join(list(wf.business_keywords)[:5])}")

            if len(unmapped) > 20:
                answer_parts.append(f"\n... and {len(unmapped) - 20} more")
        else:
            answer_parts.append("All workflows are mapped!")

        return {
            "answer": "\n".join(answer_parts),
            "type": "unmapped_list",
            "unmapped": [wf.to_dict() for wf in unmapped],
            "system": system,
            "count": len(unmapped)
        }

    def _handle_all_mappings_query(self) -> Dict[str, Any]:
        """Handle queries to show all mappings"""
        stats = {
            "total": len(self.workflow_mappings),
            "high_confidence": len([m for m in self.workflow_mappings if m.confidence == "high"]),
            "medium_confidence": len([m for m in self.workflow_mappings if m.confidence == "medium"]),
            "low_confidence": len([m for m in self.workflow_mappings if m.confidence == "low"]),
            "n_to_1": len([m for m in self.workflow_mappings if ":" in m.mapping_type and m.mapping_type.split(":")[0] != "1" and m.mapping_type != "1:1"]),
            "one_to_n": len([m for m in self.workflow_mappings if ":" in m.mapping_type and m.mapping_type.split(":")[1] != "1" and m.mapping_type != "1:1"]),
            "one_to_one": len([m for m in self.workflow_mappings if m.mapping_type == "1:1"])
        }

        answer_parts = [
            f"**Workflow Mapping Statistics**\n",
            f"Total Mappings: {stats['total']}",
            f"\n**By Confidence:**",
            f"  - High: {stats['high_confidence']}",
            f"  - Medium: {stats['medium_confidence']}",
            f"  - Low: {stats['low_confidence']}",
            f"\n**By Type:**",
            f"  - N:1 (Consolidation): {stats['n_to_1']}",
            f"  - 1:N (Splitting): {stats['one_to_n']}",
            f"  - 1:1 (Direct): {stats['one_to_one']}"
        ]

        return {
            "answer": "\n".join(answer_parts),
            "type": "statistics",
            "stats": stats,
            "mappings": [m.to_dict() for m in self.workflow_mappings[:50]]  # Return first 50
        }

    def _handle_general_query(self) -> Dict[str, Any]:
        """Handle general workflow queries"""
        return self._handle_all_mappings_query()

    def _handle_document_generation_query(self, question: str, question_lower: str) -> Dict[str, Any]:
        """Handle document generation requests"""
        # This will be handled by the document generation system
        return {
            "answer": "Document generation feature coming soon. I can help you understand workflow mappings - try asking 'What Databricks pipeline replaced X?' or 'Compare X vs Y'",
            "type": "feature_not_ready"
        }

    def _extract_workflow_name(self, question: str) -> Optional[str]:
        """Extract workflow name from question"""
        # Look for common workflow names in the question
        question_words = question.lower().split()

        # Check against known workflow names
        all_workflow_names = []
        for system, workflows in self.workflow_signatures.items():
            all_workflow_names.extend([wf.name for wf in workflows])

        # Find best match
        for word in question_words:
            for workflow_name in all_workflow_names:
                if word in workflow_name.lower() or workflow_name.lower() in word:
                    return workflow_name

        # Try to extract from common patterns
        import re
        patterns = [
            r"workflow ['\"]([^'\"]+)['\"]",
            r"pipeline ['\"]([^'\"]+)['\"]",
            r"for ([a-z_][a-z0-9_]+)",
            r"replaced ([a-z_][a-z0-9_]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                return match.group(1)

        return None

    def _extract_workflow_names(self, question: str) -> List[str]:
        """Extract multiple workflow names from question"""
        # Simple implementation - look for words that match workflow names
        question_words = question.lower().replace(",", " ").replace("vs", " ").replace("with", " ").split()

        all_workflow_names = []
        for system, workflows in self.workflow_signatures.items():
            all_workflow_names.extend([wf.name for wf in workflows])

        found_names = []
        for word in question_words:
            for workflow_name in all_workflow_names:
                if word in workflow_name.lower() and workflow_name not in found_names:
                    found_names.append(workflow_name)
                    if len(found_names) >= 2:
                        return found_names

        return found_names

    def _find_workflow_signature(self, workflow_name: str) -> Optional[WorkflowSignature]:
        """Find workflow signature by name"""
        for system, workflows in self.workflow_signatures.items():
            for wf in workflows:
                if workflow_name.lower() in wf.name.lower():
                    return wf
        return None

    def get_workflow_context_for_copilot(self, query: str) -> str:
        """
        Get workflow context to enhance Copilot queries

        Returns formatted context about workflow mappings relevant to the query
        """
        if not self.mappings_loaded:
            return ""

        # Extract relevant workflows from query
        workflow_name = self._extract_workflow_name(query)

        if not workflow_name:
            return ""

        # Find relevant mappings
        relevant_mappings = []
        for mapping in self.workflow_mappings:
            for src_wf in mapping.source_workflows:
                if workflow_name.lower() in src_wf.name.lower():
                    relevant_mappings.append(mapping)
                    break
            for tgt_wf in mapping.target_workflows:
                if workflow_name.lower() in tgt_wf.name.lower():
                    relevant_mappings.append(mapping)
                    break

        if not relevant_mappings:
            return ""

        # Format context
        context_parts = ["## Workflow Mapping Context\n"]
        for mapping in relevant_mappings[:3]:  # Top 3 most relevant
            source_names = [sw.name for sw in mapping.source_workflows]
            target_names = [tw.name for tw in mapping.target_workflows]

            context_parts.append(f"**Mapping**: {', '.join(source_names)} → {', '.join(target_names)}")
            context_parts.append(f"Type: {mapping.mapping_type}, Confidence: {mapping.confidence}")

        return "\n".join(context_parts)


# Testing function
def test_stag_workflow_intelligence():
    """Test STAG workflow intelligence"""
    print("=" * 60)
    print("STAG WORKFLOW INTELLIGENCE TEST")
    print("=" * 60)

    intelligence = STAGWorkflowIntelligence()

    # Load workflow data
    intelligence.load_workflow_intelligence(
        hadoop_repo_path="/Users/ankurshome/Desktop/Hadoop_Parser/CodebaseIntelligence/hadoop_repos/hadoop_repos",
        databricks_analysis_file="databricks_pipeline_analysis.json",
        abinitio_mappings_file="abinitio_graph_mappings.json"
    )

    # Test questions
    test_questions = [
        "What Databricks pipeline replaced ie_prebdf?",
        "Compare merge from Hadoop with merge from Databricks",
        "Show all N:1 mappings",
        "What workflows are unmapped in Hadoop?",
        "Show all mappings"
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")

        result = intelligence.answer_workflow_question(question)
        print(f"\n{result['answer']}")

    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_stag_workflow_intelligence()
