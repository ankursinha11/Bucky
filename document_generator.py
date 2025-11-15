"""
Document Generation System
Generates comparison sheets, flow diagrams, and reports in multiple formats
Based on workflow mapping intelligence and user requests
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger
from datetime import datetime

from services.intelligent_workflow_mapper import WorkflowSignature, WorkflowMapping


class DocumentGenerator:
    """
    Generate documents based on workflow intelligence

    Supported formats:
    - Excel (.xlsx) - Comparison sheets, mapping tables
    - JSON (.json) - Structured data export
    - Markdown (.md) - Human-readable reports
    - CSV (.csv) - Tabular data
    """

    def __init__(self):
        """Initialize document generator"""
        self.output_dir = Path("./outputs/generated_documents")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_comparison_sheet(
        self,
        workflow1: WorkflowSignature,
        workflow2: WorkflowSignature,
        similarity_scores: Dict[str, float],
        output_format: str = "excel"
    ) -> str:
        """
        Generate side-by-side comparison sheet for two workflows

        Args:
            workflow1: First workflow signature
            workflow2: Second workflow signature
            similarity_scores: Dict with data_flow, logic, business, overall scores
            output_format: 'excel', 'json', 'markdown'

        Returns:
            Path to generated file
        """
        logger.info(f"Generating comparison sheet: {workflow1.name} vs {workflow2.name}")

        if output_format == "excel":
            return self._generate_excel_comparison(workflow1, workflow2, similarity_scores)
        elif output_format == "json":
            return self._generate_json_comparison(workflow1, workflow2, similarity_scores)
        elif output_format == "markdown":
            return self._generate_markdown_comparison(workflow1, workflow2, similarity_scores)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _generate_excel_comparison(
        self,
        workflow1: WorkflowSignature,
        workflow2: WorkflowSignature,
        similarity_scores: Dict[str, float]
    ) -> str:
        """Generate Excel comparison sheet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{workflow1.name}_vs_{workflow2.name}_{timestamp}.xlsx"
        output_path = self.output_dir / filename

        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_data = {
                "Metric": [
                    "Workflow Name",
                    "System",
                    "File Path",
                    "Data Flow Similarity",
                    "Logic Similarity",
                    "Business Similarity",
                    "Overall Similarity"
                ],
                workflow1.system.title(): [
                    workflow1.name,
                    workflow1.system,
                    workflow1.file_path,
                    "",
                    "",
                    "",
                    ""
                ],
                workflow2.system.title(): [
                    workflow2.name,
                    workflow2.system,
                    workflow2.file_path,
                    "",
                    "",
                    "",
                    ""
                ],
                "Similarity": [
                    "",
                    "",
                    "",
                    f"{similarity_scores.get('data_flow', 0):.1%}",
                    f"{similarity_scores.get('logic', 0):.1%}",
                    f"{similarity_scores.get('business', 0):.1%}",
                    f"{similarity_scores.get('overall', 0):.1%}"
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name="Summary", index=False)

            # Sheet 2: Data Flow
            max_len = max(
                len(workflow1.source_tables) + len(workflow1.target_tables),
                len(workflow2.source_tables) + len(workflow2.target_tables)
            )

            data_flow_data = {
                f"{workflow1.system.title()} - Source Tables": list(workflow1.source_tables) + [""] * (max_len - len(workflow1.source_tables)),
                f"{workflow1.system.title()} - Target Tables": list(workflow1.target_tables) + [""] * (max_len - len(workflow1.target_tables)),
                f"{workflow2.system.title()} - Source Tables": list(workflow2.source_tables) + [""] * (max_len - len(workflow2.source_tables)),
                f"{workflow2.system.title()} - Target Tables": list(workflow2.target_tables) + [""] * (max_len - len(workflow2.target_tables)),
            }
            # Trim to actual max length
            actual_max = max(
                len(list(workflow1.source_tables)),
                len(list(workflow1.target_tables)),
                len(list(workflow2.source_tables)),
                len(list(workflow2.target_tables))
            )
            for key in data_flow_data:
                data_flow_data[key] = data_flow_data[key][:actual_max]

            df_data_flow = pd.DataFrame(data_flow_data)
            df_data_flow.to_excel(writer, sheet_name="Data Flow", index=False)

            # Sheet 3: Transformations
            max_transform_len = max(len(workflow1.transformation_types), len(workflow2.transformation_types))

            transform_data = {
                f"{workflow1.system.title()} - Transformations": workflow1.transformation_types + [""] * (max_transform_len - len(workflow1.transformation_types)),
                f"{workflow2.system.title()} - Transformations": workflow2.transformation_types + [""] * (max_transform_len - len(workflow2.transformation_types))
            }
            df_transforms = pd.DataFrame(transform_data)
            df_transforms.to_excel(writer, sheet_name="Transformations", index=False)

            # Sheet 4: Business Keywords
            max_keyword_len = max(len(workflow1.business_keywords), len(workflow2.business_keywords))

            keyword_data = {
                f"{workflow1.system.title()} - Keywords": list(workflow1.business_keywords) + [""] * (max_keyword_len - len(workflow1.business_keywords)),
                f"{workflow2.system.title()} - Keywords": list(workflow2.business_keywords) + [""] * (max_keyword_len - len(workflow2.business_keywords))
            }
            df_keywords = pd.DataFrame(keyword_data)
            df_keywords.to_excel(writer, sheet_name="Business Keywords", index=False)

        logger.info(f"Excel comparison sheet generated: {output_path}")
        return str(output_path)

    def _generate_json_comparison(
        self,
        workflow1: WorkflowSignature,
        workflow2: WorkflowSignature,
        similarity_scores: Dict[str, float]
    ) -> str:
        """Generate JSON comparison"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{workflow1.name}_vs_{workflow2.name}_{timestamp}.json"
        output_path = self.output_dir / filename

        comparison_data = {
            "generated_at": datetime.now().isoformat(),
            "workflow1": workflow1.to_dict(),
            "workflow2": workflow2.to_dict(),
            "similarity_scores": similarity_scores,
            "shared_elements": {
                "source_tables": list(workflow1.source_tables & workflow2.source_tables),
                "target_tables": list(workflow1.target_tables & workflow2.target_tables),
                "transformations": list(set(workflow1.transformation_types) & set(workflow2.transformation_types)),
                "business_keywords": list(workflow1.business_keywords & workflow2.business_keywords)
            },
            "differences": {
                "workflow1_only_sources": list(workflow1.source_tables - workflow2.source_tables),
                "workflow2_only_sources": list(workflow2.source_tables - workflow1.source_tables),
                "workflow1_only_targets": list(workflow1.target_tables - workflow2.target_tables),
                "workflow2_only_targets": list(workflow2.target_tables - workflow1.target_tables)
            }
        }

        with open(output_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)

        logger.info(f"JSON comparison generated: {output_path}")
        return str(output_path)

    def _generate_markdown_comparison(
        self,
        workflow1: WorkflowSignature,
        workflow2: WorkflowSignature,
        similarity_scores: Dict[str, float]
    ) -> str:
        """Generate Markdown comparison"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{workflow1.name}_vs_{workflow2.name}_{timestamp}.md"
        output_path = self.output_dir / filename

        markdown_content = f"""# Workflow Comparison Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Summary

| Metric | {workflow1.system.title()} | {workflow2.system.title()} | Similarity |
|--------|------------|-------------|------------|
| **Workflow Name** | {workflow1.name} | {workflow2.name} | - |
| **System** | {workflow1.system} | {workflow2.system} | - |
| **File Path** | {workflow1.file_path} | {workflow2.file_path} | - |
| **Data Flow** | - | - | {similarity_scores.get('data_flow', 0):.1%} |
| **Logic** | - | - | {similarity_scores.get('logic', 0):.1%} |
| **Business** | - | - | {similarity_scores.get('business', 0):.1%} |
| **Overall** | - | - | **{similarity_scores.get('overall', 0):.1%}** |

---

## Data Flow

### Source Tables

| {workflow1.system.title()} | {workflow2.system.title()} |
|------------|-------------|
{self._format_table_rows(workflow1.source_tables, workflow2.source_tables)}

### Target Tables

| {workflow1.system.title()} | {workflow2.system.title()} |
|------------|-------------|
{self._format_table_rows(workflow1.target_tables, workflow2.target_tables)}

---

## Transformations

| {workflow1.system.title()} | {workflow2.system.title()} |
|------------|-------------|
{self._format_list_rows(workflow1.transformation_types, workflow2.transformation_types)}

---

## Business Keywords

| {workflow1.system.title()} | {workflow2.system.title()} |
|------------|-------------|
{self._format_table_rows(workflow1.business_keywords, workflow2.business_keywords)}

---

## Shared Elements

### Tables
{chr(10).join(f"- {table}" for table in (workflow1.source_tables & workflow2.source_tables)) or "- None"}

### Transformations
{chr(10).join(f"- {t}" for t in (set(workflow1.transformation_types) & set(workflow2.transformation_types))) or "- None"}

### Business Keywords
{chr(10).join(f"- {kw}" for kw in (workflow1.business_keywords & workflow2.business_keywords)) or "- None"}

---

## Differences

### Only in {workflow1.system.title()}
- **Source Tables**: {', '.join(workflow1.source_tables - workflow2.source_tables) or 'None'}
- **Target Tables**: {', '.join(workflow1.target_tables - workflow2.target_tables) or 'None'}

### Only in {workflow2.system.title()}
- **Source Tables**: {', '.join(workflow2.source_tables - workflow1.source_tables) or 'None'}
- **Target Tables**: {', '.join(workflow2.target_tables - workflow1.target_tables) or 'None'}

---

*Generated by STAG Document Generator*
"""

        with open(output_path, 'w') as f:
            f.write(markdown_content)

        logger.info(f"Markdown comparison generated: {output_path}")
        return str(output_path)

    def _format_table_rows(self, col1_data, col2_data) -> str:
        """Format two columns for markdown table"""
        list1 = list(col1_data) if col1_data else []
        list2 = list(col2_data) if col2_data else []

        max_len = max(len(list1), len(list2))
        rows = []

        for i in range(max_len):
            val1 = list1[i] if i < len(list1) else ""
            val2 = list2[i] if i < len(list2) else ""
            rows.append(f"| {val1} | {val2} |")

        return "\n".join(rows) if rows else "| - | - |"

    def _format_list_rows(self, list1, list2) -> str:
        """Format two lists for markdown table"""
        max_len = max(len(list1) if list1 else 0, len(list2) if list2 else 0)
        rows = []

        for i in range(max_len):
            val1 = list1[i] if list1 and i < len(list1) else ""
            val2 = list2[i] if list2 and i < len(list2) else ""
            rows.append(f"| {val1} | {val2} |")

        return "\n".join(rows) if rows else "| - | - |"

    def generate_mapping_report(
        self,
        mappings: List[WorkflowMapping],
        output_format: str = "excel",
        title: str = "Workflow Mapping Report"
    ) -> str:
        """
        Generate comprehensive mapping report

        Args:
            mappings: List of workflow mappings
            output_format: 'excel', 'json', 'markdown'
            title: Report title

        Returns:
            Path to generated file
        """
        logger.info(f"Generating mapping report with {len(mappings)} mappings")

        if output_format == "excel":
            return self._generate_excel_mapping_report(mappings, title)
        elif output_format == "json":
            return self._generate_json_mapping_report(mappings, title)
        elif output_format == "markdown":
            return self._generate_markdown_mapping_report(mappings, title)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _generate_excel_mapping_report(self, mappings: List[WorkflowMapping], title: str) -> str:
        """Generate Excel mapping report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mapping_report_{timestamp}.xlsx"
        output_path = self.output_dir / filename

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_data = []
            for mapping in mappings:
                source_names = ", ".join([sw.name for sw in mapping.source_workflows])
                target_names = ", ".join([tw.name for tw in mapping.target_workflows])

                summary_data.append({
                    "Source System": mapping.source_system,
                    "Target System": mapping.target_system,
                    "Source Workflows": source_names,
                    "Target Workflows": target_names,
                    "Mapping Type": mapping.mapping_type,
                    "Confidence": mapping.confidence,
                    "Data Flow Similarity": f"{mapping.data_flow_similarity:.1%}",
                    "Logic Similarity": f"{mapping.logic_similarity:.1%}",
                    "Business Similarity": f"{mapping.business_similarity:.1%}",
                    "Overall Similarity": f"{mapping.overall_similarity:.1%}"
                })

            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name="All Mappings", index=False)

            # Sheet 2: N:1 Mappings
            n_to_1 = [m for m in mappings if ":" in m.mapping_type and m.mapping_type.split(":")[0] != "1" and m.mapping_type != "1:1"]
            if n_to_1:
                n_to_1_data = []
                for mapping in n_to_1:
                    n_to_1_data.append({
                        "Source Workflows": ", ".join([sw.name for sw in mapping.source_workflows]),
                        "Target Workflow": ", ".join([tw.name for tw in mapping.target_workflows]),
                        "Count": len(mapping.source_workflows),
                        "Confidence": mapping.confidence,
                        "Similarity": f"{mapping.overall_similarity:.1%}"
                    })
                df_n_to_1 = pd.DataFrame(n_to_1_data)
                df_n_to_1.to_excel(writer, sheet_name="N-1 Consolidation", index=False)

            # Sheet 3: 1:N Mappings
            one_to_n = [m for m in mappings if ":" in m.mapping_type and m.mapping_type.split(":")[1] != "1" and m.mapping_type != "1:1"]
            if one_to_n:
                one_to_n_data = []
                for mapping in one_to_n:
                    one_to_n_data.append({
                        "Source Workflow": ", ".join([sw.name for sw in mapping.source_workflows]),
                        "Target Workflows": ", ".join([tw.name for tw in mapping.target_workflows]),
                        "Count": len(mapping.target_workflows),
                        "Confidence": mapping.confidence,
                        "Similarity": f"{mapping.overall_similarity:.1%}"
                    })
                df_one_to_n = pd.DataFrame(one_to_n_data)
                df_one_to_n.to_excel(writer, sheet_name="1-N Splitting", index=False)

        logger.info(f"Excel mapping report generated: {output_path}")
        return str(output_path)

    def _generate_json_mapping_report(self, mappings: List[WorkflowMapping], title: str) -> str:
        """Generate JSON mapping report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mapping_report_{timestamp}.json"
        output_path = self.output_dir / filename

        report_data = {
            "generated_at": datetime.now().isoformat(),
            "title": title,
            "total_mappings": len(mappings),
            "statistics": {
                "high_confidence": len([m for m in mappings if m.confidence == "high"]),
                "medium_confidence": len([m for m in mappings if m.confidence == "medium"]),
                "low_confidence": len([m for m in mappings if m.confidence == "low"]),
                "n_to_1": len([m for m in mappings if ":" in m.mapping_type and m.mapping_type.split(":")[0] != "1" and m.mapping_type != "1:1"]),
                "one_to_n": len([m for m in mappings if ":" in m.mapping_type and m.mapping_type.split(":")[1] != "1" and m.mapping_type != "1:1"]),
                "one_to_one": len([m for m in mappings if m.mapping_type == "1:1"])
            },
            "mappings": [m.to_dict() for m in mappings]
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"JSON mapping report generated: {output_path}")
        return str(output_path)

    def _generate_markdown_mapping_report(self, mappings: List[WorkflowMapping], title: str) -> str:
        """Generate Markdown mapping report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mapping_report_{timestamp}.md"
        output_path = self.output_dir / filename

        stats = {
            "total": len(mappings),
            "high": len([m for m in mappings if m.confidence == "high"]),
            "medium": len([m for m in mappings if m.confidence == "medium"]),
            "low": len([m for m in mappings if m.confidence == "low"]),
            "n_to_1": len([m for m in mappings if ":" in m.mapping_type and m.mapping_type.split(":")[0] != "1" and m.mapping_type != "1:1"]),
            "one_to_n": len([m for m in mappings if ":" in m.mapping_type and m.mapping_type.split(":")[1] != "1" and m.mapping_type != "1:1"]),
            "one_to_one": len([m for m in mappings if m.mapping_type == "1:1"])
        }

        markdown_content = f"""# {title}

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Mappings**: {stats['total']}

---

## Statistics

| Metric | Count |
|--------|-------|
| **High Confidence** | {stats['high']} |
| **Medium Confidence** | {stats['medium']} |
| **Low Confidence** | {stats['low']} |
| **N:1 (Consolidation)** | {stats['n_to_1']} |
| **1:N (Splitting)** | {stats['one_to_n']} |
| **1:1 (Direct)** | {stats['one_to_one']} |

---

## All Mappings

"""

        for i, mapping in enumerate(mappings, 1):
            source_names = ", ".join([sw.name for sw in mapping.source_workflows])
            target_names = ", ".join([tw.name for tw in mapping.target_workflows])

            markdown_content += f"""### {i}. {mapping.mapping_type} Mapping

- **Source ({mapping.source_system})**: {source_names}
- **Target ({mapping.target_system})**: {target_names}
- **Confidence**: {mapping.confidence}
- **Similarity**: {mapping.overall_similarity:.1%}

"""

        markdown_content += "\n---\n\n*Generated by STAG Document Generator*\n"

        with open(output_path, 'w') as f:
            f.write(markdown_content)

        logger.info(f"Markdown mapping report generated: {output_path}")
        return str(output_path)

    def _generate_sttm_document(
        self,
        workflow_name: str,
        sttm_mappings: List[Dict[str, Any]],
        output_format: str = "excel"
    ) -> str:
        """
        Generate STTM (Source-To-Target Mapping) document

        Args:
            workflow_name: Name of the workflow
            sttm_mappings: List of STTM mapping dictionaries
            output_format: 'excel', 'json', or 'markdown'

        Returns:
            Path to generated file
        """
        logger.info(f"Generating STTM document for {workflow_name} in {output_format} format")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_format == "excel":
            filename = f"sttm_{workflow_name}_{timestamp}.xlsx"
            output_path = self.output_dir / filename

            # Convert to DataFrame
            df_data = []
            for mapping in sttm_mappings:
                df_data.append({
                    'Target Field': mapping.get('target_field_name', 'N/A'),
                    'Target Type': mapping.get('target_field_data_type', 'N/A'),
                    'Source Fields': ', '.join(mapping.get('source_field_names', [])),
                    'Transformation Logic': mapping.get('transformation_logic', 'N/A'),
                    'Business Rule': mapping.get('business_rule', 'N/A'),
                    'Depends On': ', '.join(mapping.get('field_depends_on', []))
                })

            df = pd.DataFrame(df_data)

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='STTM Mappings', index=False)

                # Auto-adjust column widths
                worksheet = writer.sheets['STTM Mappings']
                for idx, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).apply(len).max(),
                        len(col)
                    )
                    worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)

            logger.info(f"Excel STTM document generated: {output_path}")
            return str(output_path)

        elif output_format == "json":
            filename = f"sttm_{workflow_name}_{timestamp}.json"
            output_path = self.output_dir / filename

            output_data = {
                'workflow_name': workflow_name,
                'generated_at': timestamp,
                'total_mappings': len(sttm_mappings),
                'mappings': sttm_mappings
            }

            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"JSON STTM document generated: {output_path}")
            return str(output_path)

        elif output_format == "markdown":
            filename = f"sttm_{workflow_name}_{timestamp}.md"
            output_path = self.output_dir / filename

            markdown_content = f"""# STTM Mappings: {workflow_name}

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Mappings**: {len(sttm_mappings)}

---

## Column-Level Mappings

"""

            for i, mapping in enumerate(sttm_mappings, 1):
                markdown_content += f"""### {i}. {mapping.get('target_field_name', 'Unknown Field')}

- **Target Type**: `{mapping.get('target_field_data_type', 'N/A')}`
- **Source Fields**: {', '.join([f'`{s}`' for s in mapping.get('source_field_names', ['N/A'])])}
- **Transformation Logic**: {mapping.get('transformation_logic', 'N/A')}
- **Business Rule**: {mapping.get('business_rule', 'N/A')}
- **Dependencies**: {', '.join([f'`{d}`' for d in mapping.get('field_depends_on', ['None'])])}

"""

            markdown_content += "\n---\n\n*Generated by STAG Document Generator*\n"

            with open(output_path, 'w') as f:
                f.write(markdown_content)

            logger.info(f"Markdown STTM document generated: {output_path}")
            return str(output_path)

        else:
            raise ValueError(f"Unsupported format: {output_format}")


# Testing function
def test_document_generator():
    """Test document generator"""
    print("=" * 60)
    print("DOCUMENT GENERATOR TEST")
    print("=" * 60)

    generator = DocumentGenerator()

    # Create test workflows
    wf1 = WorkflowSignature(
        system="hadoop",
        name="ie_prebdf_commercial",
        file_path="/hadoop/ie_prebdf.pig"
    )
    wf1.source_tables = {"patient_accounts", "demographics"}
    wf1.target_tables = {"commercial_eligibility"}
    wf1.transformation_types = ["filter", "join"]
    wf1.business_keywords = {"patient", "eligibility", "commercial"}

    wf2 = WorkflowSignature(
        system="databricks",
        name="ie_prebdf_commercial_v2",
        file_path="/databricks/ie_prebdf.py"
    )
    wf2.source_tables = {"patient_accounts", "demographics"}
    wf2.target_tables = {"commercial_eligibility"}
    wf2.transformation_types = ["filter", "join"]
    wf2.business_keywords = {"patient", "eligibility", "commercial"}

    similarity_scores = {
        "data_flow": 0.0,
        "logic": 1.0,
        "business": 1.0,
        "overall": 0.5
    }

    # Generate comparison sheet in all formats
    print("\n📄 Generating comparison sheets...")
    excel_path = generator.generate_comparison_sheet(wf1, wf2, similarity_scores, output_format="excel")
    print(f"✅ Excel: {excel_path}")

    json_path = generator.generate_comparison_sheet(wf1, wf2, similarity_scores, output_format="json")
    print(f"✅ JSON: {json_path}")

    md_path = generator.generate_comparison_sheet(wf1, wf2, similarity_scores, output_format="markdown")
    print(f"✅ Markdown: {md_path}")

    print("\n✅ Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_document_generator()
