"""
Migration Validation System
Validates Hadoop→Databricks migration completeness and Ab Initio correlation
Uses intelligent workflow mapping to detect gaps and issues
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
from loguru import logger

from services.intelligent_workflow_mapper import (
    IntelligentWorkflowMapper,
    WorkflowSignature,
    WorkflowMapping
)


@dataclass
class MigrationGap:
    """Represents a gap in migration (unmapped workflow)"""
    system: str
    workflow_name: str
    file_path: str
    gap_type: str  # "unmapped_source", "unmapped_target", "partial_mapping"
    details: str
    severity: str  # "critical", "high", "medium", "low"


@dataclass
class MigrationValidationReport:
    """Complete migration validation report"""
    source_system: str
    target_system: str

    # Summary stats
    total_source_workflows: int = 0
    total_target_workflows: int = 0
    mapped_source_workflows: int = 0
    mapped_target_workflows: int = 0
    unmapped_source_workflows: int = 0
    unmapped_target_workflows: int = 0

    # Mappings breakdown
    high_confidence_mappings: int = 0
    medium_confidence_mappings: int = 0
    low_confidence_mappings: int = 0

    # Mapping types
    one_to_one_mappings: int = 0
    n_to_one_mappings: int = 0
    one_to_n_mappings: int = 0

    # Gaps and issues
    gaps: List[MigrationGap] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "source_system": self.source_system,
            "target_system": self.target_system,
            "summary": {
                "total_source_workflows": self.total_source_workflows,
                "total_target_workflows": self.total_target_workflows,
                "mapped_source": self.mapped_source_workflows,
                "mapped_target": self.mapped_target_workflows,
                "unmapped_source": self.unmapped_source_workflows,
                "unmapped_target": self.unmapped_target_workflows,
                "migration_completeness": f"{(self.mapped_source_workflows/self.total_source_workflows*100) if self.total_source_workflows else 0:.1f}%"
            },
            "mappings": {
                "high_confidence": self.high_confidence_mappings,
                "medium_confidence": self.medium_confidence_mappings,
                "low_confidence": self.low_confidence_mappings,
                "one_to_one": self.one_to_one_mappings,
                "n_to_one": self.n_to_one_mappings,
                "one_to_n": self.one_to_n_mappings
            },
            "gaps": [
                {
                    "system": gap.system,
                    "workflow": gap.workflow_name,
                    "file_path": gap.file_path,
                    "type": gap.gap_type,
                    "severity": gap.severity,
                    "details": gap.details
                }
                for gap in self.gaps
            ],
            "recommendations": self.recommendations
        }


class MigrationValidator:
    """
    Validates migration between systems

    Primary use cases:
    1. Hadoop→Databricks: Validate 2-year-old migration is complete
    2. Databricks↔Ab Initio: Understand correlation with running Ab Initio
    """

    def __init__(self, mapper: IntelligentWorkflowMapper):
        """Initialize validator with a workflow mapper"""
        self.mapper = mapper

    def validate_migration(
        self,
        source_system: str,
        target_system: str,
        similarity_threshold: float = 0.3
    ) -> MigrationValidationReport:
        """
        Validate migration from source to target system

        Returns comprehensive report with gaps and recommendations
        """
        logger.info(f"Validating migration: {source_system} → {target_system}")

        report = MigrationValidationReport(
            source_system=source_system,
            target_system=target_system
        )

        # Get workflow counts
        source_workflows = self.mapper.workflow_signatures.get(source_system, [])
        target_workflows = self.mapper.workflow_signatures.get(target_system, [])

        report.total_source_workflows = len(source_workflows)
        report.total_target_workflows = len(target_workflows)

        if not source_workflows:
            logger.warning(f"No {source_system} workflows found")
            return report

        if not target_workflows:
            logger.warning(f"No {target_system} workflows found")
            return report

        # Find mappings
        mappings = self.mapper.find_mappings(
            source_system,
            target_system,
            similarity_threshold=similarity_threshold
        )

        # Analyze mappings
        mapped_sources = set()
        mapped_targets = set()

        for mapping in mappings:
            # Count by confidence
            if mapping.confidence == "high":
                report.high_confidence_mappings += 1
            elif mapping.confidence == "medium":
                report.medium_confidence_mappings += 1
            elif mapping.confidence == "low":
                report.low_confidence_mappings += 1

            # Count by type
            if mapping.mapping_type == "1:1":
                report.one_to_one_mappings += 1
            elif ":1" in mapping.mapping_type and mapping.mapping_type != "1:1":
                report.n_to_one_mappings += 1
            elif "1:" in mapping.mapping_type and mapping.mapping_type != "1:1":
                report.one_to_n_mappings += 1

            # Track mapped workflows
            for src_wf in mapping.source_workflows:
                mapped_sources.add(src_wf.name)
            for tgt_wf in mapping.target_workflows:
                mapped_targets.add(tgt_wf.name)

        report.mapped_source_workflows = len(mapped_sources)
        report.mapped_target_workflows = len(mapped_targets)
        report.unmapped_source_workflows = report.total_source_workflows - report.mapped_source_workflows
        report.unmapped_target_workflows = report.total_target_workflows - report.mapped_target_workflows

        # Identify gaps
        report.gaps = self._identify_gaps(
            source_workflows,
            target_workflows,
            mapped_sources,
            mapped_targets,
            mappings
        )

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report, mappings)

        logger.info(f"Validation complete: {report.mapped_source_workflows}/{report.total_source_workflows} source workflows mapped")
        logger.info(f"Found {len(report.gaps)} gaps")

        return report

    def _identify_gaps(
        self,
        source_workflows: List[WorkflowSignature],
        target_workflows: List[WorkflowSignature],
        mapped_sources: Set[str],
        mapped_targets: Set[str],
        mappings: List[WorkflowMapping]
    ) -> List[MigrationGap]:
        """Identify gaps in migration"""
        gaps = []

        # Unmapped source workflows (CRITICAL - these weren't migrated)
        for src_wf in source_workflows:
            if src_wf.name not in mapped_sources:
                # Determine severity based on business keywords
                severity = "high"
                if any(keyword in src_wf.business_keywords for keyword in ['patient', 'eligibility', 'commercial', 'medicaid']):
                    severity = "critical"
                elif not src_wf.business_keywords:
                    severity = "medium"

                gaps.append(MigrationGap(
                    system=src_wf.system,
                    workflow_name=src_wf.name,
                    file_path=src_wf.file_path,
                    gap_type="unmapped_source",
                    details=f"Source workflow not found in target system. Tables: {', '.join(list(src_wf.target_tables)[:5])}",
                    severity=severity
                ))

        # Unmapped target workflows (INFO - these are new in target)
        for tgt_wf in target_workflows:
            if tgt_wf.name not in mapped_targets:
                gaps.append(MigrationGap(
                    system=tgt_wf.system,
                    workflow_name=tgt_wf.name,
                    file_path=tgt_wf.file_path,
                    gap_type="unmapped_target",
                    details=f"Target workflow has no clear source equivalent. This may be a new workflow. Tables: {', '.join(list(tgt_wf.target_tables)[:5])}",
                    severity="low"
                ))

        # Partial mappings (low confidence)
        for mapping in mappings:
            if mapping.confidence == "low":
                source_names = [sw.name for sw in mapping.source_workflows]
                target_names = [tw.name for tw in mapping.target_workflows]

                gaps.append(MigrationGap(
                    system="mapping",
                    workflow_name=f"{', '.join(source_names)} → {', '.join(target_names)}",
                    file_path="",
                    gap_type="partial_mapping",
                    details=f"Low confidence mapping (similarity: {mapping.overall_similarity:.1%}). May need manual review.",
                    severity="medium"
                ))

        # Sort gaps by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        gaps.sort(key=lambda g: severity_order.get(g.severity, 99))

        return gaps

    def _generate_recommendations(
        self,
        report: MigrationValidationReport,
        mappings: List[WorkflowMapping]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Migration completeness
        completeness = (report.mapped_source_workflows / report.total_source_workflows * 100) if report.total_source_workflows else 0

        if completeness < 70:
            recommendations.append(
                f"⚠️ Migration is only {completeness:.1f}% complete. "
                f"{report.unmapped_source_workflows} source workflows have no target equivalent. "
                "Review unmapped workflows to determine if they should be migrated."
            )
        elif completeness < 90:
            recommendations.append(
                f"✓ Migration is {completeness:.1f}% complete. "
                f"Review {report.unmapped_source_workflows} remaining unmapped workflows."
            )
        else:
            recommendations.append(
                f"✅ Migration is {completeness:.1f}% complete. Excellent coverage!"
            )

        # Low confidence mappings
        if report.low_confidence_mappings > 0:
            recommendations.append(
                f"📊 Found {report.low_confidence_mappings} low-confidence mappings. "
                "These may need manual validation to ensure correctness."
            )

        # N:1 patterns (consolidation)
        if report.n_to_one_mappings > 0:
            recommendations.append(
                f"🔀 Found {report.n_to_one_mappings} N:1 mappings (multiple sources → one target). "
                "This indicates workflow consolidation - verify logic was properly merged."
            )

        # 1:N patterns (splitting)
        if report.one_to_n_mappings > 0:
            recommendations.append(
                f"🔀 Found {report.one_to_n_mappings} 1:N mappings (one source → multiple targets). "
                "This indicates workflow splitting - verify all functionality was preserved."
            )

        # Critical gaps
        critical_gaps = [g for g in report.gaps if g.severity == "critical"]
        if critical_gaps:
            recommendations.append(
                f"🚨 CRITICAL: {len(critical_gaps)} unmapped workflows contain business-critical keywords. "
                "These should be prioritized for investigation."
            )

        # Data completeness
        missing_tables = []
        for mapping in mappings:
            if mapping.missing_in_target:
                missing_tables.extend(mapping.missing_in_target)

        if missing_tables:
            unique_missing = set(missing_tables)
            recommendations.append(
                f"📊 {len(unique_missing)} tables from source workflows are not found in target outputs. "
                "Verify these tables are not needed or are handled differently."
            )

        return recommendations

    def export_report(self, report: MigrationValidationReport, output_file: str):
        """Export validation report to JSON"""
        with open(output_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Exported validation report to {output_file}")

    def print_report_summary(self, report: MigrationValidationReport):
        """Print human-readable report summary"""
        print("\n" + "=" * 70)
        print(f"MIGRATION VALIDATION REPORT: {report.source_system} → {report.target_system}")
        print("=" * 70)

        print(f"\n📊 SUMMARY")
        print(f"   Source workflows: {report.total_source_workflows}")
        print(f"   Target workflows: {report.total_target_workflows}")
        print(f"   Mapped source: {report.mapped_source_workflows} ({report.mapped_source_workflows/report.total_source_workflows*100:.1f}%)")
        print(f"   Mapped target: {report.mapped_target_workflows} ({report.mapped_target_workflows/report.total_target_workflows*100:.1f}%)")
        print(f"   Unmapped source: {report.unmapped_source_workflows}")
        print(f"   Unmapped target: {report.unmapped_target_workflows}")

        print(f"\n🔗 MAPPINGS")
        print(f"   High confidence: {report.high_confidence_mappings}")
        print(f"   Medium confidence: {report.medium_confidence_mappings}")
        print(f"   Low confidence: {report.low_confidence_mappings}")
        print(f"   1:1 mappings: {report.one_to_one_mappings}")
        print(f"   N:1 mappings: {report.n_to_one_mappings}")
        print(f"   1:N mappings: {report.one_to_n_mappings}")

        print(f"\n⚠️ GAPS ({len(report.gaps)})")

        # Group by severity
        critical = [g for g in report.gaps if g.severity == "critical"]
        high = [g for g in report.gaps if g.severity == "high"]
        medium = [g for g in report.gaps if g.severity == "medium"]
        low = [g for g in report.gaps if g.severity == "low"]

        if critical:
            print(f"\n   🚨 CRITICAL ({len(critical)})")
            for gap in critical[:5]:  # Show first 5
                print(f"      - {gap.workflow_name}")
                print(f"        {gap.details[:100]}")

        if high:
            print(f"\n   ⚠️ HIGH ({len(high)})")
            for gap in high[:5]:
                print(f"      - {gap.workflow_name}")

        if medium:
            print(f"\n   ℹ️ MEDIUM ({len(medium)})")
            for gap in medium[:3]:
                print(f"      - {gap.workflow_name}")

        if low:
            print(f"\n   ℹ️ LOW ({len(low)})")

        print(f"\n💡 RECOMMENDATIONS")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"\n   {i}. {rec}")

        print("\n" + "=" * 70)


# Testing function
def test_migration_validator():
    """Test migration validator"""
    from services.intelligent_workflow_mapper import IntelligentWorkflowMapper

    print("=" * 60)
    print("MIGRATION VALIDATOR TEST")
    print("=" * 60)

    # Create mapper
    mapper = IntelligentWorkflowMapper()

    # Simulate some workflows (in real usage, these come from actual code analysis)
    # Hadoop workflow 1
    hadoop_sig1 = WorkflowSignature(
        system="hadoop",
        name="ie_prebdf_commercial",
        file_path="/hadoop/app-cdd/pig/ie/commercial.pig"
    )
    hadoop_sig1.source_tables = {"patient_accounts", "demographics"}
    hadoop_sig1.target_tables = {"commercial_eligibility"}
    hadoop_sig1.transformation_types = ["filter", "join"]
    hadoop_sig1.business_keywords = {"patient", "eligibility", "commercial"}

    # Hadoop workflow 2 (unmapped)
    hadoop_sig2 = WorkflowSignature(
        system="hadoop",
        name="ie_prebdf_medicaid",
        file_path="/hadoop/app-cdd/pig/ie/medicaid.pig"
    )
    hadoop_sig2.source_tables = {"patient_accounts", "coverage"}
    hadoop_sig2.target_tables = {"medicaid_eligibility"}
    hadoop_sig2.transformation_types = ["filter", "join"]
    hadoop_sig2.business_keywords = {"patient", "eligibility", "medicaid"}

    # Databricks workflow (maps to hadoop workflow 1)
    databricks_sig1 = WorkflowSignature(
        system="databricks",
        name="ie_prebdf_commercial_v2",
        file_path="/databricks/CDD/ie_prebdf/commercial.py"
    )
    databricks_sig1.source_tables = {"patient_accounts", "demographics"}
    databricks_sig1.target_tables = {"commercial_eligibility"}
    databricks_sig1.transformation_types = ["filter", "join"]
    databricks_sig1.business_keywords = {"patient", "eligibility", "commercial"}

    # Add to mapper
    mapper.workflow_signatures["hadoop"] = [hadoop_sig1, hadoop_sig2]
    mapper.workflow_signatures["databricks"] = [databricks_sig1]

    # Validate migration
    validator = MigrationValidator(mapper)
    report = validator.validate_migration("hadoop", "databricks", similarity_threshold=0.3)

    # Print report
    validator.print_report_summary(report)

    # Export report
    validator.export_report(report, "migration_validation_report.json")

    print("\n✅ Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_migration_validator()
