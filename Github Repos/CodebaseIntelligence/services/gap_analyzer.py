"""
Smart Gap Analyzer
==================
Automated cross-system comparison and gap detection

Features:
- Compare implementations across Hadoop, Databricks, and Ab Initio
- Detect missing workflows and scripts
- Identify transformation differences
- Find data quality gaps
- Suggest migration paths
- Generate detailed comparison reports
"""

import os
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from core.models import (
    Repository, WorkflowFlow, ScriptLogic,
    TransformationType, RepositoryType
)


@dataclass
class Gap:
    """Represents a detected gap between systems"""
    gap_id: str
    gap_type: str  # missing_workflow, missing_transformation, data_quality, etc.
    severity: str  # critical, high, medium, low
    source_system: str
    target_system: str
    title: str
    description: str
    recommendation: str
    affected_workflows: List[str] = field(default_factory=list)
    affected_scripts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Complete comparison report between systems"""
    report_id: str
    source_system: str
    target_system: str
    gaps: List[Gap]
    similarities: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    recommendations: List[str]
    migration_complexity: str  # low, medium, high, very_high


class SmartGapAnalyzer:
    """
    Smart analyzer for detecting gaps and differences across systems
    """

    def __init__(self, use_ai: bool = True):
        """
        Initialize gap analyzer

        Args:
            use_ai: Whether to use AI for enhanced analysis
        """
        self.use_ai = use_ai
        self.ai_analyzer = None

        if use_ai:
            try:
                from services.ai_script_analyzer import AIScriptAnalyzer
                self.ai_analyzer = AIScriptAnalyzer()
            except Exception as e:
                logger.warning(f"AI analyzer not available: {e}")
                self.use_ai = False

    def compare_systems(
        self,
        source_data: Dict[str, Any],
        target_data: Dict[str, Any],
        source_name: str,
        target_name: str
    ) -> ComparisonReport:
        """
        Compare two system implementations

        Args:
            source_data: Source system parsed data (repository, workflows, scripts)
            target_data: Target system parsed data
            source_name: Name of source system
            target_name: Name of target system

        Returns:
            ComparisonReport with detected gaps and recommendations
        """
        logger.info(f"🔍 Comparing {source_name} → {target_name}")

        # Extract data
        source_repo = source_data.get("repository")
        source_workflows = source_data.get("workflow_flows", [])
        source_scripts = source_data.get("script_logics", [])

        target_repo = target_data.get("repository")
        target_workflows = target_data.get("workflow_flows", [])
        target_scripts = target_data.get("script_logics", [])

        # Detect gaps
        gaps = []

        # 1. Workflow-level gaps
        workflow_gaps = self._detect_workflow_gaps(
            source_workflows, target_workflows, source_name, target_name
        )
        gaps.extend(workflow_gaps)

        # 2. Transformation-level gaps
        transformation_gaps = self._detect_transformation_gaps(
            source_scripts, target_scripts, source_name, target_name
        )
        gaps.extend(transformation_gaps)

        # 3. Data quality gaps
        quality_gaps = self._detect_data_quality_gaps(
            source_scripts, target_scripts, source_name, target_name
        )
        gaps.extend(quality_gaps)

        # 4. Business logic gaps
        logic_gaps = self._detect_business_logic_gaps(
            source_scripts, target_scripts, source_name, target_name
        )
        gaps.extend(logic_gaps)

        # 5. Input/Output gaps
        io_gaps = self._detect_io_gaps(
            source_scripts, target_scripts, source_name, target_name
        )
        gaps.extend(io_gaps)

        # Find similarities
        similarities = self._find_similarities(
            source_workflows, target_workflows,
            source_scripts, target_scripts
        )

        # Calculate statistics
        statistics = self._calculate_statistics(
            source_repo, target_repo,
            source_workflows, target_workflows,
            source_scripts, target_scripts,
            gaps
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(gaps, statistics)

        # Assess migration complexity
        complexity = self._assess_migration_complexity(gaps, statistics)

        report = ComparisonReport(
            report_id=f"comparison_{source_name}_{target_name}",
            source_system=source_name,
            target_system=target_name,
            gaps=gaps,
            similarities=similarities,
            statistics=statistics,
            recommendations=recommendations,
            migration_complexity=complexity
        )

        logger.info(f"✓ Comparison complete: {len(gaps)} gaps detected")

        return report

    def _detect_workflow_gaps(
        self,
        source_workflows: List[WorkflowFlow],
        target_workflows: List[WorkflowFlow],
        source_name: str,
        target_name: str
    ) -> List[Gap]:
        """Detect missing or different workflows"""
        gaps = []

        # Build workflow name sets
        source_names = {self._normalize_name(wf.workflow_name) for wf in source_workflows}
        target_names = {self._normalize_name(wf.workflow_name) for wf in target_workflows}

        # Find missing workflows
        missing_in_target = source_names - target_names

        for missing_name in missing_in_target:
            # Find original workflow
            original_wf = next(
                (wf for wf in source_workflows if self._normalize_name(wf.workflow_name) == missing_name),
                None
            )

            if original_wf:
                gap = Gap(
                    gap_id=f"workflow_missing_{missing_name}",
                    gap_type="missing_workflow",
                    severity="high",
                    source_system=source_name,
                    target_system=target_name,
                    title=f"Missing workflow: {original_wf.workflow_name}",
                    description=f"Workflow '{original_wf.workflow_name}' exists in {source_name} but not in {target_name}",
                    recommendation=f"Consider implementing equivalent workflow in {target_name}",
                    affected_workflows=[original_wf.workflow_name],
                    metadata={
                        "action_count": len(original_wf.actions),
                        "workflow_id": original_wf.workflow_id
                    }
                )
                gaps.append(gap)

        # Check for structural differences in matching workflows
        for source_wf in source_workflows:
            normalized_source = self._normalize_name(source_wf.workflow_name)

            # Find matching target workflow
            matching_target = next(
                (wf for wf in target_workflows if self._normalize_name(wf.workflow_name) == normalized_source),
                None
            )

            if matching_target:
                # Compare action counts
                if len(source_wf.actions) != len(matching_target.actions):
                    gap = Gap(
                        gap_id=f"workflow_diff_{normalized_source}",
                        gap_type="workflow_structure_difference",
                        severity="medium",
                        source_system=source_name,
                        target_system=target_name,
                        title=f"Structural difference: {source_wf.workflow_name}",
                        description=f"Workflow has {len(source_wf.actions)} actions in {source_name} vs {len(matching_target.actions)} in {target_name}",
                        recommendation="Review workflow implementations for completeness",
                        affected_workflows=[source_wf.workflow_name, matching_target.workflow_name],
                        metadata={
                            "source_actions": len(source_wf.actions),
                            "target_actions": len(matching_target.actions)
                        }
                    )
                    gaps.append(gap)

        return gaps

    def _detect_transformation_gaps(
        self,
        source_scripts: List[ScriptLogic],
        target_scripts: List[ScriptLogic],
        source_name: str,
        target_name: str
    ) -> List[Gap]:
        """Detect missing or different transformations"""
        gaps = []

        # Count transformation types in each system
        source_trans_counts = self._count_transformation_types(source_scripts)
        target_trans_counts = self._count_transformation_types(target_scripts)

        # Find missing transformation types
        for trans_type, source_count in source_trans_counts.items():
            target_count = target_trans_counts.get(trans_type, 0)

            if target_count == 0:
                gap = Gap(
                    gap_id=f"transformation_missing_{trans_type}",
                    gap_type="missing_transformation_type",
                    severity="high",
                    source_system=source_name,
                    target_system=target_name,
                    title=f"Missing transformation: {trans_type}",
                    description=f"{trans_type} transformations ({source_count} instances) found in {source_name} but not in {target_name}",
                    recommendation=f"Implement {trans_type} transformations in {target_name}",
                    metadata={
                        "transformation_type": trans_type,
                        "source_count": source_count
                    }
                )
                gaps.append(gap)

            elif target_count < source_count * 0.5:  # Less than 50% coverage
                gap = Gap(
                    gap_id=f"transformation_incomplete_{trans_type}",
                    gap_type="incomplete_transformation_coverage",
                    severity="medium",
                    source_system=source_name,
                    target_system=target_name,
                    title=f"Incomplete coverage: {trans_type}",
                    description=f"{trans_type}: {source_count} in {source_name} vs {target_count} in {target_name}",
                    recommendation=f"Review and complete {trans_type} implementations",
                    metadata={
                        "transformation_type": trans_type,
                        "source_count": source_count,
                        "target_count": target_count,
                        "coverage_percent": round((target_count / source_count) * 100, 2)
                    }
                )
                gaps.append(gap)

        return gaps

    def _detect_data_quality_gaps(
        self,
        source_scripts: List[ScriptLogic],
        target_scripts: List[ScriptLogic],
        source_name: str,
        target_name: str
    ) -> List[Gap]:
        """Detect data quality check gaps"""
        gaps = []

        # Count data quality transformations (FILTER with validations)
        source_dq_count = sum(
            1 for script in source_scripts
            for trans in script.transformations
            if trans.transformation_type == TransformationType.FILTER
            and self._is_data_quality_check(trans)
        )

        target_dq_count = sum(
            1 for script in target_scripts
            for trans in script.transformations
            if trans.transformation_type == TransformationType.FILTER
            and self._is_data_quality_check(trans)
        )

        if source_dq_count > 0 and target_dq_count == 0:
            gap = Gap(
                gap_id="data_quality_missing",
                gap_type="missing_data_quality",
                severity="critical",
                source_system=source_name,
                target_system=target_name,
                title="Missing data quality checks",
                description=f"{source_name} has {source_dq_count} data quality checks, but {target_name} has none",
                recommendation=f"Implement data quality validation in {target_name}",
                metadata={
                    "source_dq_checks": source_dq_count
                }
            )
            gaps.append(gap)

        elif target_dq_count < source_dq_count * 0.7:  # Less than 70% coverage
            gap = Gap(
                gap_id="data_quality_incomplete",
                gap_type="incomplete_data_quality",
                severity="high",
                source_system=source_name,
                target_system=target_name,
                title="Incomplete data quality checks",
                description=f"Data quality: {source_dq_count} in {source_name} vs {target_dq_count} in {target_name}",
                recommendation="Review and implement missing data quality validations",
                metadata={
                    "source_dq_checks": source_dq_count,
                    "target_dq_checks": target_dq_count,
                    "coverage_percent": round((target_dq_count / source_dq_count) * 100, 2)
                }
            )
            gaps.append(gap)

        return gaps

    def _detect_business_logic_gaps(
        self,
        source_scripts: List[ScriptLogic],
        target_scripts: List[ScriptLogic],
        source_name: str,
        target_name: str
    ) -> List[Gap]:
        """Detect business logic implementation gaps"""
        gaps = []

        # Extract business rules from scripts
        source_rules = self._extract_business_rules(source_scripts)
        target_rules = self._extract_business_rules(target_scripts)

        # Find missing rules
        missing_rules = source_rules - target_rules

        if len(missing_rules) > 0:
            gap = Gap(
                gap_id="business_logic_gaps",
                gap_type="missing_business_logic",
                severity="high",
                source_system=source_name,
                target_system=target_name,
                title="Missing business logic",
                description=f"{len(missing_rules)} business rules found in {source_name} but not in {target_name}",
                recommendation="Review and implement missing business rules",
                metadata={
                    "missing_rules_count": len(missing_rules),
                    "sample_rules": list(missing_rules)[:5]  # First 5 examples
                }
            )
            gaps.append(gap)

        return gaps

    def _detect_io_gaps(
        self,
        source_scripts: List[ScriptLogic],
        target_scripts: List[ScriptLogic],
        source_name: str,
        target_name: str
    ) -> List[Gap]:
        """Detect input/output differences"""
        gaps = []

        # Count inputs and outputs
        source_inputs = sum(len(script.inputs) for script in source_scripts)
        source_outputs = sum(len(script.outputs) for script in source_scripts)

        target_inputs = sum(len(script.inputs) for script in target_scripts)
        target_outputs = sum(len(script.outputs) for script in target_scripts)

        # Check for significant differences
        if source_inputs > 0 and target_inputs < source_inputs * 0.5:
            gap = Gap(
                gap_id="inputs_incomplete",
                gap_type="incomplete_inputs",
                severity="medium",
                source_system=source_name,
                target_system=target_name,
                title="Incomplete input coverage",
                description=f"Inputs: {source_inputs} in {source_name} vs {target_inputs} in {target_name}",
                recommendation="Review and implement missing input sources",
                metadata={
                    "source_inputs": source_inputs,
                    "target_inputs": target_inputs
                }
            )
            gaps.append(gap)

        if source_outputs > 0 and target_outputs < source_outputs * 0.5:
            gap = Gap(
                gap_id="outputs_incomplete",
                gap_type="incomplete_outputs",
                severity="medium",
                source_system=source_name,
                target_system=target_name,
                title="Incomplete output coverage",
                description=f"Outputs: {source_outputs} in {source_name} vs {target_outputs} in {target_name}",
                recommendation="Review and implement missing output destinations",
                metadata={
                    "source_outputs": source_outputs,
                    "target_outputs": target_outputs
                }
            )
            gaps.append(gap)

        return gaps

    def _find_similarities(
        self,
        source_workflows: List[WorkflowFlow],
        target_workflows: List[WorkflowFlow],
        source_scripts: List[ScriptLogic],
        target_scripts: List[ScriptLogic]
    ) -> List[Dict[str, Any]]:
        """Find similar implementations across systems"""
        similarities = []

        # Find matching workflows by name similarity
        for source_wf in source_workflows:
            for target_wf in target_workflows:
                similarity_score = self._calculate_name_similarity(
                    source_wf.workflow_name,
                    target_wf.workflow_name
                )

                if similarity_score > 0.7:  # 70% similar
                    similarities.append({
                        "type": "workflow",
                        "source_name": source_wf.workflow_name,
                        "target_name": target_wf.workflow_name,
                        "similarity_score": similarity_score,
                        "source_actions": len(source_wf.actions),
                        "target_actions": len(target_wf.actions)
                    })

        return similarities

    def _calculate_statistics(
        self,
        source_repo,
        target_repo,
        source_workflows: List[WorkflowFlow],
        target_workflows: List[WorkflowFlow],
        source_scripts: List[ScriptLogic],
        target_scripts: List[ScriptLogic],
        gaps: List[Gap]
    ) -> Dict[str, Any]:
        """Calculate comparison statistics"""
        return {
            "source_stats": {
                "workflows": len(source_workflows),
                "scripts": len(source_scripts),
                "transformations": sum(len(s.transformations) for s in source_scripts),
            },
            "target_stats": {
                "workflows": len(target_workflows),
                "scripts": len(target_scripts),
                "transformations": sum(len(s.transformations) for s in target_scripts),
            },
            "gaps_by_severity": {
                "critical": len([g for g in gaps if g.severity == "critical"]),
                "high": len([g for g in gaps if g.severity == "high"]),
                "medium": len([g for g in gaps if g.severity == "medium"]),
                "low": len([g for g in gaps if g.severity == "low"]),
            },
            "gaps_by_type": self._count_gaps_by_type(gaps),
            "total_gaps": len(gaps)
        }

    def _generate_recommendations(self, gaps: List[Gap], statistics: Dict[str, Any]) -> List[str]:
        """Generate high-level recommendations"""
        recommendations = []

        critical_gaps = [g for g in gaps if g.severity == "critical"]
        high_gaps = [g for g in gaps if g.severity == "high"]

        if len(critical_gaps) > 0:
            recommendations.append(
                f"CRITICAL: Address {len(critical_gaps)} critical gaps immediately"
            )

        if len(high_gaps) > 0:
            recommendations.append(
                f"HIGH PRIORITY: Review and fix {len(high_gaps)} high-severity gaps"
            )

        # Specific recommendations by gap type
        gap_types = self._count_gaps_by_type(gaps)

        if gap_types.get("missing_workflow", 0) > 0:
            recommendations.append(
                "Implement missing workflows to achieve feature parity"
            )

        if gap_types.get("missing_data_quality", 0) > 0 or gap_types.get("incomplete_data_quality", 0) > 0:
            recommendations.append(
                "Strengthen data quality checks to match source system"
            )

        if gap_types.get("missing_transformation_type", 0) > 0:
            recommendations.append(
                "Implement missing transformation types for complete coverage"
            )

        return recommendations

    def _assess_migration_complexity(self, gaps: List[Gap], statistics: Dict[str, Any]) -> str:
        """Assess migration complexity based on gaps"""
        critical_count = statistics["gaps_by_severity"].get("critical", 0)
        high_count = statistics["gaps_by_severity"].get("high", 0)
        total_gaps = statistics["total_gaps"]

        if critical_count > 5 or total_gaps > 50:
            return "very_high"
        elif critical_count > 0 or high_count > 10:
            return "high"
        elif high_count > 0 or total_gaps > 10:
            return "medium"
        else:
            return "low"

    # Helper methods

    def _normalize_name(self, name: str) -> str:
        """Normalize workflow/script name for comparison"""
        return name.lower().replace("_", "").replace("-", "").replace(" ", "")

    def _count_transformation_types(self, scripts: List[ScriptLogic]) -> Dict[str, int]:
        """Count transformations by type"""
        counts = defaultdict(int)

        for script in scripts:
            for trans in script.transformations:
                counts[trans.transformation_type.value] += 1

        return dict(counts)

    def _is_data_quality_check(self, transformation: Transformation) -> bool:
        """Check if transformation is a data quality check"""
        if not transformation.condition:
            return False

        condition_lower = transformation.condition.lower()

        # Keywords indicating data quality
        dq_keywords = [
            'is not null', 'is null', 'not null',
            'length', 'len(',
            'valid', 'invalid',
            'check', 'validate',
            'between', 'in (',
            'regexp', 'regex', 'matches'
        ]

        return any(keyword in condition_lower for keyword in dq_keywords)

    def _extract_business_rules(self, scripts: List[ScriptLogic]) -> Set[str]:
        """Extract business rules from scripts"""
        rules = set()

        for script in scripts:
            for trans in script.transformations:
                if trans.business_meaning:
                    rules.add(trans.business_meaning)

                if trans.condition and len(trans.condition) > 10:
                    rules.add(trans.condition)

        return rules

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names"""
        norm1 = self._normalize_name(name1)
        norm2 = self._normalize_name(name2)

        if norm1 == norm2:
            return 1.0

        # Simple token-based similarity
        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def _count_gaps_by_type(self, gaps: List[Gap]) -> Dict[str, int]:
        """Count gaps by type"""
        counts = defaultdict(int)

        for gap in gaps:
            counts[gap.gap_type] += 1

        return dict(counts)

    def generate_report_text(self, report: ComparisonReport) -> str:
        """
        Generate human-readable text report

        Args:
            report: ComparisonReport object

        Returns:
            Formatted text report
        """
        lines = [
            "=" * 80,
            f"CROSS-SYSTEM COMPARISON REPORT",
            "=" * 80,
            f"Source System: {report.source_system}",
            f"Target System: {report.target_system}",
            f"Migration Complexity: {report.migration_complexity.upper()}",
            "",
            "STATISTICS",
            "-" * 80,
            f"Source: {report.statistics['source_stats']['workflows']} workflows, "
            f"{report.statistics['source_stats']['scripts']} scripts, "
            f"{report.statistics['source_stats']['transformations']} transformations",
            f"Target: {report.statistics['target_stats']['workflows']} workflows, "
            f"{report.statistics['target_stats']['scripts']} scripts, "
            f"{report.statistics['target_stats']['transformations']} transformations",
            "",
            f"Total Gaps Detected: {report.statistics['total_gaps']}",
            f"  - Critical: {report.statistics['gaps_by_severity']['critical']}",
            f"  - High: {report.statistics['gaps_by_severity']['high']}",
            f"  - Medium: {report.statistics['gaps_by_severity']['medium']}",
            f"  - Low: {report.statistics['gaps_by_severity']['low']}",
            "",
        ]

        # Top gaps
        if report.gaps:
            lines.extend([
                "TOP GAPS",
                "-" * 80
            ])

            # Show critical and high gaps
            important_gaps = [g for g in report.gaps if g.severity in ["critical", "high"]][:10]

            for gap in important_gaps:
                lines.extend([
                    f"[{gap.severity.upper()}] {gap.title}",
                    f"  {gap.description}",
                    f"  Recommendation: {gap.recommendation}",
                    ""
                ])

        # Recommendations
        if report.recommendations:
            lines.extend([
                "RECOMMENDATIONS",
                "-" * 80
            ])

            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")

            lines.append("")

        # Similarities
        if report.similarities:
            lines.extend([
                "SIMILAR IMPLEMENTATIONS",
                "-" * 80,
                f"Found {len(report.similarities)} similar workflows/scripts",
                ""
            ])

        lines.append("=" * 80)

        return "\n".join(lines)
