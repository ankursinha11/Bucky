"""
Repository-Level Model (Tier 1)
Highest level view of entire codebase repository
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class RepositoryType(Enum):
    """Type of repository"""
    HADOOP = "hadoop"
    DATABRICKS = "databricks"
    ABINITIO = "abinitio"
    CUSTOM = "custom"


@dataclass
class Repository:
    """
    Repository-level view of entire codebase

    Tier 1: Highest level - repository overview
    Contains summary of all workflows, scripts, and dependencies
    """
    id: str
    name: str
    repo_type: RepositoryType
    base_path: str

    # High-level statistics
    total_workflows: int = 0
    total_scripts: int = 0
    total_notebooks: int = 0
    total_components: int = 0

    # Script breakdown by type
    pig_scripts: int = 0
    spark_scripts: int = 0
    hive_scripts: int = 0
    python_scripts: int = 0
    shell_scripts: int = 0
    sql_scripts: int = 0

    # Workflow/Process references
    workflow_ids: List[str] = field(default_factory=list)
    process_ids: List[str] = field(default_factory=list)

    # Data sources and targets
    data_sources: List[str] = field(default_factory=list)  # Input paths/tables
    data_targets: List[str] = field(default_factory=list)  # Output paths/tables

    # Dependencies
    depends_on_repositories: List[str] = field(default_factory=list)
    used_by_repositories: List[str] = field(default_factory=list)

    # Business context
    business_domains: List[str] = field(default_factory=list)  # e.g., ["CDD", "GMRN", "Lead Discovery"]
    functional_areas: List[str] = field(default_factory=list)  # e.g., ["Patient Matching", "Coverage Discovery"]

    # Technology stack
    technologies: List[str] = field(default_factory=list)  # e.g., ["Pig", "Spark", "Hive"]

    # Metadata
    description: Optional[str] = None
    owner: Optional[str] = None
    version: Optional[str] = None
    last_modified: Optional[datetime] = None

    # Summary text for search
    summary: Optional[str] = None

    # AI-generated insights
    ai_summary: Optional[str] = None  # High-level purpose from AI
    ai_architecture: Optional[str] = None  # Architecture description from AI

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for indexing"""
        return {
            "id": self.id,
            "name": self.name,
            "repo_type": self.repo_type.value,
            "base_path": self.base_path,
            "total_workflows": self.total_workflows,
            "total_scripts": self.total_scripts,
            "total_notebooks": self.total_notebooks,
            "pig_scripts": self.pig_scripts,
            "spark_scripts": self.spark_scripts,
            "hive_scripts": self.hive_scripts,
            "workflow_ids": self.workflow_ids,
            "data_sources": self.data_sources,
            "data_targets": self.data_targets,
            "depends_on_repositories": self.depends_on_repositories,
            "business_domains": self.business_domains,
            "functional_areas": self.functional_areas,
            "technologies": self.technologies,
            "description": self.description,
            "summary": self.summary,
            "ai_summary": self.ai_summary,
            "ai_architecture": self.ai_architecture,
        }

    def get_searchable_content(self) -> str:
        """Generate rich searchable content"""
        parts = [
            f"Repository: {self.name}",
            f"Type: {self.repo_type.value}",
            f"Path: {self.base_path}",
            f"",
            f"Statistics:",
            f"- {self.total_workflows} workflows/pipelines",
            f"- {self.total_scripts} scripts total",
            f"  - {self.pig_scripts} Pig scripts",
            f"  - {self.spark_scripts} Spark scripts",
            f"  - {self.hive_scripts} Hive scripts",
            f"  - {self.python_scripts} Python scripts",
            f"",
        ]

        if self.description:
            parts.append(f"Description: {self.description}")
            parts.append("")

        if self.business_domains:
            parts.append(f"Business Domains: {', '.join(self.business_domains)}")

        if self.functional_areas:
            parts.append(f"Functional Areas: {', '.join(self.functional_areas)}")

        if self.technologies:
            parts.append(f"Technologies: {', '.join(self.technologies)}")
            parts.append("")

        if self.data_sources:
            parts.append(f"Data Sources ({len(self.data_sources)}):")
            for src in self.data_sources[:10]:  # First 10
                parts.append(f"  - {src}")
            if len(self.data_sources) > 10:
                parts.append(f"  ... and {len(self.data_sources) - 10} more")
            parts.append("")

        if self.data_targets:
            parts.append(f"Data Targets ({len(self.data_targets)}):")
            for tgt in self.data_targets[:10]:
                parts.append(f"  - {tgt}")
            if len(self.data_targets) > 10:
                parts.append(f"  ... and {len(self.data_targets) - 10} more")
            parts.append("")

        if self.depends_on_repositories:
            parts.append(f"Depends On: {', '.join(self.depends_on_repositories)}")

        if self.summary:
            parts.append("")
            parts.append("Summary:")
            parts.append(self.summary)

        if self.ai_summary:
            parts.append("")
            parts.append("AI Analysis:")
            parts.append(self.ai_summary)

        if self.ai_architecture:
            parts.append("")
            parts.append("Architecture:")
            parts.append(self.ai_architecture)

        return "\n".join(parts)

    def __repr__(self):
        return f"<Repository {self.name} ({self.repo_type.value}): {self.total_workflows} workflows, {self.total_scripts} scripts>"
