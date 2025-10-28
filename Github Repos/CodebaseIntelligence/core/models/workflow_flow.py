"""
Workflow Flow Model (Tier 2)
Mid-level view of workflow execution flow and action dependencies
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class ActionType(Enum):
    """Type of action in workflow"""
    PIG_SCRIPT = "pig"
    SPARK_JOB = "spark"
    HIVE_QUERY = "hive"
    PYTHON_SCRIPT = "python"
    SHELL_SCRIPT = "shell"
    SQL_QUERY = "sql"
    SQOOP_IMPORT = "sqoop"
    NOTEBOOK = "notebook"
    SUB_WORKFLOW = "sub_workflow"
    UNKNOWN = "unknown"


@dataclass
class ActionNode:
    """
    Single action/step in a workflow

    Represents one executable unit (script, query, job) in the workflow
    Links to Tier 3 (ScriptLogic) for detailed logic
    """
    action_id: str
    action_name: str
    action_type: ActionType

    # Script/File reference
    script_path: Optional[str] = None
    script_content_id: Optional[str] = None  # Links to Tier 3 ScriptLogic

    # Data flow
    input_tables: List[str] = field(default_factory=list)
    output_tables: List[str] = field(default_factory=list)
    input_paths: List[str] = field(default_factory=list)
    output_paths: List[str] = field(default_factory=list)

    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    environment_vars: Dict[str, str] = field(default_factory=dict)

    # Execution
    execution_order: Optional[int] = None
    depends_on: List[str] = field(default_factory=list)  # Other action IDs
    triggers: List[str] = field(default_factory=list)  # Action IDs this triggers

    # Business context
    description: Optional[str] = None
    business_purpose: Optional[str] = None  # AI-generated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_name": self.action_name,
            "action_type": self.action_type.value,
            "script_path": self.script_path,
            "script_content_id": self.script_content_id,
            "input_tables": self.input_tables,
            "output_tables": self.output_tables,
            "input_paths": self.input_paths,
            "output_paths": self.output_paths,
            "parameters": self.parameters,
            "execution_order": self.execution_order,
            "depends_on": self.depends_on,
            "triggers": self.triggers,
            "description": self.description,
            "business_purpose": self.business_purpose,
        }


@dataclass
class FlowEdge:
    """
    Connection between two actions in workflow

    Represents data or control flow between actions
    """
    edge_id: str
    from_action: str  # Action ID
    to_action: str    # Action ID

    # Data passing through this edge
    data_passed: List[str] = field(default_factory=list)  # Table/file names

    # Edge properties
    edge_type: str = "data"  # data, control, dependency
    condition: Optional[str] = None  # Conditional edge (e.g., "if success")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "from_action": self.from_action,
            "to_action": self.to_action,
            "data_passed": self.data_passed,
            "edge_type": self.edge_type,
            "condition": self.condition,
        }


@dataclass
class WorkflowFlow:
    """
    Workflow-level flow diagram and execution plan

    Tier 2: Mid-level - workflow flow and action orchestration
    Links actions together and shows data/control flow
    """
    workflow_id: str
    workflow_name: str
    workflow_type: str  # oozie_workflow, oozie_coordinator, notebook, pipeline

    # Repository context
    repository_id: Optional[str] = None
    file_path: Optional[str] = None

    # Actions and flow
    actions: List[ActionNode] = field(default_factory=list)
    edges: List[FlowEdge] = field(default_factory=list)

    # Entry and exit points
    entry_actions: List[str] = field(default_factory=list)  # Action IDs that start the workflow
    exit_actions: List[str] = field(default_factory=list)   # Action IDs that end the workflow

    # Data flow summary
    overall_inputs: List[str] = field(default_factory=list)   # All input tables/paths
    overall_outputs: List[str] = field(default_factory=list)  # All output tables/paths
    intermediate_data: List[str] = field(default_factory=list)  # Temp tables/paths

    # Flow diagrams
    flow_diagram_mermaid: Optional[str] = None  # Mermaid.js format
    flow_diagram_ascii: Optional[str] = None    # ASCII art

    # Execution
    schedule: Optional[str] = None
    estimated_runtime: Optional[str] = None
    parallelizable_actions: List[List[str]] = field(default_factory=list)  # Groups that can run in parallel

    # Business context
    business_purpose: Optional[str] = None
    business_domain: Optional[str] = None
    functional_area: Optional[str] = None

    # AI analysis
    ai_flow_summary: Optional[str] = None  # High-level flow description
    ai_optimization_suggestions: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "workflow_type": self.workflow_type,
            "repository_id": self.repository_id,
            "file_path": self.file_path,
            "actions": [a.to_dict() for a in self.actions],
            "edges": [e.to_dict() for e in self.edges],
            "entry_actions": self.entry_actions,
            "exit_actions": self.exit_actions,
            "overall_inputs": self.overall_inputs,
            "overall_outputs": self.overall_outputs,
            "flow_diagram_mermaid": self.flow_diagram_mermaid,
            "business_purpose": self.business_purpose,
            "ai_flow_summary": self.ai_flow_summary,
        }

    def get_action_by_id(self, action_id: str) -> Optional[ActionNode]:
        """Get action by ID"""
        for action in self.actions:
            if action.action_id == action_id:
                return action
        return None

    def get_downstream_actions(self, action_id: str) -> List[str]:
        """Get all actions triggered by this action"""
        return [e.to_action for e in self.edges if e.from_action == action_id]

    def get_upstream_actions(self, action_id: str) -> List[str]:
        """Get all actions that trigger this action"""
        return [e.from_action for e in self.edges if e.to_action == action_id]

    def get_critical_path(self) -> List[str]:
        """Get critical path (longest execution path)"""
        # Simple topological sort for now
        # TODO: Implement proper critical path analysis
        return [a.action_id for a in sorted(self.actions, key=lambda a: a.execution_order or 0)]

    def get_searchable_content(self) -> str:
        """Generate rich searchable content"""
        parts = [
            f"Workflow: {self.workflow_name}",
            f"Type: {self.workflow_type}",
            f"",
            f"Flow: {len(self.actions)} actions, {len(self.edges)} connections",
        ]

        if self.business_purpose:
            parts.append(f"Purpose: {self.business_purpose}")

        if self.business_domain:
            parts.append(f"Domain: {self.business_domain}")

        if self.actions:
            parts.append("")
            parts.append(f"Actions ({len(self.actions)}):")
            for action in self.actions[:20]:  # First 20
                parts.append(f"  {action.execution_order or '?'}. {action.action_name} ({action.action_type.value})")
                if action.script_path:
                    parts.append(f"     Script: {action.script_path}")
                if action.business_purpose:
                    parts.append(f"     Purpose: {action.business_purpose}")

        if self.overall_inputs:
            parts.append("")
            parts.append(f"Overall Inputs ({len(self.overall_inputs)}):")
            for inp in self.overall_inputs[:10]:
                parts.append(f"  - {inp}")

        if self.overall_outputs:
            parts.append("")
            parts.append(f"Overall Outputs ({len(self.overall_outputs)}):")
            for out in self.overall_outputs[:10]:
                parts.append(f"  - {out}")

        if self.flow_diagram_ascii:
            parts.append("")
            parts.append("Flow Diagram:")
            parts.append(self.flow_diagram_ascii)

        if self.flow_diagram_mermaid:
            parts.append("")
            parts.append("Flow Diagram (Mermaid):")
            parts.append(self.flow_diagram_mermaid)

        if self.ai_flow_summary:
            parts.append("")
            parts.append("AI Flow Analysis:")
            parts.append(self.ai_flow_summary)

        if self.ai_optimization_suggestions:
            parts.append("")
            parts.append("Optimization Suggestions:")
            parts.append(self.ai_optimization_suggestions)

        return "\n".join(parts)

    def __repr__(self):
        return f"<WorkflowFlow {self.workflow_name}: {len(self.actions)} actions, {len(self.edges)} edges>"
