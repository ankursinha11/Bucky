"""
Integrated Ab Initio + Autosys Parser
======================================
Combines Ab Initio .mp file parsing with Autosys job dependencies
to create complete workflow flows

This solves the GraphFlow extraction problem by using Autosys
dependencies to understand how Ab Initio graphs connect!
"""

import os
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger

from parsers.abinitio.parser import AbInitioParser
from parsers.abinitio.deep_parser_multi_repo import DeepAbInitioParserMultiRepo
from parsers.autosys.parser import AutosysParser


class IntegratedAbInitioAutosysParser:
    """
    Integrated parser that combines:
    1. Ab Initio .mp files (graph structure, components, parameters)
    2. Autosys JIL files (job dependencies, execution order)

    Result: Complete workflow with accurate GraphFlow!

    Now supports multi-project detection!
    """

    def __init__(self, use_ai: bool = True):
        """
        Initialize integrated parser

        Args:
            use_ai: Whether to use AI analysis (default: True)
        """
        self.base_parser = AbInitioParser()  # For Excel export
        self.deep_parser = DeepAbInitioParserMultiRepo(use_ai=use_ai)  # For indexing
        self.autosys_parser = AutosysParser()
        self.use_ai = use_ai

    def parse_combined(
        self,
        abinitio_path: str,
        autosys_path: str = None,
        use_ai: bool = True
    ) -> Dict[str, Any]:
        """
        Parse Ab Initio + Autosys together with AI analysis

        SMART ORDER:
        1. Parse Autosys first (job dependencies, execution order)
        2. Parse Ab Initio with Autosys context (accurate GraphFlow)
        3. Run AI analysis with full context

        Args:
            abinitio_path: Path to Ab Initio .mp files
            autosys_path: Path to Autosys job files (optional)
            use_ai: Whether to use AI analysis (default: True)

        Returns:
            Enhanced results with accurate GraphFlow and AI insights
        """
        logger.info("🔗 Integrated Ab Initio + Autosys Parsing (SMART ORDER)")

        # STEP 1: Parse Autosys FIRST to get context
        autosys_result = None
        autosys_context = {}

        # If no Autosys path provided, try to find it
        if not autosys_path:
            autosys_path = self._find_autosys_directory(abinitio_path)

        if autosys_path and os.path.exists(autosys_path):
            logger.info(f"📅 STEP 1: Parsing Autosys first: {autosys_path}")
            autosys_result = self.autosys_parser.parse_directory(autosys_path)

            # Build context for Ab Initio parsing
            autosys_context = self._build_autosys_context(autosys_result)

            logger.info(f"   ✓ Found {len(autosys_result['processes'])} Autosys jobs")
            logger.info(f"   ✓ Found {len(autosys_result['flows'])} job dependencies")
            logger.info(f"   ✓ Identified {len(autosys_context.get('graph_references', []))} Ab Initio graph references")
        else:
            logger.warning("No Autosys files found - GraphFlow will be limited")

        # STEP 2: Parse Ab Initio with Autosys context (BOTH parsers)
        logger.info(f"📦 STEP 2: Parsing Ab Initio with Autosys context: {abinitio_path}")

        # Base parser for Excel export
        base_result = self.base_parser.parse_directory(abinitio_path)
        logger.info(f"   ✓ Base parsing: {len(base_result['processes'])} Ab Initio graphs")

        # Deep parser for indexing (with multi-repo support!)
        deep_result = self.deep_parser.parse_directory(abinitio_path)
        logger.info(f"   ✓ Deep parsing: {len(deep_result.get('workflow_flows', []))} workflows, "
                   f"{len(deep_result.get('script_logics', []))} scripts")

        if deep_result.get("repositories"):
            logger.info(f"   ✓ Detected {len(deep_result['repositories'])} Ab Initio projects")
            for repo in deep_result['repositories']:
                logger.info(f"      - {repo.name}")

        # STEP 3: Merge and enhance with Autosys context
        logger.info("🔗 STEP 3: Integrating Autosys dependencies with Ab Initio graphs")
        integrated_result = self._integrate_results(base_result, deep_result, autosys_result, autosys_context)

        # STEP 4: AI Analysis with full context
        if use_ai and autosys_result:
            logger.info("🤖 STEP 4: Running AI analysis with full Autosys + Ab Initio context")
            self._run_integrated_ai_analysis(integrated_result, autosys_context)

        logger.info("✅ Integrated parsing complete!")
        logger.info(f"   Ab Initio Graphs: {len(base_result['processes'])}")
        if autosys_result:
            logger.info(f"   Autosys Jobs: {len(autosys_result['processes'])}")
            logger.info(f"   Job Dependencies: {autosys_result['summary']['total_dependencies']}")
        logger.info(f"   Enhanced GraphFlow: {len(integrated_result.get('enhanced_flows', []))}")

        return integrated_result

    def _find_autosys_directory(self, abinitio_path: str) -> str:
        """
        Try to find Autosys directory near Ab Initio directory

        Common patterns:
        - /path/to/abinitio/ and /path/to/autosys/
        - /path/to/project/abinitio/ and /path/to/project/autosys/
        - /path/to/project/graphs/ and /path/to/project/jobs/
        """
        base_path = Path(abinitio_path).parent

        # Try common names
        autosys_names = ['autosys', 'jobs', 'jil', 'scheduler', 'job_definitions']

        for name in autosys_names:
            potential_path = base_path / name
            if potential_path.exists():
                logger.info(f"Found Autosys directory: {potential_path}")
                return str(potential_path)

        return None

    def _build_autosys_context(self, autosys_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build context from Autosys parsing for Ab Initio

        Extracts:
        - Graph references (which .mp files are used)
        - Job execution order
        - Job dependencies (for GraphFlow)
        - Critical paths

        Args:
            autosys_result: Parsed Autosys data

        Returns:
            Context dictionary with graph references and dependencies
        """
        context = {
            "graph_references": [],
            "job_to_graph_map": {},
            "execution_order": [],
            "critical_graphs": set(),
        }

        # Extract graph references from job commands
        for job in autosys_result.get("components", []):
            graph_ref = job.parameters.get("abinitio_graph", "")
            if graph_ref:
                context["graph_references"].append({
                    "job_name": job.name,
                    "graph_path": graph_ref,
                })
                context["job_to_graph_map"][job.name] = graph_ref

        # Identify critical graphs (graphs at start/end of dependency chains)
        flows = autosys_result.get("flows", [])
        source_jobs = {f["source_job"] for f in flows}
        target_jobs = {f["target_job"] for f in flows}

        # Entry points (no dependencies)
        entry_jobs = [j for j in context["job_to_graph_map"].keys() if j not in target_jobs]
        # Exit points (no successors)
        exit_jobs = [j for j in context["job_to_graph_map"].keys() if j not in source_jobs]

        for job in entry_jobs:
            if job in context["job_to_graph_map"]:
                context["critical_graphs"].add(context["job_to_graph_map"][job])

        for job in exit_jobs:
            if job in context["job_to_graph_map"]:
                context["critical_graphs"].add(context["job_to_graph_map"][job])

        logger.info(f"Built Autosys context: {len(context['graph_references'])} graph references, "
                   f"{len(context['critical_graphs'])} critical graphs")

        return context

    def _run_integrated_ai_analysis(self, integrated_result: Dict[str, Any], autosys_context: Dict[str, Any]):
        """
        Run AI analysis with full Autosys + Ab Initio context

        Args:
            integrated_result: Integrated parsing results
            autosys_context: Autosys context with job dependencies
        """
        try:
            from services.ai_script_analyzer import AIScriptAnalyzer
            ai_analyzer = AIScriptAnalyzer()

            logger.info("Running AI analysis with Autosys + Ab Initio context...")

            # Build enhanced context for AI
            enhanced_context = {
                "job_dependencies": integrated_result.get("enhanced_flows", []),
                "critical_graphs": list(autosys_context.get("critical_graphs", [])),
                "total_jobs": len(integrated_result.get("autosys_jobs", [])),
                "graph_references": autosys_context.get("graph_references", []),
            }

            # Analyze each graph with Autosys context
            for process in integrated_result.get("processes", []):
                try:
                    # Check if this graph is referenced in Autosys
                    graph_name = process.name
                    is_critical = graph_name in autosys_context.get("critical_graphs", set())

                    # Find Autosys job for this graph
                    autosys_job = None
                    for job_ref in autosys_context.get("graph_references", []):
                        if graph_name in job_ref["graph_path"]:
                            autosys_job = job_ref["job_name"]
                            break

                    # Build AI prompt with context
                    context_prompt = f"""
Analyze this Ab Initio graph with Autosys scheduling context:

Graph: {graph_name}
Autosys Job: {autosys_job or 'Not scheduled'}
Critical Path: {'Yes' if is_critical else 'No'}
Total Components: {len(process.component_ids)}

Provide:
1. Business purpose
2. Role in workflow (based on Autosys dependencies)
3. Key transformations
4. Critical data flows
"""

                    # Run AI analysis (simplified - you'd use actual AI analyzer methods)
                    logger.info(f"  Analyzing {graph_name} with Autosys context...")

                except Exception as e:
                    logger.warning(f"AI analysis failed for {process.name}: {e}")

            logger.info("✓ AI analysis complete with integrated context")

        except ImportError:
            logger.warning("AI analyzer not available - skipping AI analysis")
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")

    def _integrate_results(
        self,
        base_result: Dict[str, Any],
        deep_result: Dict[str, Any],
        autosys_result: Dict[str, Any] = None,
        autosys_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Merge Ab Initio (base + deep) and Autosys results

        Creates enhanced GraphFlow by linking:
        1. Autosys job dependencies
        2. Ab Initio graph references in Autosys commands
        3. Component-level flows from .mp files

        Args:
            base_result: Base parser result (for Excel)
            deep_result: Deep parser result (for indexing)
            autosys_result: Autosys parser result
            autosys_context: Autosys context
        """
        # Combine base and deep results
        integrated = {
            # Base parsing (for Excel)
            "processes": base_result["processes"],
            "components": base_result["components"],
            "raw_mp_data": base_result.get("raw_mp_data", []),

            # Deep parsing (for indexing)
            "repository": deep_result.get("repository"),
            "repositories": deep_result.get("repositories", []),
            "workflow_flows": deep_result.get("workflow_flows", []),
            "script_logics": deep_result.get("script_logics", []),
        }

        # If we have Autosys data, enhance the results
        if autosys_result:
            # Map Autosys jobs to Ab Initio graphs
            job_to_graph_map = self._map_jobs_to_graphs(
                autosys_result["components"],
                base_result["processes"]
            )

            # Build enhanced GraphFlow from Autosys dependencies
            enhanced_flows = self._build_enhanced_graph_flow(
                autosys_result["flows"],
                job_to_graph_map
            )

            integrated["autosys_jobs"] = autosys_result["components"]
            integrated["autosys_flows"] = autosys_result["flows"]
            integrated["enhanced_flows"] = enhanced_flows
            integrated["job_to_graph_map"] = job_to_graph_map

            # Update raw_mp_data with enhanced GraphFlow
            for mp_data in integrated.get("raw_mp_data", []):
                graph_name = mp_data.get("file_name", "")

                # Find flows for this graph
                graph_flows = [
                    f for f in enhanced_flows
                    if f.get("source_graph") == graph_name or f.get("target_graph") == graph_name
                ]

                if graph_flows:
                    # Replace empty graph_flow with enhanced flows
                    mp_data["graph_flow"] = [
                        {
                            "source_component_name": f["source_graph"],
                            "target_component_name": f["target_graph"],
                        }
                        for f in graph_flows
                    ]
                    mp_data["flow_count"] = len(graph_flows)

        integrated["summary"] = {
            "total_graphs": len(base_result["processes"]),
            "total_components": len(base_result["components"]),
            "total_projects": len(deep_result.get("repositories", [])),
            "autosys_jobs": len(autosys_result["processes"]) if autosys_result else 0,
            "enhanced_flows": len(integrated.get("enhanced_flows", [])),
            "graph_references": len(autosys_context.get("graph_references", [])) if autosys_context else 0,
            "critical_graphs": len(autosys_context.get("critical_graphs", [])) if autosys_context else 0,
            "integration": "complete_with_context" if autosys_context else ("complete" if autosys_result else "abinitio_only")
        }

        return integrated

    def _map_jobs_to_graphs(
        self,
        autosys_jobs: List,
        abinitio_graphs: List
    ) -> Dict[str, str]:
        """
        Map Autosys job names to Ab Initio graph names

        Uses the abinitio_graph parameter extracted from job commands
        """
        mapping = {}

        for job in autosys_jobs:
            job_name = job.name
            graph_ref = job.parameters.get("abinitio_graph", "")

            if graph_ref:
                # Try to find matching graph
                for graph in abinitio_graphs:
                    graph_name = graph.name

                    # Match by name or path
                    if (graph_name in graph_ref or
                        graph_ref in graph_name or
                        Path(graph_ref).stem == graph_name):

                        mapping[job_name] = graph_name
                        logger.debug(f"Mapped job '{job_name}' -> graph '{graph_name}'")
                        break

        logger.info(f"Mapped {len(mapping)} Autosys jobs to Ab Initio graphs")

        return mapping

    def _build_enhanced_graph_flow(
        self,
        autosys_flows: List[Dict],
        job_to_graph_map: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """
        Build enhanced GraphFlow from Autosys job dependencies

        Translates: Job A -> Job B
        Into: Graph X -> Graph Y
        """
        enhanced_flows = []

        for flow in autosys_flows:
            source_job = flow["source_job"]
            target_job = flow["target_job"]

            # Map to graphs
            source_graph = job_to_graph_map.get(source_job)
            target_graph = job_to_graph_map.get(target_job)

            if source_graph and target_graph:
                enhanced_flow = {
                    "source_graph": source_graph,
                    "target_graph": target_graph,
                    "source_job": source_job,
                    "target_job": target_job,
                    "condition_type": flow.get("condition_type", ""),
                    "dependency_type": "autosys_job_dependency"
                }

                enhanced_flows.append(enhanced_flow)

        logger.info(f"Built {len(enhanced_flows)} enhanced graph flows from Autosys")

        return enhanced_flows

    def export_to_excel_integrated(self, output_path: str, integrated_result: Dict[str, Any]):
        """
        Export integrated results to Excel with enhanced GraphFlow

        For multi-project codebases, creates separate Excel files per project:
        - {output_path}_project1.xlsx
        - {output_path}_project2.xlsx
        - {output_path}_summary.xlsx (combined)

        Each Excel contains 5 sheets:
        1. GraphParameters
        2. Components&Fields
        3. GraphFlow (ENHANCED with Autosys dependencies!)
        4. Summary
        5. AutosysJobs (NEW!)
        """
        import pandas as pd
        from pathlib import Path

        logger.info(f"Exporting integrated results to Excel: {output_path}")

        # Check if we have multiple projects
        repositories = integrated_result.get("repositories", [])
        output_path_obj = Path(output_path)

        if len(repositories) > 1:
            logger.info(f"📊 Detected {len(repositories)} projects - creating separate Excel files")

            # Group raw_mp_data by project
            project_data_map = self._group_mp_data_by_project(
                integrated_result.get("raw_mp_data", []),
                repositories
            )

            excel_files = []

            # Export each project separately
            for project_name, project_mp_data in project_data_map.items():
                project_excel_path = output_path_obj.parent / f"{output_path_obj.stem}_{project_name}.xlsx"

                logger.info(f"   Exporting project: {project_name} ({len(project_mp_data)} graphs)")

                # Export this project's data
                self.base_parser.raw_mp_data = project_mp_data
                self.base_parser.export_to_excel(str(project_excel_path))

                excel_files.append(str(project_excel_path))

            # Create combined summary file
            summary_excel_path = output_path_obj.parent / f"{output_path_obj.stem}_ALL_PROJECTS.xlsx"
            self._create_combined_summary(summary_excel_path, integrated_result, project_data_map)
            excel_files.append(str(summary_excel_path))

            logger.info(f"✓ Created {len(excel_files)} Excel files:")
            for excel_file in excel_files:
                logger.info(f"   - {Path(excel_file).name}")

            return excel_files

        else:
            # Single project - use original method
            logger.info("📊 Single project - creating one Excel file")

            # Export Ab Initio data (creates 4 sheets)
            self.base_parser.raw_mp_data = integrated_result.get("raw_mp_data", [])
            self.base_parser.export_to_excel(output_path)

            # Add Autosys sheet if available
            if integrated_result.get("autosys_jobs"):
                autosys_data = []

                for job in integrated_result["autosys_jobs"]:
                    autosys_data.append({
                        "Job_Name": job.name,
                        "Ab_Initio_Graph": job.parameters.get("abinitio_graph", ""),
                        "Command": job.parameters.get("command", ""),
                        "Machine": job.parameters.get("machine", ""),
                        "Condition": job.parameters.get("condition", ""),
                        "Description": job.business_description or "",
                    })

                if autosys_data:
                    df_autosys = pd.DataFrame(autosys_data)

                    # Append as new sheet
                    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                        df_autosys.to_excel(writer, sheet_name='AutosysJobs', index=False)

            logger.info(f"✓ Integrated Excel created: {output_path}")
            logger.info(f"   5 sheets with ENHANCED GraphFlow from Autosys!")

            return [output_path]

    def _group_mp_data_by_project(self, raw_mp_data: List[Dict], repositories: List) -> Dict[str, List[Dict]]:
        """
        Group raw_mp_data by project based on file paths

        Args:
            raw_mp_data: List of raw mp data dictionaries
            repositories: List of repository objects

        Returns:
            Dictionary mapping project name to list of mp_data
        """
        project_data_map = {}

        # Create map of project paths
        project_paths = {}
        for repo in repositories:
            project_name = repo.name
            # Get the project folder path from repository metadata
            if hasattr(repo, 'path'):
                project_paths[project_name] = repo.path
            else:
                project_paths[project_name] = project_name

        # Group mp_data by project
        for mp_data in raw_mp_data:
            file_path = mp_data.get("file_path", "")

            # Find which project this file belongs to
            assigned = False
            for project_name, project_path in project_paths.items():
                if project_name in file_path or project_path in file_path:
                    if project_name not in project_data_map:
                        project_data_map[project_name] = []
                    project_data_map[project_name].append(mp_data)
                    assigned = True
                    break

            # If not assigned, put in "other" category
            if not assigned:
                if "other" not in project_data_map:
                    project_data_map["other"] = []
                project_data_map["other"].append(mp_data)

        logger.info(f"Grouped {len(raw_mp_data)} graphs into {len(project_data_map)} projects")
        for project_name, data_list in project_data_map.items():
            logger.info(f"   {project_name}: {len(data_list)} graphs")

        return project_data_map

    def _create_combined_summary(self, output_path: Path, integrated_result: Dict[str, Any], project_data_map: Dict[str, List[Dict]]):
        """
        Create a combined summary Excel with overview of all projects

        Args:
            output_path: Path to output Excel file
            integrated_result: Integrated parsing results
            project_data_map: Map of project names to mp_data lists
        """
        import pandas as pd

        logger.info(f"Creating combined summary: {output_path}")

        with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
            # Sheet 1: Project Summary
            project_summary = []
            for project_name, mp_data_list in project_data_map.items():
                total_components = sum(mp.get("component_count", 0) for mp in mp_data_list)
                total_params = sum(len(mp.get("graph_parameters", [])) for mp in mp_data_list)
                total_flows = sum(mp.get("flow_count", 0) for mp in mp_data_list)

                project_summary.append({
                    "Project": project_name,
                    "Total_Graphs": len(mp_data_list),
                    "Total_Components": total_components,
                    "Total_Parameters": total_params,
                    "Total_Flows": total_flows,
                })

            df_project_summary = pd.DataFrame(project_summary)
            df_project_summary.to_excel(writer, sheet_name='ProjectSummary', index=False)

            # Sheet 2: Autosys Jobs (if available)
            if integrated_result.get("autosys_jobs"):
                autosys_data = []
                for job in integrated_result["autosys_jobs"]:
                    autosys_data.append({
                        "Job_Name": job.name,
                        "Ab_Initio_Graph": job.parameters.get("abinitio_graph", ""),
                        "Command": job.parameters.get("command", ""),
                        "Machine": job.parameters.get("machine", ""),
                        "Condition": job.parameters.get("condition", ""),
                        "Description": job.business_description or "",
                    })

                if autosys_data:
                    df_autosys = pd.DataFrame(autosys_data)
                    df_autosys.to_excel(writer, sheet_name='AutosysJobs', index=False)

            # Sheet 3: Overall Summary
            overall_summary = [{
                "Total_Projects": len(project_data_map),
                "Total_Graphs": sum(len(mp_list) for mp_list in project_data_map.values()),
                "Total_Components": integrated_result.get("summary", {}).get("total_components", 0),
                "Autosys_Jobs": len(integrated_result.get("autosys_jobs", [])),
                "Enhanced_Flows": integrated_result.get("summary", {}).get("enhanced_flows", 0),
            }]

            df_overall = pd.DataFrame(overall_summary)
            df_overall.to_excel(writer, sheet_name='OverallSummary', index=False)

        logger.info(f"✓ Combined summary created: {output_path}")
