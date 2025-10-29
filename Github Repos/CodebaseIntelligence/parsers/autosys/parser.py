"""
Autosys JIL Parser
==================
Parses Autosys Job Information Language (JIL) files

Features:
- Extract job definitions
- Parse job dependencies (conditions)
- Identify Ab Initio graph references in commands
- Build workflow execution flow
- Link jobs to Ab Initio graphs
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from core.models import Process, Component, ProcessType, ComponentType, SystemType


class AutosysParser:
    """
    Parse Autosys JIL (Job Information Language) files

    Extracts:
    - Job definitions
    - Dependencies (conditions)
    - Commands (links to Ab Initio graphs)
    - Schedules and calendars
    """

    def __init__(self):
        """Initialize Autosys parser"""
        pass

    def parse_directory(self, autosys_path: str) -> Dict[str, Any]:
        """
        Parse directory containing Autosys JIL files

        Args:
            autosys_path: Path to directory with .jil files

        Returns:
            Dict with 'processes' (workflows), 'components' (jobs), and 'flows' (dependencies)
        """
        logger.info(f"Parsing Autosys directory: {autosys_path}")

        jil_files = self._find_jil_files(autosys_path)
        logger.info(f"Found {len(jil_files)} JIL files")

        all_jobs = []
        all_flows = []

        for jil_file in jil_files:
            jobs, flows = self.parse_jil_file(jil_file)
            all_jobs.extend(jobs)
            all_flows.extend(flows)

        # Convert to Process/Component model
        processes, components = self._convert_to_models(all_jobs, all_flows, autosys_path)

        logger.info(f"✓ Parsed {len(processes)} workflows, {len(components)} jobs")

        return {
            "processes": processes,
            "components": components,
            "flows": all_flows,
            "summary": {
                "total_workflows": len(processes),
                "total_jobs": len(all_jobs),
                "total_dependencies": len(all_flows),
                "source_path": autosys_path,
            }
        }

    def parse_jil_file(self, file_path: str) -> tuple:
        """
        Parse single JIL file

        Args:
            file_path: Path to .jil file

        Returns:
            Tuple of (jobs_list, flows_list)
        """
        logger.info(f"Parsing JIL file: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return ([], [])

        # Split content into individual job blocks
        job_blocks = self._split_job_blocks(content)

        jobs = []
        flows = []

        for block in job_blocks:
            job_data = self._parse_job_block(block, file_path)
            if job_data:
                jobs.append(job_data)

                # Extract dependencies from this job
                job_flows = self._extract_dependencies(job_data)
                flows.extend(job_flows)

        logger.info(f"✓ Parsed {len(jobs)} jobs, {len(flows)} dependencies from {Path(file_path).name}")

        return (jobs, flows)

    def _find_jil_files(self, base_path: str) -> List[str]:
        """
        Find all JIL files recursively

        Handles both:
        - Files with .jil extension
        - Files without extension (validated by content)
        """
        jil_files = []

        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)

                # Accept .jil files directly
                if file.endswith('.jil') or file.endswith('.JIL'):
                    jil_files.append(file_path)
                # For files without extension, validate content
                elif '.' not in file or file.split('.')[-1].isupper():
                    # Check if it's a JIL file by content
                    if self._is_jil_file(file_path):
                        jil_files.append(file_path)

        return sorted(jil_files)

    def _is_jil_file(self, file_path: str) -> bool:
        """
        Check if file is JIL format by looking for JIL keywords

        Returns True if file contains Autosys job definition keywords
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first 20 lines to check for JIL content
                first_lines = []
                for i in range(20):
                    line = f.readline()
                    if not line:
                        break
                    first_lines.append(line)

                content = ''.join(first_lines).lower()

                # JIL files must have these keywords
                jil_keywords = [
                    'insert_job:', 'update_job:', 'delete_job:',
                    'job_type:', 'command:', 'machine:', 'condition:',
                    'owner:', 'permission:', 'date_conditions:', 'box_name:'
                ]

                # File is JIL if it contains at least 2 JIL keywords
                keyword_count = sum(1 for keyword in jil_keywords if keyword in content)
                return keyword_count >= 2

        except Exception as e:
            logger.debug(f"Could not read file {file_path}: {e}")
            return False

    def _split_job_blocks(self, content: str) -> List[str]:
        """
        Split JIL content into individual job blocks

        Each job starts with: insert_job: job_name or update_job: job_name
        """
        blocks = []

        # Split by insert_job or update_job
        pattern = r'(insert_job:|update_job:)'
        parts = re.split(pattern, content)

        current_block = ""
        for i, part in enumerate(parts):
            if part in ['insert_job:', 'update_job:']:
                if current_block.strip():
                    blocks.append(current_block.strip())
                current_block = part
            else:
                current_block += part

        # Add last block
        if current_block.strip():
            blocks.append(current_block.strip())

        return blocks

    def _parse_job_block(self, block: str, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse individual job block

        Returns:
            Dict with job information
        """
        lines = block.split('\n')

        job_data = {
            "file_path": file_path,
            "raw_block": block,
        }

        # First line should have job name
        first_line = lines[0].strip()
        job_name_match = re.search(r'(insert_job|update_job):\s*(.+)', first_line)
        if job_name_match:
            job_data["job_name"] = job_name_match.group(2).strip()
        else:
            return None

        # Parse key-value pairs
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Pattern: key: value
            match = re.match(r'([a-z_]+):\s*(.+)', line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip().strip('"')

                job_data[key] = value

        return job_data

    def _extract_dependencies(self, job_data: Dict) -> List[Dict[str, str]]:
        """
        Extract job dependencies from condition field

        Autosys conditions like:
        - success(job_a) or s(job_a)
        - done(job_b) or d(job_b)
        - failure(job_c) or f(job_c)
        - s(job_a) & s(job_b)
        """
        flows = []

        condition = job_data.get("condition", "")
        if not condition:
            return flows

        job_name = job_data.get("job_name", "")

        # Pattern 1: Shorthand notation - s(), f(), d(), r(), t(), n()
        # Most common in your environment
        shorthand_pattern = r'([sfdrntp])\(([^)]+)\)'
        shorthand_matches = re.findall(shorthand_pattern, condition, re.IGNORECASE)

        shorthand_map = {
            's': 'success',
            'f': 'failure',
            'd': 'done',
            'r': 'running',
            'n': 'notrunning',
            't': 'terminated',
            'p': 'pending'
        }

        for match in shorthand_matches:
            shorthand = match[0].lower()
            dependent_job = match[1].strip()

            condition_type = shorthand_map.get(shorthand, shorthand)

            flow = {
                "source_job": dependent_job,
                "target_job": job_name,
                "condition_type": condition_type,
                "condition_full": condition,
            }

            flows.append(flow)

        # Pattern 2: Full word notation - success(), failure(), done()
        # For compatibility with other Autosys environments
        if not shorthand_matches:  # Only check if shorthand didn't match
            full_pattern = r'(success|done|failure|running|terminated|notrunning)\(([^)]+)\)'
            full_matches = re.findall(full_pattern, condition, re.IGNORECASE)

            for match in full_matches:
                condition_type = match[0].lower()
                dependent_job = match[1].strip()

                flow = {
                    "source_job": dependent_job,
                    "target_job": job_name,
                    "condition_type": condition_type,
                    "condition_full": condition,
                }

                flows.append(flow)

        return flows

    def _convert_to_models(
        self,
        jobs: List[Dict],
        flows: List[Dict],
        base_path: str
    ) -> tuple:
        """
        Convert Autosys jobs to Process/Component models

        Creates:
        - One Process per job workflow
        - Components for each job
        """
        processes = []
        components = []

        # Group jobs by workflow (jobs without dependencies = separate workflows)
        # For simplicity, create one process per job for now
        # In production, you'd group related jobs

        for job in jobs:
            job_name = job.get("job_name", "unknown")
            command = job.get("command", "")
            description = job.get("description", "")
            job_type = job.get("job_type", "CMD")

            # Generate unique ID
            job_hash = hashlib.md5(f"{job_name}_{command}".encode()).hexdigest()[:8]
            process_id = f"autosys_{job_name}_{job_hash}"

            # Check if command references Ab Initio
            abinitio_graph = self._extract_abinitio_graph(command)

            # Create Process
            process = Process(
                id=process_id,
                name=job_name,
                system=SystemType.AUTOSYS,
                process_type=ProcessType.AUTOSYS_JOB,
                file_path=job.get("file_path", ""),
                description=description or f"Autosys job: {job_name}",
                component_ids=[process_id],  # Self-reference
            )

            processes.append(process)

            # Create Component
            component = Component(
                id=process_id,
                name=job_name,
                component_type=ComponentType.SHELL_SCRIPT if job_type == "CMD" else ComponentType.UNKNOWN,
                system="autosys",
                file_path=job.get("file_path", ""),
                process_id=process_id,
                process_name=job_name,
                business_description=description,
                parameters={
                    "command": command,
                    "job_type": job_type,
                    "machine": job.get("machine", ""),
                    "owner": job.get("owner", ""),
                    "start_times": job.get("start_times", ""),
                    "days_of_week": job.get("days_of_week", ""),
                    "condition": job.get("condition", ""),
                    "abinitio_graph": abinitio_graph or "",
                },
            )

            components.append(component)

        return (processes, components)

    def _extract_abinitio_graph(self, command: str) -> Optional[str]:
        """
        Extract Ab Initio graph name/path from command

        Common patterns:
        - air sandbox run <graph_path>
        - /opt/abinitio/bin/air sandbox run graph.mp
        - run_graph.sh graph_name
        - runpset.ksh -P pset_name.pset (converts to graph_name)
        """
        if not command:
            return None

        # Look for runpset.ksh -P pattern (most common in your environment)
        pset_match = re.search(r'runpset\.ksh\s+-P\s+([a-zA-Z0-9_\-]+)\.pset', command)
        if pset_match:
            # Extract pset name and convert to graph name
            pset_name = pset_match.group(1)
            # Remove .pset extension if present, add .mp
            return pset_name

        # Look for .pset file references (without runpset.ksh)
        pset_file_match = re.search(r'([a-zA-Z0-9_\-/]+)\.pset', command)
        if pset_file_match:
            return pset_file_match.group(1)

        # Look for .mp file references
        mp_match = re.search(r'([a-zA-Z0-9_\-/]+\.mp)', command)
        if mp_match:
            return mp_match.group(1)

        # Look for .plan file references (Data Ingestion graphs)
        plan_match = re.search(r'([a-zA-Z0-9_\-/]+\.plan)', command)
        if plan_match:
            return plan_match.group(1)

        # Look for air sandbox run command
        air_match = re.search(r'air\s+sandbox\s+run\s+([a-zA-Z0-9_\-/]+)', command)
        if air_match:
            return air_match.group(1)

        # Look for graph name patterns
        graph_match = re.search(r'graph[=\s]+([a-zA-Z0-9_\-]+)', command, re.IGNORECASE)
        if graph_match:
            return graph_match.group(1)

        return None

    def build_workflow_flow(self, jobs: List[Dict], flows: List[Dict]) -> Dict[str, Any]:
        """
        Build workflow execution flow from jobs and dependencies

        Returns:
            Dict with workflow structure and execution order
        """
        # Build dependency graph
        job_map = {j["job_name"]: j for j in jobs}

        # Find entry points (jobs with no dependencies)
        dependent_jobs = {f["target_job"] for f in flows}
        entry_jobs = [j for j in jobs if j["job_name"] not in dependent_jobs]

        workflow = {
            "entry_jobs": [j["job_name"] for j in entry_jobs],
            "total_jobs": len(jobs),
            "total_dependencies": len(flows),
            "dependency_graph": flows,
            "abinitio_graphs": []
        }

        # Extract all Ab Initio graphs referenced
        for job in jobs:
            command = job.get("command", "")
            graph = self._extract_abinitio_graph(command)
            if graph:
                workflow["abinitio_graphs"].append({
                    "job_name": job["job_name"],
                    "graph_path": graph,
                })

        return workflow
