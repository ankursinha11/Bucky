"""
System Mapping Tab
==================
Discover equivalent or related system implementations across platforms

Features:
- Cross-system entity mapping (1-to-many)
- AI-powered similarity detection
- File-level links to implementations
- Comparison summaries
"""

import streamlit as st
import pandas as pd
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from services.codebase_copilot import CodebaseCopilot
from services.lineage.lineage_agents import SimilarityAgent
from loguru import logger


def render_system_mapping_tab():
    """
    Render System Mapping tab

    Allows users to find equivalent implementations across systems
    """
    st.subheader("🧩 Cross-System Mapping Discovery")

    st.info("""
    **System Mapping** helps you discover equivalent implementations across platforms.

    **Use Cases:**
    - Find which Databricks pipelines replace an Ab Initio graph
    - Discover all Hadoop workflows that match a Databricks notebook
    - Identify 1-to-many mappings (1 Ab Initio graph → 3 Databricks pipelines)

    **How it works:**
    1. Select source system and enter entity name
    2. AI searches all target systems for similar implementations
    3. Returns ranked matches with similarity scores
    4. Provides AI comparison summary
    """)

    st.markdown("---")

    # Initialize agents if needed
    if 'copilot' not in st.session_state:
        indexer = st.session_state.get('indexer')
        ai_analyzer = st.session_state.get('ai_analyzer')

        st.session_state.copilot = CodebaseCopilot(
            indexer=indexer,
            ai_analyzer=ai_analyzer
        )

    if 'similarity_agent' not in st.session_state:
        indexer = st.session_state.get('indexer')
        logic_comparator = st.session_state.get('comparator')

        st.session_state.similarity_agent = SimilarityAgent(
            indexer=indexer,
            logic_comparator=logic_comparator
        )

    # Input section
    st.markdown("### 📋 Step 1: Select Source Entity")

    col1, col2 = st.columns(2)

    with col1:
        source_system = st.selectbox(
            "Source System",
            options=["Ab Initio", "Hadoop", "Databricks"],
            key="mapping_source_system"
        )

        entity_name = st.text_input(
            f"{source_system} Entity Name",
            placeholder="e.g., customer_load, bdf_download, 100_commGenPrePrep",
            help="Enter workflow, graph, or pipeline name",
            key="mapping_entity_name"
        )

    with col2:
        target_systems = st.multiselect(
            "Search in Target Systems",
            options=["Ab Initio", "Hadoop", "Databricks"],
            default=["Databricks"] if source_system != "Databricks" else ["Hadoop"],
            help="Systems to search for equivalent implementations",
            key="mapping_target_systems"
        )

        min_similarity = st.slider(
            "Minimum Similarity Score",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Filter results by minimum similarity",
            key="mapping_min_similarity"
        )

    # Search button
    if st.button("🔍 Find Equivalent Implementations", type="primary", use_container_width=True):
        if not entity_name:
            st.error("Please enter an entity name")
            return

        if not target_systems:
            st.error("Please select at least one target system")
            return

        # Perform mapping search
        perform_system_mapping(entity_name, source_system, target_systems, min_similarity)

    # Display results
    if 'mapping_results' in st.session_state and st.session_state.mapping_results:
        render_mapping_results(st.session_state.mapping_results)


def perform_system_mapping(
    entity_name: str,
    source_system: str,
    target_systems: List[str],
    min_similarity: float
):
    """
    Perform cross-system mapping search

    Args:
        entity_name: Entity to find equivalents for
        source_system: Source system (Ab Initio, Hadoop, Databricks)
        target_systems: Target systems to search
        min_similarity: Minimum similarity threshold
    """
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Map display names to internal names
    system_map = {
        "Ab Initio": "abinitio",
        "Hadoop": "hadoop",
        "Databricks": "databricks"
    }

    source_sys = system_map[source_system]
    target_sys = [system_map[t] for t in target_systems]

    try:
        status_text.text(f"🔍 Searching for '{entity_name}' in {source_system}...")
        progress_bar.progress(20)

        # Step 1: Use Copilot to get source entity context
        copilot = st.session_state.copilot
        source_context = copilot.retrieve_context_for_query(
            query=entity_name,
            systems=[source_sys],
            context_type="mapping"
        )

        logger.info(f"Found source context from {len(source_context.files_read)} files")

        status_text.text(f"🔍 Searching for equivalents in {len(target_sys)} target systems...")
        progress_bar.progress(50)

        # Step 2: Search each target system
        all_matches = []

        for target_system in target_sys:
            matches = search_target_system(
                entity_name=entity_name,
                source_system=source_sys,
                target_system=target_system,
                min_similarity=min_similarity
            )
            all_matches.extend(matches)

        progress_bar.progress(80)

        # Step 3: Generate AI comparison summaries
        status_text.text("🤖 Generating AI comparison summaries...")

        for match in all_matches:
            match['ai_summary'] = generate_match_summary(
                entity_name, source_sys, match
            )

        progress_bar.progress(100)

        # Store results
        st.session_state.mapping_results = {
            'source_entity': entity_name,
            'source_system': source_system,
            'target_systems': target_systems,
            'matches': all_matches,
            'total_matches': len(all_matches),
            'timestamp': datetime.now().isoformat()
        }

        status_text.empty()
        progress_bar.empty()

        st.success(f"✓ Found {len(all_matches)} equivalent implementations across {len(target_sys)} systems")

    except Exception as e:
        st.error(f"Error performing system mapping: {e}")
        logger.error(f"System mapping error: {e}", exc_info=True)
    finally:
        progress_bar.empty()
        status_text.empty()


def analyze_source_entity_logic(
    entity_name: str,
    source_system: str
) -> Optional[Dict[str, Any]]:
    """
    Deep analysis of source entity workflow/logic

    For Hadoop:
    - If it's a folder, look for workflow.xml (Oozie)
    - Parse workflow to get all actions and scripts
    - Read and analyze each script
    - Extract: tables, transformations, business logic

    Returns:
        Workflow profile with complete logic understanding
    """
    copilot = st.session_state.copilot
    indexer = st.session_state.indexer

    logger.info(f"🔬 Deep analyzing: {entity_name} in {source_system}")

    # Step 1: Search for entity in source system (folder or file)
    search_results = indexer.search_multi_collection(
        query=entity_name,
        collections=[f"{source_system}_collection"],
        top_k=50  # Get more results to find folders
    )

    if f"{source_system}_collection" not in search_results:
        logger.warning(f"No results found for {entity_name}")
        return None

    results = search_results[f"{source_system}_collection"]

    # Step 2: Find workflow.xml or main entity
    workflow_file = None
    entity_folder = None
    all_files_in_entity = []

    for result in results:
        metadata = result.get('metadata', {})
        file_path = metadata.get('absolute_file_path', '')
        file_name = metadata.get('file_name', '')

        # Check if this is a workflow.xml
        if 'workflow.xml' in file_name.lower() and entity_name.lower() in file_path.lower():
            workflow_file = file_path
            entity_folder = str(Path(file_path).parent)
            logger.info(f"✓ Found workflow: {workflow_file}")

        # Collect all files related to entity
        if entity_name.lower() in file_path.lower():
            all_files_in_entity.append({
                'path': file_path,
                'name': file_name,
                'content': result.get('content', ''),
                'metadata': metadata
            })

    # Step 3: Parse workflow if found
    workflow_logic = {
        'entity_name': entity_name,
        'source_system': source_system,
        'workflow_type': 'unknown',
        'scripts': [],
        'transformations': [],
        'input_tables': set(),
        'output_tables': set(),
        'business_logic': [],
        'data_flow': []
    }

    if workflow_file and Path(workflow_file).exists():
        # Parse Oozie workflow
        workflow_logic = parse_oozie_workflow_deep(workflow_file, entity_folder)
    elif all_files_in_entity:
        # Analyze as standalone script/folder
        workflow_logic = analyze_standalone_entity(entity_name, all_files_in_entity, source_system)
    else:
        logger.warning(f"No workflow or files found for: {entity_name}")
        return None

    # Step 4: Read and analyze all scripts in workflow
    for script_info in workflow_logic['scripts']:
        script_path = script_info.get('path')
        if script_path and Path(script_path).exists():
            script_logic = analyze_script_logic(script_path, source_system)
            script_info['logic'] = script_logic

            # Aggregate logic
            workflow_logic['transformations'].extend(script_logic.get('transformations', []))
            workflow_logic['input_tables'].update(script_logic.get('input_tables', []))
            workflow_logic['output_tables'].update(script_logic.get('output_tables', []))
            workflow_logic['business_logic'].extend(script_logic.get('business_logic', []))

    # Convert sets to lists for JSON serialization
    workflow_logic['input_tables'] = list(workflow_logic['input_tables'])
    workflow_logic['output_tables'] = list(workflow_logic['output_tables'])

    logger.info(f"✓ Analyzed {len(workflow_logic['scripts'])} scripts, {len(workflow_logic['transformations'])} transformations")

    return workflow_logic


def parse_oozie_workflow_deep(workflow_file: str, base_folder: str) -> Dict[str, Any]:
    """Parse Oozie workflow.xml and extract all actions and scripts"""
    import xml.etree.ElementTree as ET

    logger.info(f"📄 Parsing Oozie workflow: {workflow_file}")

    workflow_logic = {
        'entity_name': Path(base_folder).name,
        'source_system': 'hadoop',
        'workflow_type': 'oozie',
        'scripts': [],
        'actions': [],
        'transformations': [],
        'input_tables': set(),
        'output_tables': set(),
        'business_logic': [],
        'data_flow': []
    }

    try:
        tree = ET.parse(workflow_file)
        root = tree.getroot()

        # Parse all actions
        for action in root.findall('.//{*}action'):
            action_name = action.get('name', 'unknown')

            # Pig action
            pig_elem = action.find('{*}pig')
            if pig_elem is not None:
                script_elem = pig_elem.find('{*}script')
                if script_elem is not None:
                    script_path = script_elem.text
                    full_path = str(Path(base_folder) / script_path) if not Path(script_path).is_absolute() else script_path

                    workflow_logic['scripts'].append({
                        'action_name': action_name,
                        'type': 'pig',
                        'path': full_path,
                        'name': Path(script_path).name
                    })

            # Hive action
            hive_elem = action.find('{*}hive')
            if hive_elem is not None:
                script_elem = hive_elem.find('{*}script')
                if script_elem is not None:
                    script_path = script_elem.text
                    full_path = str(Path(base_folder) / script_path) if not Path(script_path).is_absolute() else script_path

                    workflow_logic['scripts'].append({
                        'action_name': action_name,
                        'type': 'hive',
                        'path': full_path,
                        'name': Path(script_path).name
                    })

            # Shell action
            shell_elem = action.find('{*}shell')
            if shell_elem is not None:
                exec_elem = shell_elem.find('{*}exec')
                if exec_elem is not None:
                    script_name = exec_elem.text
                    full_path = str(Path(base_folder) / script_name) if not Path(script_name).is_absolute() else script_name

                    workflow_logic['scripts'].append({
                        'action_name': action_name,
                        'type': 'shell',
                        'path': full_path,
                        'name': script_name
                    })

            # Spark action
            spark_elem = action.find('{*}spark')
            if spark_elem is not None:
                jar_elem = spark_elem.find('{*}jar') or spark_elem.find('{*}class')
                if jar_elem is not None:
                    workflow_logic['scripts'].append({
                        'action_name': action_name,
                        'type': 'spark',
                        'path': jar_elem.text,
                        'name': Path(jar_elem.text).name
                    })

        logger.info(f"  ✓ Found {len(workflow_logic['scripts'])} actions in workflow")

    except Exception as e:
        logger.error(f"Error parsing workflow.xml: {e}")

    return workflow_logic


def analyze_standalone_entity(entity_name: str, files: List[Dict], source_system: str) -> Dict[str, Any]:
    """Analyze entity that's not a workflow (standalone script or folder of scripts)"""
    logger.info(f"📦 Analyzing standalone entity: {entity_name}")

    workflow_logic = {
        'entity_name': entity_name,
        'source_system': source_system,
        'workflow_type': 'standalone',
        'scripts': [],
        'transformations': [],
        'input_tables': set(),
        'output_tables': set(),
        'business_logic': [],
        'data_flow': []
    }

    for file_info in files:
        file_path = file_info['path']
        file_name = file_info['name']

        # Determine script type
        script_type = 'unknown'
        if file_name.endswith('.pig'):
            script_type = 'pig'
        elif file_name.endswith(('.hive', '.sql', '.hql')):
            script_type = 'hive'
        elif file_name.endswith('.py'):
            script_type = 'python'
        elif file_name.endswith('.sh'):
            script_type = 'shell'

        if script_type != 'unknown':
            workflow_logic['scripts'].append({
                'action_name': file_name,
                'type': script_type,
                'path': file_path,
                'name': file_name
            })

    return workflow_logic


def analyze_script_logic(script_path: str, source_system: str) -> Dict[str, Any]:
    """Deep analysis of a single script to extract logic"""
    logger.info(f"  📄 Analyzing script: {Path(script_path).name}")

    script_logic = {
        'transformations': [],
        'input_tables': [],
        'output_tables': [],
        'business_logic': []
    }

    if not Path(script_path).exists():
        logger.warning(f"    ⚠️ Script not found: {script_path}")
        return script_logic

    try:
        with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Use AI analyzer if available
        ai_analyzer = st.session_state.get('ai_analyzer')
        if ai_analyzer and ai_analyzer.enabled:
            script_logic = ai_analyze_script_logic(content, script_path, ai_analyzer)
        else:
            # Fallback to regex parsing
            script_logic = regex_analyze_script_logic(content, script_path)

    except Exception as e:
        logger.error(f"    Error analyzing {script_path}: {e}")

    return script_logic


def ai_analyze_script_logic(content: str, script_path: str, ai_analyzer) -> Dict[str, Any]:
    """Use AI to deeply understand script logic"""
    logger.info(f"    🤖 AI analyzing logic...")

    prompt = f"""Analyze this script and extract its business logic:

Script: {Path(script_path).name}

Content (first 3000 chars):
{content[:3000]}

Extract:
1. Input tables/datasets (list them)
2. Output tables/datasets (list them)
3. Key transformations (FILTER, JOIN, GROUP BY, etc.)
4. Business logic description (what does this script do?)

Format as JSON:
{{
  "input_tables": ["table1", "table2"],
  "output_tables": ["output_table"],
  "transformations": ["FILTER invalid records", "JOIN with reference data", "AGGREGATE by customer"],
  "business_logic": ["Cleanses customer data", "Validates SSN format", "Aggregates monthly totals"]
}}
"""

    try:
        response = ai_analyzer.client.chat.completions.create(
            model=ai_analyzer.deployment_name,
            messages=[
                {"role": "system", "content": "You are a data pipeline expert. Extract business logic from scripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        result_text = response.choices[0].message.content.strip()

        # Try to parse as JSON
        import json
        result = json.loads(result_text)

        logger.info(f"    ✓ AI found: {len(result.get('input_tables', []))} inputs, {len(result.get('transformations', []))} transformations")

        return result

    except Exception as e:
        logger.warning(f"    AI analysis failed: {e}, falling back to regex")
        return regex_analyze_script_logic(content, script_path)


def regex_analyze_script_logic(content: str, script_path: str) -> Dict[str, Any]:
    """Regex-based script analysis (fallback)"""
    import re

    script_logic = {
        'transformations': [],
        'input_tables': [],
        'output_tables': [],
        'business_logic': []
    }

    # Detect script type
    if script_path.endswith('.pig'):
        # Pig patterns
        loads = re.findall(r"LOAD\s+'([^']+)'", content, re.IGNORECASE)
        stores = re.findall(r"STORE\s+\w+\s+INTO\s+'([^']+)'", content, re.IGNORECASE)
        filters = re.findall(r"FILTER\s+", content, re.IGNORECASE)
        joins = re.findall(r"JOIN\s+", content, re.IGNORECASE)
        groups = re.findall(r"GROUP\s+", content, re.IGNORECASE)

        script_logic['input_tables'] = loads
        script_logic['output_tables'] = stores
        script_logic['transformations'] = [f"{len(filters)} FILTER", f"{len(joins)} JOIN", f"{len(groups)} GROUP"]

    elif script_path.endswith(('.hive', '.sql')):
        # Hive/SQL patterns
        froms = re.findall(r"FROM\s+(\w+)", content, re.IGNORECASE)
        inserts = re.findall(r"INSERT\s+(?:INTO|OVERWRITE)\s+TABLE\s+(\w+)", content, re.IGNORECASE)

        script_logic['input_tables'] = froms
        script_logic['output_tables'] = inserts

    elif script_path.endswith('.py'):
        # PySpark patterns
        reads = re.findall(r'spark\.read\.table\(["\'](\w+)["\']\)', content)
        writes = re.findall(r'\.saveAsTable\(["\'](\w+)["\']\)', content)

        script_logic['input_tables'] = reads
        script_logic['output_tables'] = writes

    return script_logic


def search_by_logic_patterns(
    source_logic: Dict[str, Any],
    target_system: str,
    min_similarity: float
) -> List[Dict[str, Any]]:
    """
    Search target system for entities with matching LOGIC patterns using CHUNK-BASED approach

    NEW APPROACH:
    1. Break source workflow into chunks (per script)
    2. Search for matches for EACH chunk separately
    3. Aggregate all matches
    4. Don't require matching the entire workflow - partial matches are good!
    """
    logger.info(f"🎯 Searching {target_system} for logic patterns (chunk-based)...")

    indexer = st.session_state.indexer

    # CHUNK 1: Search by individual scripts (if workflow)
    all_matches = {}  # file_path -> match dict

    if source_logic.get('scripts'):
        logger.info(f"  📦 Source has {len(source_logic['scripts'])} scripts - searching for each chunk")

        for script_info in source_logic['scripts']:
            script_logic = script_info.get('logic', {})
            if not script_logic:
                continue

            # Search for this specific script's logic
            chunk_matches = search_for_chunk(
                chunk_logic=script_logic,
                chunk_name=script_info.get('name', 'unknown'),
                target_system=target_system,
                min_similarity=min_similarity * 0.7,  # Lower threshold for chunks
                indexer=indexer
            )

            # Merge matches
            for match in chunk_matches:
                file_path = match['file_path']
                if file_path in all_matches:
                    # Increase score if multiple chunks match same file
                    all_matches[file_path]['similarity_score'] = max(
                        all_matches[file_path]['similarity_score'],
                        match['similarity_score']
                    )
                    all_matches[file_path]['matched_chunks'] = all_matches[file_path].get('matched_chunks', 1) + 1
                else:
                    match['matched_chunks'] = 1
                    all_matches[file_path] = match

    # CHUNK 2: Also search by entity name (folder-level match)
    entity_name = source_logic.get('entity_name', '')
    if entity_name:
        name_matches = search_by_entity_name(
            entity_name=entity_name,
            target_system=target_system,
            indexer=indexer
        )

        for match in name_matches:
            file_path = match['file_path']
            if file_path not in all_matches:
                all_matches[file_path] = match

    # CHUNK 3: Search by overall workflow pattern (tables + transformations)
    pattern_matches = search_by_overall_pattern(
        source_logic=source_logic,
        target_system=target_system,
        min_similarity=min_similarity,
        indexer=indexer
    )

    for match in pattern_matches:
        file_path = match['file_path']
        if file_path in all_matches:
            # Boost score if both chunk AND pattern match
            all_matches[file_path]['similarity_score'] = min(
                (all_matches[file_path]['similarity_score'] + match['similarity_score']) / 2 * 1.2,
                1.0
            )
        else:
            all_matches[file_path] = match

    # Convert to list and sort
    matches = list(all_matches.values())
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)

    logger.info(f"  ✓ Found {len(matches)} chunk-based matches (from {len(source_logic.get('scripts', []))} source chunks)")

    return matches


def search_for_chunk(
    chunk_logic: Dict[str, Any],
    chunk_name: str,
    target_system: str,
    min_similarity: float,
    indexer
) -> List[Dict[str, Any]]:
    """Search for a single chunk/script logic in target system"""

    # Build search query from chunk logic
    search_parts = []

    if chunk_logic.get('transformations'):
        search_parts.append(" ".join(str(t) for t in chunk_logic['transformations'][:3]))

    if chunk_logic.get('input_tables'):
        # Extract just table names, not full paths
        table_names = [Path(t).name for t in chunk_logic['input_tables'][:2]]
        search_parts.append(" ".join(table_names))

    if chunk_logic.get('business_logic'):
        search_parts.append(" ".join(chunk_logic['business_logic'][:2]))

    if not search_parts:
        return []

    search_query = " ".join(search_parts)
    logger.info(f"    🔍 Searching for chunk '{chunk_name}': {search_query[:100]}...")

    # Search
    results = indexer.search_multi_collection(
        query=search_query,
        collections=[f"{target_system}_collection"],
        top_k=20
    )

    if f"{target_system}_collection" not in results:
        return []

    # Analyze candidates
    matches = []

    for result in results[f"{target_system}_collection"]:
        metadata = result.get('metadata', {})
        file_path = metadata.get('absolute_file_path', '')
        file_name = metadata.get('file_name', 'unknown')

        if not file_path or not Path(file_path).exists():
            continue

        # Analyze target
        target_logic = analyze_script_logic(file_path, target_system)

        # Calculate similarity for this chunk (using AI if available)
        ai_analyzer = st.session_state.get('ai_analyzer')
        chunk_score = calculate_logic_similarity(chunk_logic, target_logic, ai_analyzer)

        if chunk_score >= min_similarity:
            match = {
                'target_system': target_system,
                'entity_name': file_name,
                'file_path': file_path,
                'similarity_score': chunk_score,
                'description': f"Matches chunk '{chunk_name}': {', '.join(target_logic.get('transformations', [])[:3])}",
                'metadata': metadata,
                'matched_chunk': chunk_name,
                'logic_breakdown': {
                    'input_tables': target_logic.get('input_tables', []),
                    'output_tables': target_logic.get('output_tables', []),
                    'transformations': target_logic.get('transformations', [])
                }
            }
            matches.append(match)

    logger.info(f"      ✓ Chunk '{chunk_name}' matched {len(matches)} files")

    return matches


def search_by_entity_name(
    entity_name: str,
    target_system: str,
    indexer
) -> List[Dict[str, Any]]:
    """Search by entity/folder name (lightweight name-based search)"""

    logger.info(f"  📁 Searching for entity name: {entity_name}")

    results = indexer.search_multi_collection(
        query=entity_name,
        collections=[f"{target_system}_collection"],
        top_k=10
    )

    if f"{target_system}_collection" not in results:
        return []

    matches = []

    for result in results[f"{target_system}_collection"]:
        metadata = result.get('metadata', {})
        file_path = metadata.get('absolute_file_path', '')
        file_name = metadata.get('file_name', 'unknown')

        # Check if entity name appears in path
        if entity_name.lower() in file_path.lower():
            match = {
                'target_system': target_system,
                'entity_name': file_name,
                'file_path': file_path,
                'similarity_score': 0.6,  # Moderate score for name match
                'description': f"Name match: contains '{entity_name}'",
                'metadata': metadata,
                'logic_breakdown': {}
            }
            matches.append(match)

    logger.info(f"    ✓ Found {len(matches)} name-based matches")

    return matches


def search_by_overall_pattern(
    source_logic: Dict[str, Any],
    target_system: str,
    min_similarity: float,
    indexer
) -> List[Dict[str, Any]]:
    """Search for overall workflow pattern (aggregated logic)"""

    # Build search query from overall pattern
    search_queries = []

    # Transformation pattern
    transformations = source_logic.get('transformations', [])
    if transformations:
        # Sample transformations, not all
        sampled = transformations[::2][:5]  # Every 2nd, max 5
        trans_pattern = " ".join([str(t) for t in sampled])
        search_queries.append(trans_pattern)

    # Table names (extract just names, not paths)
    input_tables = source_logic.get('input_tables', [])
    if input_tables:
        table_names = [Path(t).name for t in input_tables[:3]]
        search_queries.append(" ".join(table_names))

    # Business logic
    business_logic = source_logic.get('business_logic', [])
    if business_logic:
        search_queries.append(" ".join(business_logic[:3]))

    if not search_queries:
        return []

    combined_query = " ".join(search_queries)

    logger.info(f"  🎯 Searching for overall pattern: {combined_query[:150]}...")

    # Search
    results = indexer.search_multi_collection(
        query=combined_query,
        collections=[f"{target_system}_collection"],
        top_k=20
    )

    if f"{target_system}_collection" not in results:
        return []

    # Analyze candidates
    matches = []

    for result in results[f"{target_system}_collection"]:
        metadata = result.get('metadata', {})
        file_path = metadata.get('absolute_file_path', '')
        file_name = metadata.get('file_name', 'unknown')

        if not file_path or not Path(file_path).exists():
            continue

        # Analyze target
        target_logic = analyze_script_logic(file_path, target_system)

        # Calculate similarity (using AI if available)
        ai_analyzer = st.session_state.get('ai_analyzer')
        pattern_score = calculate_logic_similarity(source_logic, target_logic, ai_analyzer)

        if pattern_score >= min_similarity:
            match = {
                'target_system': target_system,
                'entity_name': file_name,
                'file_path': file_path,
                'similarity_score': pattern_score,
                'description': f"Pattern match: {', '.join(target_logic.get('transformations', [])[:3])}",
                'metadata': metadata,
                'logic_breakdown': {
                    'input_tables': target_logic.get('input_tables', []),
                    'output_tables': target_logic.get('output_tables', []),
                    'transformations': target_logic.get('transformations', [])
                }
            }
            matches.append(match)

    logger.info(f"    ✓ Found {len(matches)} pattern-based matches")

    return matches


def ai_calculate_similarity(source: Dict, target: Dict, ai_analyzer) -> Optional[float]:
    """
    Use AI to semantically compare two logic profiles

    This is more powerful than rule-based comparison because AI can understand:
    - Semantic equivalence (different names, same meaning)
    - Functional similarity (same outcome, different approach)
    - Context-aware matching (understands business logic intent)
    """
    try:
        prompt = f"""Compare these two data pipeline workflows and determine their semantic similarity (0.0 to 1.0).

SOURCE WORKFLOW:
- Input Tables: {source.get('input_tables', [])}
- Output Tables: {source.get('output_tables', [])}
- Transformations: {source.get('transformations', [])}
- Business Logic: {source.get('business_logic', [])}

TARGET WORKFLOW:
- Input Tables: {target.get('input_tables', [])}
- Output Tables: {target.get('output_tables', [])}
- Transformations: {target.get('transformations', [])}
- Business Logic: {target.get('business_logic', [])}

Consider:
1. Do they perform similar transformations? (even if expressed differently)
2. Do they work with related data? (similar table names/patterns)
3. Do they achieve similar business outcomes?
4. Ignore minor differences in naming or syntax

Return ONLY a single number between 0.0 and 1.0 representing similarity score.
Examples: 0.0 (completely different), 0.5 (moderately similar), 0.9 (very similar)

Your response (number only):"""

        response = ai_analyzer.client.chat.completions.create(
            model=ai_analyzer.deployment_name,
            messages=[
                {"role": "system", "content": "You are a data pipeline expert. Compare workflows semantically and return a similarity score."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=50
        )

        result_text = response.choices[0].message.content.strip()

        # Extract number
        import re
        match = re.search(r'(\d+\.?\d*)', result_text)
        if match:
            score = float(match.group(1))
            # Ensure 0-1 range
            if score > 1.0:
                score = score / 100.0  # Handle percentage (e.g., 85 -> 0.85)
            return min(max(score, 0.0), 1.0)

        logger.warning(f"Could not parse AI similarity score: {result_text}")
        return None

    except Exception as e:
        logger.warning(f"AI similarity calculation failed: {e}")
        return None


def calculate_logic_similarity(source: Dict, target: Dict, ai_analyzer=None) -> float:
    """
    Calculate similarity based on logic patterns, not just names

    NEW: Uses AI for semantic comparison when available, falls back to rule-based

    Compares:
    - Transformation types (FILTER, JOIN, etc.)
    - Input/output table patterns
    - Business logic overlap
    """
    # TRY AI FIRST (more intelligent)
    if ai_analyzer and ai_analyzer.enabled:
        ai_score = ai_calculate_similarity(source, target, ai_analyzer)
        if ai_score is not None:
            logger.debug(f"    🤖 AI similarity: {ai_score:.2f}")
            return ai_score
        # If AI fails, fall through to rule-based
        logger.debug("    ⚙️ AI failed, using rule-based similarity")

    # FALLBACK: Rule-based similarity
    score = 0.0
    max_score = 0.0

    # Compare transformations (40% weight)
    source_trans = set(str(t).lower() for t in source.get('transformations', []))
    target_trans = set(str(t).lower() for t in target.get('transformations', []))

    if source_trans or target_trans:
        trans_overlap = len(source_trans & target_trans)
        trans_total = len(source_trans | target_trans)
        trans_score = trans_overlap / trans_total if trans_total > 0 else 0
        score += trans_score * 0.4
        max_score += 0.4

    # Compare table usage (30% weight)
    source_tables = set(str(t).lower() for t in source.get('input_tables', []) + source.get('output_tables', []))
    target_tables = set(str(t).lower() for t in target.get('input_tables', []) + target.get('output_tables', []))

    if source_tables or target_tables:
        table_overlap = len(source_tables & target_tables)
        table_score = table_overlap / max(len(source_tables), len(target_tables), 1)
        score += table_score * 0.3
        max_score += 0.3

    # Compare business logic (30% weight)
    source_logic = set(str(l).lower() for l in source.get('business_logic', []))
    target_logic = set(str(l).lower() for l in target.get('business_logic', []))

    if source_logic or target_logic:
        logic_overlap = len(source_logic & target_logic)
        logic_total = len(source_logic | target_logic)
        logic_score = logic_overlap / logic_total if logic_total > 0 else 0
        score += logic_score * 0.3
        max_score += 0.3

    # Normalize to 0-1 range
    final_score = score / max_score if max_score > 0 else 0

    # BOOST: Apply fuzzy matching for transformation names
    source_trans_norm = [normalize_transformation(t) for t in source.get('transformations', [])]
    target_trans_norm = [normalize_transformation(t) for t in target.get('transformations', [])]

    if source_trans_norm and target_trans_norm:
        fuzzy_overlap = count_fuzzy_matches(source_trans_norm, target_trans_norm)
        fuzzy_boost = (fuzzy_overlap / max(len(source_trans_norm), len(target_trans_norm), 1)) * 0.2
        final_score = min(final_score + fuzzy_boost, 1.0)

    return final_score


def normalize_transformation(trans: str) -> str:
    """Normalize transformation names for fuzzy matching"""
    trans_lower = str(trans).lower()

    # Map variations to standard names
    if 'filter' in trans_lower or 'where' in trans_lower:
        return 'filter'
    elif 'join' in trans_lower:
        return 'join'
    elif 'group' in trans_lower or 'aggregate' in trans_lower or 'agg' in trans_lower:
        return 'group'
    elif 'distinct' in trans_lower or 'duplicate' in trans_lower or 'dedup' in trans_lower:
        return 'distinct'
    elif 'sort' in trans_lower or 'order' in trans_lower:
        return 'sort'
    elif 'union' in trans_lower:
        return 'union'
    else:
        return trans_lower


def count_fuzzy_matches(list1: List[str], list2: List[str]) -> int:
    """Count fuzzy matches between two lists"""
    matches = 0
    for item1 in list1:
        for item2 in list2:
            if item1 == item2 or item1 in item2 or item2 in item1:
                matches += 1
                break  # Count each item1 only once

    return matches


def search_target_system(
    entity_name: str,
    source_system: str,
    target_system: str,
    min_similarity: float
) -> List[Dict[str, Any]]:
    """
    Search a target system for equivalent implementations using DEEP LOGIC ANALYSIS

    Process:
    1. Analyze source entity workflow/logic deeply
    2. Extract business logic, transformations, data flow
    3. Search target system for matching LOGIC patterns (not just names)
    4. Rank by logic similarity, not name similarity

    Returns list of matches with similarity scores
    """
    logger.info(f"🔍 Deep logic search: {entity_name} in {source_system} → {target_system}")

    # Step 1: Analyze source entity deeply
    source_logic = analyze_source_entity_logic(entity_name, source_system)

    if not source_logic:
        logger.warning(f"Could not analyze source entity: {entity_name}")
        return []

    logger.info(f"📊 Source workflow profile: {source_logic['workflow_type']}, {len(source_logic['scripts'])} scripts, {len(source_logic['transformations'])} transformations")

    # Step 2: Search target system for matching logic
    matches = search_by_logic_patterns(
        source_logic=source_logic,
        target_system=target_system,
        min_similarity=min_similarity
    )

    logger.info(f"Found {len(matches)} logic-based matches in {target_system}")

    return matches


def generate_match_summary(
    source_entity: str,
    source_system: str,
    match: Dict[str, Any]
) -> str:
    """
    Generate AI summary of why two entities match

    Args:
        source_entity: Source entity name
        source_system: Source system
        match: Match dictionary with target entity info

    Returns:
        AI-generated summary string
    """
    ai_analyzer = st.session_state.get('ai_analyzer')
    if not ai_analyzer or not ai_analyzer.enabled:
        return f"Similar implementation detected (similarity: {match['similarity_score']:.0%})"

    # Read target file content
    target_file = match.get('file_path')
    target_content = ""

    if target_file and Path(target_file).exists():
        try:
            with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                target_content = f.read()[:3000]  # First 3000 chars
        except:
            pass

    prompt = f"""Analyze why these two entities are similar.

Source: {source_system}.{source_entity}
Target: {match['target_system']}.{match['entity_name']}
Similarity Score: {match['similarity_score']:.0%}

Target Entity Content (first 3000 chars):
```
{target_content}
```

Provide a 1-2 sentence summary explaining:
1. What this target entity does
2. Why it's similar to the source entity
3. Any key differences

Keep it concise and actionable.
"""

    try:
        response = ai_analyzer.client.chat.completions.create(
            model=ai_analyzer.deployment_name,
            messages=[
                {"role": "system", "content": "You are a data pipeline expert. Provide concise, actionable summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.debug(f"AI summary generation failed: {e}")
        return f"Similar implementation detected (similarity: {match['similarity_score']:.0%})"


def render_mapping_results(results: Dict[str, Any]):
    """
    Display system mapping results

    Args:
        results: Mapping results dictionary
    """
    st.markdown("---")
    st.markdown("### 📊 Step 2: Mapping Results")

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Source Entity", results['source_entity'])
    with col2:
        st.metric("Total Matches", results['total_matches'])
    with col3:
        avg_similarity = sum(m['similarity_score'] for m in results['matches']) / len(results['matches']) if results['matches'] else 0
        st.metric("Avg Similarity", f"{avg_similarity:.0%}")

    # Group matches by target system
    matches_by_system = {}
    for match in results['matches']:
        system = match['target_system']
        if system not in matches_by_system:
            matches_by_system[system] = []
        matches_by_system[system].append(match)

    # Display each system's matches
    for system, matches in matches_by_system.items():
        st.markdown("---")

        with st.expander(
            f"📊 {system.upper()} - {len(matches)} matches",
            expanded=True
        ):
            # Show mapping type
            if len(matches) == 1:
                st.info("🔵 **1-to-1 Mapping**: Single equivalent implementation found")
            else:
                st.warning(f"🔶 **1-to-Many Mapping**: Source entity maps to {len(matches)} implementations")

            # Display matches as table
            matches_data = []
            for idx, match in enumerate(matches, 1):
                matches_data.append({
                    "#": idx,
                    "Entity Name": match['entity_name'],
                    "Similarity": f"{match['similarity_score']:.1%}",
                    "File Path": match['file_path']
                })

            df = pd.DataFrame(matches_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Detailed view for each match
            st.markdown("**Detailed Analysis:**")

            for idx, match in enumerate(matches, 1):
                with st.container():
                    st.markdown(f"**{idx}. {match['entity_name']}** (Similarity: {match['similarity_score']:.0%})")

                    # AI Summary
                    if 'ai_summary' in match:
                        st.markdown(f"💡 {match['ai_summary']}")

                    # File link
                    if match['file_path']:
                        st.markdown(f"📄 **File**: `{match['file_path']}`")

                    # Show snippet
                    if match.get('description'):
                        with st.expander("View snippet"):
                            st.code(match['description'], language="text")

                    st.markdown("---")

    # Export results
    st.markdown("### 📤 Export Mapping Results")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Download as JSON", use_container_width=True):
            json_str = json.dumps(results, indent=2)

            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"system_mapping_{results['source_entity']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    with col2:
        if st.button("📊 Download as Excel", use_container_width=True):
            # Create Excel with all matches
            all_matches_data = []
            for match in results['matches']:
                all_matches_data.append({
                    "Source Entity": results['source_entity'],
                    "Source System": results['source_system'],
                    "Target Entity": match['entity_name'],
                    "Target System": match['target_system'],
                    "Similarity Score": match['similarity_score'],
                    "File Path": match['file_path'],
                    "AI Summary": match.get('ai_summary', '')
                })

            df = pd.DataFrame(all_matches_data)

            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='System_Mappings', index=False)

            output.seek(0)

            st.download_button(
                label="Download Excel",
                data=output,
                file_name=f"system_mapping_{results['source_entity']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # Comparison view (if multiple matches)
    if results['total_matches'] > 1:
        st.markdown("---")
        st.markdown("### 🔄 Cross-Implementation Comparison")

        st.info("""
        Select up to 3 implementations to compare side-by-side.
        This will show you the differences in logic, transformations, and data flow.
        """)

        # Allow selecting matches for comparison
        selected_matches = st.multiselect(
            "Select implementations to compare:",
            options=[f"{m['target_system']}.{m['entity_name']}" for m in results['matches']],
            max_selections=3,
            key="comparison_selection"
        )

        if len(selected_matches) >= 2:
            if st.button("🔍 Compare Selected", type="primary"):
                st.info("Comparison feature will use LogicComparator to show detailed side-by-side analysis")
                # This would integrate with existing LogicComparator
                # For now, placeholder
