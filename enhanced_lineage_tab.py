"""
Enhanced Lineage Tracking Tab
==============================
AI-driven table and column lineage with complete history

Features:
- Auto-populated dropdowns for tables and columns
- Complete multi-hop lineage across all repos
- AI graph view (visual flow)
- AI table view (textual summary)
- AI summary (natural language description)
"""

import streamlit as st
import pandas as pd
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from services.metadata_extractor import MetadataExtractor
from services.codebase_copilot import CodebaseCopilot
from services.lineage.column_journey_tracker import ColumnJourneyTracker
from loguru import logger


@dataclass
class LineageHop:
    """Single hop in lineage journey"""
    hop_number: int
    source_table: str
    source_column: Optional[str]
    transformation: str
    transformation_type: str
    script_name: str
    script_path: str
    target_table: str
    target_column: Optional[str]
    system: str


def render_enhanced_lineage_tab():
    """
    Render enhanced lineage tracking tab with dropdowns and graph view
    """
    st.subheader("🔍 AI-Driven Lineage Tracking")

    st.info("""
    **Enhanced Lineage Tracking** traces complete data flow across all repositories.

    **Features:**
    - 🔽 Auto-populated dropdowns for tables and columns
    - 🔄 Complete multi-hop lineage (source → transform → target → ...)
    - 📊 AI graph view showing visual flow
    - 📋 AI table view with hop-by-hop details
    - 💡 AI summary in natural language

    **How it works:**
    1. Select system and table from dropdown
    2. Optionally select a specific column
    3. AI traces complete lineage across all repos
    4. Shows where data originated, how it transformed, where it went
    """)

    st.markdown("---")

    # Initialize services
    if 'metadata_extractor' not in st.session_state:
        indexer = st.session_state.get('indexer')
        ai_analyzer = st.session_state.get('ai_analyzer')

        st.session_state.metadata_extractor = MetadataExtractor(
            indexer=indexer,
            ai_analyzer=ai_analyzer
        )

    if 'copilot' not in st.session_state:
        indexer = st.session_state.get('indexer')
        ai_analyzer = st.session_state.get('ai_analyzer')

        st.session_state.copilot = CodebaseCopilot(
            indexer=indexer,
            ai_analyzer=ai_analyzer
        )

    if 'column_journey_tracker' not in st.session_state:
        indexer = st.session_state.get('indexer')
        ai_analyzer = st.session_state.get('ai_analyzer')

        st.session_state.column_journey_tracker = ColumnJourneyTracker(
            indexer=indexer,
            ai_analyzer=ai_analyzer
        )

    # Input section
    st.markdown("### 📋 Step 1: Select Entity to Trace")

    col1, col2 = st.columns(2)

    with col1:
        system = st.selectbox(
            "Select System",
            options=["Hadoop", "Databricks", "Ab Initio"],
            key="lineage_system"
        )

        # Map display name to internal name
        system_map = {
            "Hadoop": "hadoop",
            "Databricks": "databricks",
            "Ab Initio": "abinitio"
        }
        internal_system = system_map[system]

        # Get tables for this system
        with st.spinner(f"Loading tables from {system}..."):
            tables = st.session_state.metadata_extractor.get_tables(internal_system)

        if not tables:
            st.warning(f"No tables found in {system}. Make sure the system is indexed.")
            tables = []

        selected_table = st.selectbox(
            f"Select Table/Dataset ({len(tables)} available)",
            options=[""] + tables,
            format_func=lambda x: "Select a table..." if x == "" else x,
            key="lineage_table"
        )

    with col2:
        lineage_type = st.radio(
            "Lineage Type",
            options=["Table-Level", "Column-Level"],
            help="Table-Level: Trace table transformations\nColumn-Level: Trace specific column",
            key="lineage_type"
        )

        # Column selection (only for column-level)
        selected_column = None
        if lineage_type == "Column-Level" and selected_table:
            with st.spinner(f"Loading columns from {selected_table}..."):
                columns = st.session_state.metadata_extractor.get_columns(
                    internal_system,
                    selected_table
                )

            if not columns:
                st.warning("No columns found. Enter column name manually:")
                selected_column = st.text_input(
                    "Column Name",
                    placeholder="e.g., SSN, customer_id",
                    key="lineage_column_manual"
                )
            else:
                selected_column = st.selectbox(
                    f"Select Column ({len(columns)} available)",
                    options=[""] + columns,
                    format_func=lambda x: "Select a column..." if x == "" else x,
                    key="lineage_column"
                )

    # Additional options
    with st.expander("⚙️ Advanced Options"):
        max_hops = st.slider(
            "Maximum Hops",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum number of transformation hops to trace",
            key="lineage_max_hops"
        )

        include_systems = st.multiselect(
            "Search in Systems",
            options=["Hadoop", "Databricks", "Ab Initio"],
            default=[system],
            help="Systems to search for lineage",
            key="lineage_include_systems"
        )

    # Trace button
    if st.button("🔍 Trace Complete Lineage", type="primary", use_container_width=True):
        if not selected_table:
            st.error("Please select a table")
            return

        if lineage_type == "Column-Level" and not selected_column:
            st.error("Please select or enter a column name")
            return

        # Perform lineage tracing
        perform_lineage_tracing(
            system=internal_system,
            table=selected_table,
            column=selected_column if lineage_type == "Column-Level" else None,
            max_hops=max_hops,
            include_systems=[system_map[s] for s in include_systems]
        )

    # Display results
    if 'lineage_results' in st.session_state and st.session_state.lineage_results:
        render_lineage_results(st.session_state.lineage_results)


def perform_lineage_tracing(
    system: str,
    table: str,
    column: Optional[str],
    max_hops: int,
    include_systems: List[str]
):
    """
    Perform complete lineage tracing with multi-hop support

    Args:
        system: Source system
        table: Table name
        column: Optional column name
        max_hops: Maximum hops to trace
        include_systems: Systems to search
    """
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        entity_desc = f"{table}.{column}" if column else table

        status_text.text(f"🔍 Tracing lineage for {system}.{entity_desc}...")
        progress_bar.progress(10)

        # Use column journey tracker if column-level
        if column:
            logger.info(f"Performing column-level lineage for {column}")

            tracker = st.session_state.column_journey_tracker

            journeys = tracker.track_column_journey(
                column_name=column,
                system=system,
                target_systems=[s for s in include_systems if s != system]
            )

            # Convert column journey to lineage format
            lineage_hops = convert_column_journey_to_lineage(journeys, system)

        else:
            logger.info(f"Performing table-level lineage for {table}")

            # Use copilot for table-level lineage
            lineage_hops = trace_table_lineage(
                system=system,
                table=table,
                max_hops=max_hops,
                include_systems=include_systems
            )

        progress_bar.progress(70)

        # Generate AI summary
        status_text.text("🤖 Generating AI summary...")

        ai_summary = generate_lineage_summary(
            system, table, column, lineage_hops
        )

        progress_bar.progress(90)

        # Build graph data
        status_text.text("📊 Building lineage graph...")

        graph_data = build_lineage_graph(lineage_hops)

        progress_bar.progress(100)

        # Store results
        st.session_state.lineage_results = {
            'system': system,
            'table': table,
            'column': column,
            'lineage_type': 'column' if column else 'table',
            'total_hops': len(lineage_hops),
            'hops': lineage_hops,
            'ai_summary': ai_summary,
            'graph_data': graph_data,
            'timestamp': datetime.now().isoformat()
        }

        status_text.empty()
        progress_bar.empty()

        hop_text = "hop" if len(lineage_hops) == 1 else "hops"
        st.success(f"✓ Traced {len(lineage_hops)} lineage {hop_text} for {entity_desc}")

    except Exception as e:
        st.error(f"Error tracing lineage: {e}")
        logger.error(f"Lineage tracing error: {e}", exc_info=True)
    finally:
        progress_bar.empty()
        status_text.empty()


def trace_table_lineage(
    system: str,
    table: str,
    max_hops: int,
    include_systems: List[str]
) -> List[LineageHop]:
    """
    Trace table-level lineage using copilot with AI semantic understanding

    This finds all scripts that read/write the table and builds
    a multi-hop lineage path by understanding actual data flow.
    """
    copilot = st.session_state.copilot
    ai_analyzer = st.session_state.ai_analyzer

    lineage_hops = []
    visited_scripts = set()  # Avoid duplicates
    tables_to_trace = {table}  # Start with requested table
    all_discovered_tables = set()  # Track all tables in lineage chain

    hop_number = 1

    for hop_iteration in range(max_hops):
        if not tables_to_trace:
            logger.info(f"No more tables to trace after {hop_iteration} iterations")
            break

        current_tables = tables_to_trace.copy()
        tables_to_trace.clear()

        for current_table in current_tables:
            # Find scripts that use this table
            context = copilot.retrieve_context_for_query(
                query=f"table {current_table} FROM INSERT CREATE",
                systems=include_systems,
                context_type="lineage"
            )

            if not context.snippets:
                logger.debug(f"No files found for table {current_table}")
                continue

            # Use AI to understand each script
            for snippet in context.snippets[:5]:  # Top 5 most relevant
                script_path = snippet.get('file_path', snippet.get('file_name', ''))

                # Skip if already processed
                if script_path in visited_scripts:
                    continue

                visited_scripts.add(script_path)

                # Get file content
                file_content = snippet.get('content', '')[:4000]  # Limit to 4000 chars

                # Use AI to understand data flow in this script
                if ai_analyzer and ai_analyzer.enabled and file_content:
                    try:
                        ai_analysis = ai_analyzer.analyze_with_context(
                            query=f"""Analyze this {snippet.get('system', 'unknown')} script and extract data lineage:

1. What tables/files does it READ FROM (input sources)?
2. What tables/files does it WRITE TO (output targets)?
3. What is the main transformation/purpose?
4. What type of operation is this (CREATE TABLE, INSERT, SELECT, ETL, etc.)?

File: {snippet.get('file_name', 'Unknown')}
Content:
{file_content}

Respond in JSON format:
{{
    "input_sources": ["table1", "table2"],
    "output_targets": ["target_table"],
    "transformation": "Description of what this script does",
    "operation_type": "CREATE_TABLE|INSERT|SELECT|ETL|INGESTION|REPORTING"
}}""",
                            context=""
                        )

                        # Parse AI response
                        import json
                        import re

                        # Extract JSON from response
                        response_text = ai_analysis.get('analysis', ai_analysis.get('response', ''))

                        if '```json' in response_text:
                            json_str = response_text.split('```json')[1].split('```')[0].strip()
                        elif '```' in response_text:
                            json_str = response_text.split('```')[1].split('```')[0].strip()
                        elif '{' in response_text:
                            # Find JSON object
                            start = response_text.find('{')
                            end = response_text.rfind('}') + 1
                            json_str = response_text[start:end]
                        else:
                            json_str = response_text

                        try:
                            flow_info = json.loads(json_str)
                        except:
                            logger.warning(f"Could not parse AI response as JSON for {script_path}")
                            # Fallback: extract from text
                            flow_info = {
                                "input_sources": [current_table],
                                "output_targets": [current_table],
                                "transformation": "Transform",
                                "operation_type": "UNKNOWN"
                            }

                        input_sources = flow_info.get('input_sources', [current_table])
                        output_targets = flow_info.get('output_targets', [current_table])
                        transformation = flow_info.get('transformation', 'Transform')
                        operation_type = flow_info.get('operation_type', 'TABLE_TRANSFORM')

                        # Create hops for each input → output pair
                        for input_src in input_sources:
                            for output_tgt in output_targets:
                                # Skip self-references unless it's the only thing
                                if input_src == output_tgt and len(input_sources) > 1:
                                    continue

                                hop = LineageHop(
                                    hop_number=hop_number,
                                    source_table=input_src,
                                    source_column=None,
                                    transformation=transformation[:100],  # Truncate long descriptions
                                    transformation_type=operation_type,
                                    script_name=snippet['file_name'],
                                    script_path=script_path,
                                    target_table=output_tgt,
                                    target_column=None,
                                    system=snippet['system']
                                )

                                lineage_hops.append(hop)
                                hop_number += 1

                                # Add output tables for next iteration tracing
                                if output_tgt not in all_discovered_tables and output_tgt != current_table:
                                    tables_to_trace.add(output_tgt)
                                    all_discovered_tables.add(output_tgt)

                    except Exception as e:
                        logger.error(f"AI analysis failed for {script_path}: {e}")
                        # Fallback: create basic hop
                        hop = LineageHop(
                            hop_number=hop_number,
                            source_table=current_table,
                            source_column=None,
                            transformation="Transform (AI analysis unavailable)",
                            transformation_type="TABLE_TRANSFORM",
                            script_name=snippet['file_name'],
                            script_path=script_path,
                            target_table=current_table,
                            target_column=None,
                            system=snippet['system']
                        )
                        lineage_hops.append(hop)
                        hop_number += 1

                else:
                    # No AI available, create basic hop
                    hop = LineageHop(
                        hop_number=hop_number,
                        source_table=current_table,
                        source_column=None,
                        transformation="Transform",
                        transformation_type="TABLE_TRANSFORM",
                        script_name=snippet['file_name'],
                        script_path=script_path,
                        target_table=current_table,
                        system=snippet['system']
                    )
                    lineage_hops.append(hop)
                    hop_number += 1

    logger.info(f"Traced {len(lineage_hops)} lineage hops across {len(visited_scripts)} scripts")
    return lineage_hops


def convert_column_journey_to_lineage(
    journeys: Dict[str, Any],
    source_system: str
) -> List[LineageHop]:
    """
    Convert column journey format to lineage hop format

    Args:
        journeys: Column journey results from ColumnJourneyTracker
        source_system: Source system name

    Returns:
        List of LineageHop objects
    """
    lineage_hops = []

    for system, journey in journeys.items():
        for step in journey.steps:
            hop = LineageHop(
                hop_number=step.step_number,
                source_table=step.source_table,
                source_column=step.source_column,
                transformation=step.transformation,
                transformation_type=step.transformation_type,
                script_name=step.script_name,
                script_path=step.script_path,
                target_table=step.target_table,
                target_column=step.target_column,
                system=system
            )
            lineage_hops.append(hop)

    return lineage_hops


def generate_lineage_summary(
    system: str,
    table: str,
    column: Optional[str],
    hops: List[LineageHop]
) -> str:
    """
    Generate AI summary of lineage in natural language

    Returns a narrative description of the data flow
    """
    ai_analyzer = st.session_state.get('ai_analyzer')

    if not ai_analyzer or not ai_analyzer.enabled:
        return f"Traced {len(hops)} transformation hops for {table}"

    # Build context from hops
    hops_summary = ""
    for hop in hops[:10]:  # First 10 hops
        hops_summary += f"Hop {hop.hop_number}: {hop.source_table} → {hop.transformation} → {hop.target_table} (in {hop.script_name})\n"

    entity = f"{table}.{column}" if column else table

    prompt = f"""Provide a natural language summary of this data lineage.

Entity: {system}.{entity}
Total Hops: {len(hops)}

Lineage Path:
{hops_summary}

Write a 2-3 sentence summary explaining:
1. Where the data originates
2. How it transforms through the pipeline
3. Where it ends up

Make it clear and business-friendly.
"""

    try:
        response = ai_analyzer.client.chat.completions.create(
            model=ai_analyzer.deployment_name,
            messages=[
                {"role": "system", "content": "You are a data lineage expert. Explain lineage in clear, business-friendly language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=300
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.debug(f"AI summary generation failed: {e}")
        return f"Data flows through {len(hops)} transformation steps from {system}.{table}"


def build_lineage_graph(hops: List[LineageHop]) -> Dict[str, Any]:
    """
    Build graph data structure for visualization

    Returns dict with nodes and edges for graph rendering
    """
    nodes = []
    edges = []
    node_ids = set()

    for hop in hops:
        # Source node
        source_id = f"{hop.source_table}_{hop.source_column or 'table'}"
        if source_id not in node_ids:
            nodes.append({
                'id': source_id,
                'label': f"{hop.source_table}\n{hop.source_column or ''}" if hop.source_column else hop.source_table,
                'type': 'table' if not hop.source_column else 'column',
                'system': hop.system
            })
            node_ids.add(source_id)

        # Target node
        target_id = f"{hop.target_table}_{hop.target_column or 'table'}"
        if target_id not in node_ids:
            nodes.append({
                'id': target_id,
                'label': f"{hop.target_table}\n{hop.target_column or ''}" if hop.target_column else hop.target_table,
                'type': 'table' if not hop.target_column else 'column',
                'system': hop.system
            })
            node_ids.add(target_id)

        # Edge (transformation)
        edges.append({
            'from': source_id,
            'to': target_id,
            'label': hop.transformation,
            'script': hop.script_name
        })

    return {
        'nodes': nodes,
        'edges': edges
    }


def render_lineage_results(results: Dict[str, Any]):
    """Display lineage results with graph, table, and AI summary"""
    st.markdown("---")
    st.markdown("### 📊 Step 2: Lineage Analysis Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    entity = f"{results['table']}.{results['column']}" if results['column'] else results['table']

    with col1:
        st.metric("Entity", entity)
    with col2:
        st.metric("Lineage Type", results['lineage_type'].title())
    with col3:
        st.metric("Total Hops", results['total_hops'])
    with col4:
        systems_involved = len(set(hop.system for hop in results['hops']))
        st.metric("Systems", systems_involved)

    # Three views: Graph, Table, AI Summary
    view_tabs = st.tabs(["📊 AI Graph View", "📋 AI Table View", "💡 AI Summary"])

    with view_tabs[0]:
        render_graph_view(results['graph_data'], results['hops'])

    with view_tabs[1]:
        render_table_view(results['hops'])

    with view_tabs[2]:
        render_summary_view(results['ai_summary'], results)


def render_graph_view(graph_data: Dict[str, Any], hops: List[LineageHop]):
    """Render graph visualization of lineage"""
    st.markdown("#### 📊 Visual Lineage Flow")

    if not hops:
        st.info("No lineage hops to display")
        return

    st.info("💡 Visual graph rendering coming soon! For now, showing text-based flow.")

    # Text-based flow diagram
    st.markdown("**Complete Data Flow:**")

    for i, hop in enumerate(hops):
        source = f"{hop.source_table}.{hop.source_column}" if hop.source_column else hop.source_table
        target = f"{hop.target_table}.{hop.target_column}" if hop.target_column else hop.target_table

        st.markdown(f"**Hop {hop.hop_number}**: `{source}`")
        st.markdown(f"↓ *{hop.transformation}* (via `{hop.script_name}`)")

        if i == len(hops) - 1:
            st.markdown(f"✅ **Final**: `{target}`")

    # Graph data for export
    with st.expander("📥 Export Graph Data"):
        st.json(graph_data)


def render_table_view(hops: List[LineageHop]):
    """Render table view of lineage hops"""
    st.markdown("#### 📋 Hop-by-Hop Details")

    if not hops:
        st.info("No lineage hops to display")
        return

    # Convert hops to dataframe
    hops_data = []
    for hop in hops:
        source = f"{hop.source_table}.{hop.source_column}" if hop.source_column else hop.source_table
        target = f"{hop.target_table}.{hop.target_column}" if hop.target_column else hop.target_table

        hops_data.append({
            "Hop": hop.hop_number,
            "Source": source,
            "Transformation": hop.transformation,
            "Type": hop.transformation_type,
            "Script": hop.script_name,
            "Target": target,
            "System": hop.system.upper()
        })

    df = pd.DataFrame(hops_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Detailed view for each hop
    st.markdown("**Detailed Information:**")

    for hop in hops:
        with st.expander(f"Hop {hop.hop_number}: {hop.script_name}"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Source**: `{hop.source_table}`")
                if hop.source_column:
                    st.markdown(f"**Column**: `{hop.source_column}`")
                st.markdown(f"**System**: {hop.system.upper()}")

            with col2:
                st.markdown(f"**Target**: `{hop.target_table}`")
                if hop.target_column:
                    st.markdown(f"**Column**: `{hop.target_column}`")
                st.markdown(f"**Type**: {hop.transformation_type}")

            st.markdown(f"**Transformation**: `{hop.transformation}`")
            st.markdown(f"**Script**: `{hop.script_path}`")


def render_summary_view(ai_summary: str, results: Dict[str, Any]):
    """Render AI summary view"""
    st.markdown("#### 💡 AI-Generated Summary")

    st.markdown(ai_summary)

    st.markdown("---")

    st.markdown("**Lineage Overview:**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Entity Traced**: `{results['system']}.{results['table']}`")
        if results['column']:
            st.markdown(f"**Column**: `{results['column']}`")
        st.markdown(f"**Total Transformations**: {results['total_hops']}")

    with col2:
        systems_involved = list(set(hop.system for hop in results['hops']))
        st.markdown(f"**Systems Involved**: {', '.join([s.upper() for s in systems_involved])}")

        scripts_involved = list(set(hop.script_name for hop in results['hops']))
        st.markdown(f"**Scripts Involved**: {len(scripts_involved)}")

    # Export options
    st.markdown("---")
    st.markdown("**Export Lineage:**")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Download as JSON"):
            # Convert hops to dict
            export_data = {
                **results,
                'hops': [asdict(hop) for hop in results['hops']]
            }

            json_str = json.dumps(export_data, indent=2, default=str)

            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"lineage_{results['table']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    with col2:
        if st.button("📊 Download as Excel"):
            hops_data = []
            for hop in results['hops']:
                hops_data.append({
                    "Hop Number": hop.hop_number,
                    "Source Table": hop.source_table,
                    "Source Column": hop.source_column or "",
                    "Transformation": hop.transformation,
                    "Type": hop.transformation_type,
                    "Script": hop.script_name,
                    "Target Table": hop.target_table,
                    "Target Column": hop.target_column or "",
                    "System": hop.system
                })

            df = pd.DataFrame(hops_data)

            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Lineage', index=False)

            output.seek(0)

            st.download_button(
                label="Download Excel",
                data=output,
                file_name=f"lineage_{results['table']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
