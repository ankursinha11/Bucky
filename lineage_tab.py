"""
STAG Lineage Tracking Tab
==========================
AI-driven lineage tracking UI component for STAG

Features:
- Select entity from any system (Ab Initio, Hadoop, Databricks)
- Generate STTM mappings
- Find equivalent implementations across systems
- Display 3-level lineage (flow, logic, column)
- Export lineage data and STTM mappings
"""

import streamlit as st
import pandas as pd
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from services.lineage.lineage_agents import LineageOrchestrator
from services.lineage.sttm_generator import STTMGenerator
from services.lineage.column_journey_tracker import ColumnJourneyTracker
from services.lineage.enhanced_lineage_tracker import EnhancedLineageTracker
from services.lineage.lineage_visualizer import LineageVisualizer
from services.multi_collection_indexer import MultiCollectionIndexer
from services.ai_script_analyzer import AIScriptAnalyzer
import io

from loguru import logger


def render_lineage_tab():
    """
    Render the Lineage Tracking tab in STAG

    Workflow:
    1. List available entities from all systems
    2. User selects an entity
    3. Analyze lineage (parse + STTM + cross-system matching)
    4. Display results with 3-level lineage
    5. Export options (Excel, JSON)
    """
    st.subheader("🔗 Lineage Tracking & Cross-System Analysis")

    # Add mode selection
    lineage_mode = st.radio(
        "Select Lineage Mode:",
        options=["📊 Workflow Lineage", "🔍 Column Journey", "🌐 Dependency Graph (Enhanced)"],
        horizontal=True,
        help="Workflow Lineage: Analyze a specific workflow/pipeline\nColumn Journey: Trace a column across entire repository\nDependency Graph: Enhanced lineage with interactive graph visualization"
    )

    if lineage_mode == "🔍 Column Journey":
        render_column_journey_tab()
        return
    elif lineage_mode == "🌐 Dependency Graph (Enhanced)":
        render_enhanced_lineage_tab()
        return

    st.markdown("""
    **AI-powered lineage tracking** that:
    - Generates column-level mappings (STTM)
    - Finds equivalent implementations across systems
    - Traces data flow through transformations
    - Compares logic semantically
    """)

    # Initialize lineage orchestrator if not already done
    if 'lineage_orchestrator' not in st.session_state:
        with st.spinner("Initializing Lineage Agents..."):
            try:
                indexer = st.session_state.get('indexer')
                ai_analyzer = AIScriptAnalyzer()

                st.session_state.lineage_orchestrator = LineageOrchestrator(
                    ai_analyzer=ai_analyzer,
                    indexer=indexer
                )
                logger.info("✓ Lineage Orchestrator initialized")
            except Exception as e:
                st.error(f"Error initializing lineage orchestrator: {e}")
                return

    # Main workflow sections
    st.markdown("---")

    # Section 1: Entity Selection
    render_entity_selection()

    # Section 2: Lineage Analysis (if entity selected)
    if 'selected_entity' in st.session_state and st.session_state.selected_entity:
        render_lineage_analysis()


def render_entity_selection():
    """Render entity selection interface"""
    st.markdown("### 📋 Step 1: Select Entity to Analyze")

    # Add explanation box
    st.info("""
    **What is an Entity?**

    An entity is a **workflow, pipeline, graph, or notebook name** - NOT a column or table name.

    Examples:
    - **Hadoop**: `bdf_download`, `customer_pipeline`, `daily_load`
    - **Databricks**: `customer_etl`, `bdf_download`, `data_pipeline`
    - **Ab Initio**: `customer_load.graph`, `100_commGenPrePrep`

    ℹ️ The system will find all scripts/files related to this workflow and extract column-level lineage (like "SSN") from them.
    """)

    col1, col2 = st.columns(2)

    with col1:
        # System selection
        system_type = st.selectbox(
            "Source System",
            options=["Ab Initio", "Hadoop", "Databricks"],
            key="lineage_system_type"
        )

        # Map display names to internal names
        system_map = {
            "Ab Initio": "abinitio",
            "Hadoop": "hadoop",
            "Databricks": "databricks"
        }
        internal_system = system_map[system_type]

    with col2:
        # Entity name input with system-specific placeholder
        placeholders = {
            "Ab Initio": "100_commGenPrePrep or customer_load",
            "Hadoop": "bdf_download or customer_pipeline",
            "Databricks": "bdf_download or customer_etl"
        }

        entity_name = st.text_input(
            f"{system_type} Workflow/Pipeline Name",
            placeholder=f"e.g., {placeholders[system_type]}",
            key="lineage_entity_name",
            help=f"Enter the workflow/pipeline/graph name (NOT a column or table name)"
        )

    # Optional: File path for parsing
    st.markdown("**Optional:** Provide file path for direct parsing")
    file_path = st.text_input(
        "File Path (Optional)",
        placeholder="/path/to/file.mp or upload below",
        key="lineage_file_path"
    )

    # File upload option
    uploaded_file = st.file_uploader(
        "Or Upload File",
        type=['mp', 'xml', 'py', 'ipynb', 'hql', 'pig'],
        key="lineage_file_upload"
    )

    # Target systems for equivalent search
    st.markdown("**Find equivalents in these systems:**")
    target_systems = st.multiselect(
        "Target Systems",
        options=["abinitio", "hadoop", "databricks"],
        default=["hadoop", "databricks"] if internal_system == "abinitio" else ["abinitio"],
        key="lineage_target_systems"
    )

    # Partner/Domain
    partner = st.text_input(
        "Business Partner/Domain",
        value="default",
        key="lineage_partner"
    )

    # Analyze button
    if st.button("🔍 Analyze Lineage", type="primary", use_container_width=True):
        if not entity_name:
            st.error("Please enter an entity name")
            return

        # Handle file upload
        temp_file_path = None
        if uploaded_file:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp())
            temp_file_path = temp_dir / uploaded_file.name
            with open(temp_file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            file_path = str(temp_file_path)

        # Store selection in session state
        st.session_state.selected_entity = {
            "system_type": internal_system,
            "entity_name": entity_name,
            "file_path": file_path if file_path else None,
            "target_systems": target_systems,
            "partner": partner
        }

        # Trigger analysis
        run_lineage_analysis()


def run_lineage_analysis():
    """Run the lineage analysis workflow"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        entity = st.session_state.selected_entity
        orchestrator = st.session_state.lineage_orchestrator

        status_text.text("Step 1/5: Parsing entity...")
        progress_bar.progress(10)

        # Run complete lineage analysis
        result = orchestrator.analyze_lineage(
            system_type=entity["system_type"],
            entity_name=entity["entity_name"],
            file_path=entity["file_path"],
            target_systems=entity["target_systems"],
            partner=entity["partner"]
        )

        progress_bar.progress(100)
        status_text.empty()

        # Store result in session state
        st.session_state.lineage_result = result

        st.success(f"✓ Lineage analysis complete! Found {len(result.sttm_mappings)} column mappings and {len(result.matched_systems)} system matches")
        st.balloons()

    except Exception as e:
        st.error(f"Error during lineage analysis: {e}")
        logger.error(f"Lineage analysis error: {e}", exc_info=True)
    finally:
        progress_bar.empty()


def render_lineage_analysis():
    """Render lineage analysis results"""
    if 'lineage_result' not in st.session_state:
        st.info("Click 'Analyze Lineage' to start analysis")
        return

    result = st.session_state.lineage_result

    st.markdown("---")
    st.markdown("### 📊 Step 2: Lineage Analysis Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("STTM Mappings", len(result.sttm_mappings))
    with col2:
        st.metric("Matched Systems", len(result.matched_systems))
    with col3:
        st.metric("Column Lineage", len(result.column_lineage))
    with col4:
        st.metric("Confidence", f"{result.confidence_score:.0%}")

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 STTM Mappings",
        "🔗 Cross-System Matches",
        "📊 3-Level Lineage",
        "🧠 AI Reasoning",
        "📤 Export"
    ])

    with tab1:
        render_sttm_mappings(result)

    with tab2:
        render_cross_system_matches(result)

    with tab3:
        render_three_level_lineage(result)

    with tab4:
        render_ai_reasoning(result)

    with tab5:
        render_export_options(result)


def render_sttm_mappings(result):
    """Render STTM mappings table"""
    st.markdown("#### 📋 Source-Transform-Target Mappings (STTM)")

    if not result.sttm_mappings:
        st.info("No STTM mappings generated")
        return

    # Convert to DataFrame
    df = pd.DataFrame(result.sttm_mappings)

    # Column selection for display
    display_cols = [
        'target_field_name', 'target_table_name', 'target_field_data_type',
        'source_field_names', 'source_dataset_name', 'transformation_logic',
        'field_type', 'contains_pii', 'confidence_score'
    ]

    available_cols = [col for col in display_cols if col in df.columns]
    df_display = df[available_cols]

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        filter_pii = st.checkbox("Show PII fields only")
    with col2:
        filter_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.1)
    with col3:
        filter_field_type = st.multiselect(
            "Field Type",
            options=df['field_type'].unique().tolist() if 'field_type' in df.columns else [],
            default=None
        )

    # Apply filters
    df_filtered = df_display.copy()
    if filter_pii and 'contains_pii' in df.columns:
        df_filtered = df_filtered[df['contains_pii'] == True]
    if 'confidence_score' in df.columns:
        df_filtered = df_filtered[df['confidence_score'] >= filter_confidence]
    if filter_field_type and 'field_type' in df.columns:
        df_filtered = df_filtered[df['field_type'].isin(filter_field_type)]

    # Display table
    st.dataframe(df_filtered, use_container_width=True, height=400)

    # Statistics
    st.markdown("**Statistics:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        pii_count = df['contains_pii'].sum() if 'contains_pii' in df.columns else 0
        st.metric("PII Fields", pii_count)
    with col2:
        avg_conf = df['confidence_score'].mean() if 'confidence_score' in df.columns else 0
        st.metric("Avg Confidence", f"{avg_conf:.0%}")
    with col3:
        unique_tables = df['target_table_name'].nunique() if 'target_table_name' in df.columns else 0
        st.metric("Unique Tables", unique_tables)


def render_cross_system_matches(result):
    """Render cross-system equivalent implementations"""
    st.markdown("#### 🔗 Equivalent Implementations in Other Systems")

    if not result.matched_systems:
        st.info("No cross-system matches found")
        return

    # Display matches for each system
    for system, match_info in result.matched_systems.items():
        with st.expander(f"🎯 {system.upper()} - Similarity: {match_info.get('similarity_score', 0):.0%}", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Match Details:**")
                st.write(f"**Entity:** {match_info.get('entity', 'unknown')}")
                st.write(f"**Similarity:** {match_info.get('similarity_score', 0):.0%}")
                st.write(f"**Total Matches:** {match_info.get('match_count', 0)}")

            with col2:
                # Similarity gauge
                similarity = match_info.get('similarity_score', 0)
                if similarity >= 0.8:
                    st.success(f"✅ HIGH Similarity: {similarity:.0%}")
                elif similarity >= 0.6:
                    st.warning(f"⚠️ MEDIUM Similarity: {similarity:.0%}")
                else:
                    st.error(f"❌ LOW Similarity: {similarity:.0%}")

            # Show comparison summary
            if 'comparison' in match_info:
                st.markdown("**Comparison:**")
                st.write(match_info['comparison'])


def render_three_level_lineage(result):
    """Render 3-level lineage visualization"""
    st.markdown("#### 📊 Three-Level Lineage Structure")

    # Level tabs
    level_tab1, level_tab2, level_tab3 = st.tabs([
        "🌊 Flow-Level",
        "⚙️ Logic-Level",
        "📍 Column-Level"
    ])

    with level_tab1:
        st.markdown("**Flow-Level Lineage** (Process-to-Process Mapping)")

        if result.flow_level_lineage:
            # Create flow diagram data
            flows_df = pd.DataFrame(result.flow_level_lineage)

            st.dataframe(flows_df, use_container_width=True)

            # Visualization placeholder
            st.info("💡 Visual graph rendering coming soon! Use React Flow or Cytoscape.js")

            # Export flow JSON
            if st.button("Export Flow JSON"):
                flow_json = json.dumps(result.flow_level_lineage, indent=2)
                st.download_button(
                    "Download",
                    data=flow_json,
                    file_name=f"flow_lineage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("No flow-level lineage data")

    with level_tab2:
        st.markdown("**Logic-Level Lineage** (Transformation Comparison)")

        if result.logic_level_lineage:
            logic_df = pd.DataFrame(result.logic_level_lineage)

            # Show transformation summary
            st.dataframe(logic_df, use_container_width=True)

            # Transformation details
            st.markdown("**Transformation Details:**")
            for idx, transform in enumerate(result.logic_level_lineage):
                with st.expander(f"Transform {idx+1}: {transform.get('transformation_type', 'unknown')}"):
                    st.write(f"**Logic:** {transform.get('transformation_logic', 'N/A')}")
                    st.write(f"**Fields:** {transform.get('field_count', 0)}")
                    st.write(f"**Complexity:** {transform.get('complexity', 'UNKNOWN')}")
                    st.write(f"**Source Fields:** {', '.join(transform.get('source_fields', []))}")
                    st.write(f"**Target Fields:** {', '.join(transform.get('target_fields', []))}")
        else:
            st.info("No logic-level lineage data")

    with level_tab3:
        st.markdown("**Column-Level Lineage** (Field-Level Derivation)")

        if result.column_lineage:
            column_df = pd.DataFrame(result.column_lineage)

            # Column selection for detailed view
            selected_column = st.selectbox(
                "Select Column for Details",
                options=column_df['column_name'].tolist() if 'column_name' in column_df.columns else []
            )

            if selected_column:
                # Find column details
                column_detail = next(
                    (c for c in result.column_lineage if c.get('column_name') == selected_column),
                    None
                )

                if column_detail:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Target:**")
                        st.write(f"**Table:** {column_detail.get('target_table', 'N/A')}")
                        st.write(f"**Column:** {column_detail.get('column_name', 'N/A')}")
                        st.write(f"**Data Type:** {column_detail.get('data_type', 'N/A')}")
                        st.write(f"**PII:** {'Yes' if column_detail.get('contains_pii') else 'No'}")

                    with col2:
                        st.markdown("**Source:**")
                        st.write(f"**Dataset:** {column_detail.get('source_dataset', 'N/A')}")
                        st.write(f"**Columns:** {', '.join(column_detail.get('source_columns', []))}")

                    st.markdown("**Transformation:**")
                    st.code(column_detail.get('transformation_logic', 'N/A'), language='text')

                    st.markdown("**Dependencies:**")
                    deps = column_detail.get('dependencies', [])
                    if deps:
                        for dep in deps:
                            st.write(f"- {dep}")
                    else:
                        st.write("No dependencies")

            # Full table view
            st.markdown("**All Column Mappings:**")
            st.dataframe(column_df, use_container_width=True, height=300)
        else:
            st.info("No column-level lineage data")


def render_ai_reasoning(result):
    """Render AI reasoning notes"""
    st.markdown("#### 🧠 AI Reasoning & Insights")

    st.info(result.ai_reasoning_notes)

    # Additional insights
    st.markdown("**Analysis Metadata:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("System", result.selected_system.upper())
    with col2:
        st.metric("Entity", result.selected_entity)
    with col3:
        st.metric("Analyzed At", result.created_at[:10])

    # Show comparisons summary
    if result.comparisons:
        st.markdown("**Comparison Summary:**")
        for level, summary in result.comparisons.items():
            st.write(f"**{level.replace('_', ' ').title()}:** {summary}")


def render_export_options(result):
    """Render export options for lineage data"""
    st.markdown("#### 📤 Export Lineage Data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Export STTM Mappings:**")

        # Excel export
        if st.button("Export STTM to Excel", use_container_width=True):
            try:
                df = pd.DataFrame(result.sttm_mappings)
                output = io.BytesIO()

                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='STTM_Mappings', index=False)

                output.seek(0)

                st.download_button(
                    label="Download Excel",
                    data=output,
                    file_name=f"sttm_mappings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success("✓ Excel ready for download!")
            except Exception as e:
                st.error(f"Export error: {e}")

        # CSV export
        if st.button("Export STTM to CSV", use_container_width=True):
            try:
                df = pd.DataFrame(result.sttm_mappings)
                csv = df.to_csv(index=False)

                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"sttm_mappings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.success("✓ CSV ready for download!")
            except Exception as e:
                st.error(f"Export error: {e}")

    with col2:
        st.markdown("**Export Complete Lineage:**")

        # JSON export (full result)
        if st.button("Export Full Lineage JSON", use_container_width=True):
            try:
                json_data = result.to_json()

                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"lineage_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                st.success("✓ JSON ready for download!")
            except Exception as e:
                st.error(f"Export error: {e}")

        # Save to database option
        if st.button("Save to Metadata Store", use_container_width=True):
            try:
                # Save to outputs directory
                output_dir = Path("./outputs/lineage")
                output_dir.mkdir(parents=True, exist_ok=True)

                filename = f"lineage_{result.selected_system}_{result.selected_entity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = output_dir / filename

                with open(filepath, 'w') as f:
                    f.write(result.to_json())

                st.success(f"✓ Saved to {filepath}")
            except Exception as e:
                st.error(f"Save error: {e}")

    st.markdown("---")
    st.markdown("**Preview JSON Structure:**")

    with st.expander("View JSON Preview"):
        # Fix: Parse JSON first, then display
        json_str = result.to_json()
        try:
            json_data = json.loads(json_str)

            # Create a preview version with limited data
            if len(json_str) > 5000:
                preview_data = {
                    "selected_system": json_data.get("selected_system"),
                    "selected_entity": json_data.get("selected_entity"),
                    "sttm_mappings_count": len(json_data.get("sttm_mappings", [])),
                    "matched_systems_count": len(json_data.get("matched_systems", {})),
                    "column_lineage_count": len(json_data.get("column_lineage", [])),
                    "confidence_score": json_data.get("confidence_score"),
                    "sample_sttm_mapping": json_data.get("sttm_mappings", [{}])[0] if json_data.get("sttm_mappings") else {},
                    "note": "Full JSON is too large. Download to see all data."
                }
                st.json(preview_data)
            else:
                st.json(json_data)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {e}")
            st.text("Raw JSON (first 2000 chars):")
            st.code(json_str[:2000], language="json")


# Helper functions
def load_available_entities():
    """Load list of available entities from indexed data"""
    # This would query the vector database to get list of indexed entities
    # Placeholder for now
    return {
        "abinitio": ["customer_load.graph", "product_transform.graph"],
        "hadoop": ["customer_hive_workflow", "product_pig_script"],
        "databricks": ["customer_notebook", "product_pipeline"]
    }


def render_column_journey_tab():
    """
    Render column journey tracking interface

    Allows users to track a single column's complete path across entire repository
    """
    st.markdown("### 🔍 Track Complete Column Journey")

    st.info("""
    **Column Journey Tracking** traces a single column's complete path across your entire repository.

    **How it works:**
    1. Enter a column name (e.g., "SSN", "customer_id", "order_total")
    2. System searches ALL scripts in the selected system
    3. Uses AI to extract transformation logic at each step
    4. Shows complete flow from source to target
    5. Finds equivalent journey in other systems

    **Example**: Enter "SSN" → See how SSN transforms from raw → staging → master → final
    """)

    st.markdown("---")

    # Initialize tracker if needed
    if 'column_journey_tracker' not in st.session_state:
        indexer = st.session_state.get('indexer')
        ai_analyzer = st.session_state.get('ai_analyzer')

        st.session_state.column_journey_tracker = ColumnJourneyTracker(
            indexer=indexer,
            ai_analyzer=ai_analyzer
        )

    # Input section
    st.markdown("### 📋 Step 1: Select Column to Track")

    col1, col2 = st.columns(2)

    with col1:
        column_name = st.text_input(
            "Column Name",
            placeholder="e.g., SSN, customer_id, order_total",
            help="Enter the column name you want to trace (case-insensitive)",
            key="column_journey_name"
        )

        source_system = st.selectbox(
            "Source System",
            options=["Hadoop", "Databricks", "Ab Initio"],
            key="column_journey_system"
        )

    with col2:
        target_systems = st.multiselect(
            "Find equivalent in:",
            options=["Hadoop", "Databricks", "Ab Initio"],
            default=["Databricks"] if source_system == "Hadoop" else ["Hadoop"],
            help="Systems to search for equivalent column journey",
            key="column_journey_targets"
        )

    # Track button
    if st.button("🔍 Track Column Journey", type="primary", use_container_width=True):
        if not column_name:
            st.error("Please enter a column name")
            return

        # Map display names to internal names
        system_map = {
            "Hadoop": "hadoop",
            "Databricks": "databricks",
            "Ab Initio": "abinitio"
        }

        source_sys = system_map[source_system]
        target_sys = [system_map[t] for t in target_systems]

        # Track journey
        with st.spinner(f"🔍 Searching entire {source_system} repository for '{column_name}'..."):
            tracker = st.session_state.column_journey_tracker

            try:
                journeys = tracker.track_column_journey(
                    column_name=column_name,
                    system=source_sys,
                    target_systems=target_sys
                )

                # Store results
                st.session_state.column_journeys = journeys

                total_steps = sum(j.total_steps for j in journeys.values())
                st.success(f"✓ Column journey complete! Found {total_steps} total transformation steps across {len(journeys)} systems")

            except Exception as e:
                st.error(f"Error tracking column journey: {e}")
                logger.error(f"Column journey error: {e}", exc_info=True)
                return

    # Display results
    if 'column_journeys' in st.session_state and st.session_state.column_journeys:
        render_column_journey_results(st.session_state.column_journeys)


def render_column_journey_results(journeys: Dict[str, Any]):
    """Display column journey results"""
    st.markdown("---")
    st.markdown("### 📊 Step 2: Column Journey Results")

    # Summary metrics
    total_steps = sum(j.total_steps for j in journeys.values())
    avg_confidence = sum(j.confidence for j in journeys.values()) / len(journeys) if journeys else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Systems Analyzed", len(journeys))
    with col2:
        st.metric("Total Steps", total_steps)
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.0%}")

    # Display each system's journey
    for system, journey in journeys.items():
        st.markdown("---")

        with st.expander(
            f"📊 {system.upper()} - {journey.column_name} Journey ({journey.total_steps} steps)",
            expanded=True
        ):
            if journey.total_steps == 0:
                st.warning(f"No transformation steps found for '{journey.column_name}' in {system}")
                st.info(f"Found {len(journey.all_occurrences)} script(s) mentioning '{journey.column_name}', but could not extract transformation flow.")

                # Show occurrences
                if journey.all_occurrences:
                    st.markdown("**Scripts mentioning this column:**")
                    for occ in journey.all_occurrences:
                        st.markdown(f"- `{occ.script_name}` - {occ.transformation or 'unknown transformation'}")

                continue

            # Display journey flowchart
            st.markdown("**Complete Transformation Flow:**")

            for step in journey.steps:
                # Step header
                st.markdown(f"**Step {step.step_number}**: `{step.source_table}.{step.source_column}`")

                # Script and transformation
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown(f"📄 **Script**: `{step.script_name}`")
                    if step.transformation_type:
                        type_emoji = {
                            "STRING_MANIPULATION": "✂️",
                            "PRIVACY": "🔒",
                            "JOIN": "🔗",
                            "FILTER": "🔍",
                            "AGGREGATION": "📊",
                            "TYPE_CONVERSION": "🔄",
                            "OTHER": "⚙️"
                        }
                        emoji = type_emoji.get(step.transformation_type, "⚙️")
                        st.markdown(f"{emoji} **Type**: {step.transformation_type}")

                with col2:
                    st.markdown(f"🔧 **Transformation**: `{step.transformation}`")
                    if step.code_snippet:
                        with st.expander("View code snippet"):
                            st.code(step.code_snippet, language="text")

                # Arrow to next step
                if step.step_number < len(journey.steps):
                    st.markdown("⬇️")
                else:
                    # Final step
                    st.markdown(f"✅ **Final**: `{step.target_table}.{step.target_column}`")

            # Journey metadata
            st.markdown("---")
            st.markdown("**Journey Summary:**")

            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.metric("Total Steps", journey.total_steps)
            with summary_col2:
                st.metric("Source", journey.source_table or "unknown")
            with summary_col3:
                st.metric("Target", journey.target_table or "unknown")

            st.metric("Confidence", f"{journey.confidence:.0%}")

            # Export journey
            st.markdown("**Export this journey:**")

            col1, col2 = st.columns(2)

            with col1:
                if st.button(f"📥 Download {system.upper()} Journey (JSON)", key=f"export_json_{system}"):
                    journey_dict = journey.to_dict()
                    json_str = json.dumps(journey_dict, indent=2)

                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"column_journey_{system}_{journey.column_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key=f"download_json_{system}"
                    )

            with col2:
                if st.button(f"📊 Download {system.upper()} Journey (Excel)", key=f"export_excel_{system}"):
                    # Convert steps to DataFrame
                    steps_data = [
                        {
                            "Step": step.step_number,
                            "Source Table": step.source_table,
                            "Source Column": step.source_column,
                            "Script": step.script_name,
                            "Transformation": step.transformation,
                            "Type": step.transformation_type,
                            "Target Table": step.target_table,
                            "Target Column": step.target_column
                        }
                        for step in journey.steps
                    ]

                    df = pd.DataFrame(steps_data)
                    output = io.BytesIO()

                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name=f'{journey.column_name}_Journey', index=False)

                    output.seek(0)

                    st.download_button(
                        label="Download Excel",
                        data=output,
                        file_name=f"column_journey_{system}_{journey.column_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_excel_{system}"
                    )

    # Cross-system comparison
    if len(journeys) > 1:
        st.markdown("---")
        st.markdown("### 🔄 Cross-System Comparison")

        systems = list(journeys.keys())
        source_system = systems[0]
        target_system = systems[1] if len(systems) > 1 else systems[0]

        source_journey = journeys[source_system]
        target_journey = journeys[target_system]

        comparison_col1, comparison_col2 = st.columns(2)

        with comparison_col1:
            st.markdown(f"**{source_system.upper()}**")
            st.metric("Steps", source_journey.total_steps)
            st.metric("Confidence", f"{source_journey.confidence:.0%}")

        with comparison_col2:
            st.markdown(f"**{target_system.upper()}**")
            st.metric("Steps", target_journey.total_steps)
            st.metric("Confidence", f"{target_journey.confidence:.0%}")

        # Observations
        st.markdown("**Observations:**")

        step_diff = abs(source_journey.total_steps - target_journey.total_steps)
        if step_diff == 0:
            st.success(f"✅ Both systems have same number of transformation steps ({source_journey.total_steps})")
        elif step_diff <= 2:
            st.warning(f"⚠️ Similar number of steps (difference: {step_diff})")
        else:
            st.error(f"❌ Significant difference in transformation steps (difference: {step_diff})")

        # Compare transformations
        source_trans = [step.transformation_type for step in source_journey.steps]
        target_trans = [step.transformation_type for step in target_journey.steps]

        common_trans = set(source_trans) & set(target_trans)
        if common_trans:
            st.info(f"🔹 Common transformation types: {', '.join(common_trans)}")

        only_source = set(source_trans) - set(target_trans)
        if only_source:
            st.warning(f"⚠️ Only in {source_system}: {', '.join(only_source)}")

        only_target = set(target_trans) - set(source_trans)
        if only_target:
            st.warning(f"⚠️ Only in {target_system}: {', '.join(only_target)}")


def render_enhanced_lineage_tab():
    """
    Render Enhanced Lineage tab with dependency graph visualization
    
    Features:
    - Proper script parsing (SQL, Pig, Shell, Python)
    - Classifies scripts by operation type
    - Follows actual data locations
    - Interactive dependency graph
    """
    st.markdown("### 🌐 Enhanced Lineage Tracking with Dependency Graph")
    
    st.info("""
    **Enhanced Lineage Features:**
    - ✅ Parses actual script logic (not just keywords)
    - ✅ Classifies scripts: Source, Transform, Consumer, Definition
    - ✅ Follows HDFS paths and table locations
    - ✅ Builds accurate dependency graphs
    - ✅ Interactive graph visualization
    """)
    
    # Initialize enhanced tracker if not already done
    if 'enhanced_lineage_tracker' not in st.session_state:
        with st.spinner("Initializing Enhanced Lineage Tracker..."):
            try:
                indexer = st.session_state.get('indexer')
                ai_analyzer = st.session_state.get('ai_analyzer')
                
                st.session_state.enhanced_lineage_tracker = EnhancedLineageTracker(
                    indexer=indexer,
                    ai_analyzer=ai_analyzer
                )
                logger.info("✓ Enhanced Lineage Tracker initialized")
            except Exception as e:
                st.error(f"Error initializing enhanced tracker: {e}")
                return
    
    st.markdown("---")
    st.markdown("### 📋 Step 1: Select Table/Entity to Track")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_system = st.selectbox(
            "System",
            options=["Hadoop", "Databricks", "Ab Initio"],
            key="enhanced_lineage_system"
        )
        
        table_name = st.text_input(
            "Table/Entity Name",
            placeholder="e.g., ack, customer_data, bdf_download",
            help="Enter table name or entity name to track",
            key="enhanced_lineage_table"
        )
    
    with col2:
        max_depth = st.slider(
            "Maximum Depth",
            min_value=1,
            max_value=20,
            value=10,
            help="How deep to traverse the dependency graph",
            key="enhanced_lineage_depth"
        )
        
        layout_type = st.select_slider(
            "Graph Layout",
            options=["hierarchical", "spring", "circular"],
            value="hierarchical",
            help="Layout algorithm for graph visualization",
            key="enhanced_lineage_layout"
        )
    
    # Track button
    if st.button("🔍 Track Lineage", type="primary", use_container_width=True):
        if not table_name:
            st.error("Please enter a table/entity name")
            return
        
        # Map display names to internal names
        system_map = {
            "Hadoop": "hadoop",
            "Databricks": "databricks",
            "Ab Initio": "abinitio"
        }
        
        source_sys = system_map[source_system]
        
        # Track lineage
        with st.spinner(f"🔍 Analyzing {table_name} in {source_system}..."):
            tracker = st.session_state.enhanced_lineage_tracker
            
            try:
                graph = tracker.track_table_lineage(
                    table_name=table_name,
                    system=source_sys,
                    max_depth=max_depth
                )
                
                # Store results
                st.session_state.lineage_graph = graph
                st.session_state.lineage_table = table_name
                st.session_state.lineage_system = source_system
                
                st.success(f"✓ Lineage tracked! Found {len(graph.nodes)} nodes and {len(graph.edges)} connections")
                
            except Exception as e:
                st.error(f"Error tracking lineage: {e}")
                logger.error(f"Enhanced lineage error: {e}", exc_info=True)
                return
    
    # Display results
    if 'lineage_graph' in st.session_state and st.session_state.lineage_graph:
        render_enhanced_lineage_results(
            st.session_state.lineage_graph,
            st.session_state.lineage_table,
            st.session_state.lineage_system,
            layout_type
        )


def render_enhanced_lineage_results(graph, table_name, system, layout_type):
    """Display enhanced lineage results with graph visualization"""
    st.markdown("---")
    st.markdown("### 📊 Step 2: Lineage Dependency Graph")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Nodes", len(graph.nodes))
    with col2:
        st.metric("Total Edges", len(graph.edges))
    with col3:
        st.metric("Entry Points (Sources)", len(graph.entry_points))
    with col4:
        st.metric("Exit Points (Consumers)", len(graph.exit_points))
    
    # Graph visualization
    st.markdown("---")
    st.markdown("#### 🌐 Interactive Dependency Graph")
    
    try:
        # Create interactive graph
        graph_dict = graph.to_dict()
        visualizer = LineageVisualizer()
        
        fig = visualizer.create_interactive_graph(
            lineage_graph=graph_dict,
            layout=layout_type
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating graph visualization: {e}")
        logger.error(f"Graph visualization error: {e}", exc_info=True)
    
    # Node details
    st.markdown("---")
    st.markdown("#### 📋 Node Details")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        node_type_filter = st.multiselect(
            "Filter by Node Type",
            options=["data_location", "script"],
            default=["data_location", "script"],
            key="node_type_filter"
        )
    
    with col2:
        script_type_filter = st.multiselect(
            "Filter by Script Type",
            options=["source", "transform", "consumer", "definition", "orchestrator", "unknown"],
            default=["source", "transform", "consumer"],
            key="script_type_filter"
        )
    
    # Display nodes in tabs
    tab1, tab2, tab3 = st.tabs(["📍 Data Locations", "📜 Scripts", "🔗 Connections"])
    
    with tab1:
        st.markdown("**Data Locations (Tables, HDFS Paths, Files)**")
        
        location_data = []
        for node_id, node in graph.nodes.items():
            if node.node_type == "data_location":
                details = node.details
                location_data.append({
                    "Name": node.name,
                    "Type": details.get('location_type', 'unknown'),
                    "External": "Yes" if details.get('is_external') else "No",
                    "Node ID": node_id
                })
        
        if location_data:
            df = pd.DataFrame(location_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No data locations found")
    
    with tab2:
        st.markdown("**Scripts (Source, Transform, Consumer, Definition)**")
        
        script_data = []
        for node_id, node in graph.nodes.items():
            if node.node_type == "script":
                details = node.details
                script_type = details.get('type', 'unknown')
                
                if script_type in script_type_filter:
                    # Color code by type
                    if script_type == 'source':
                        type_badge = "🟢 Source"
                    elif script_type == 'transform':
                        type_badge = "🟠 Transform"
                    elif script_type == 'consumer':
                        type_badge = "🔴 Consumer"
                    elif script_type == 'definition':
                        type_badge = "🟣 Definition"
                    else:
                        type_badge = "⚪ Unknown"
                    
                    script_data.append({
                        "Script": node.name,
                        "Type": type_badge,
                        "Transformations": ", ".join(details.get('transformations', [])),
                        "Confidence": f"{details.get('confidence', 0):.0%}",
                        "Path": details.get('path', 'N/A')
                    })
        
        if script_data:
            df = pd.DataFrame(script_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Show detailed script analysis
            st.markdown("---")
            selected_script = st.selectbox(
                "View Detailed Analysis",
                options=[s["Script"] for s in script_data],
                key="selected_script_detail"
            )
            
            if selected_script:
                # Find the node
                for node_id, node in graph.nodes.items():
                    if node.node_type == "script" and node.name == selected_script:
                        st.markdown(f"#### 📄 {selected_script}")
                        
                        details = node.details
                        
                        info_col1, info_col2 = st.columns(2)
                        with info_col1:
                            st.markdown(f"**Type:** {details.get('type', 'unknown')}")
                            st.markdown(f"**Confidence:** {details.get('confidence', 0):.0%}")
                        with info_col2:
                            st.markdown(f"**Path:** `{details.get('path', 'N/A')}`")
                        
                        st.markdown("**Transformations:**")
                        trans = details.get('transformations', [])
                        if trans:
                            for t in trans:
                                st.markdown(f"- {t}")
                        else:
                            st.markdown("_No transformations detected_")
                        
                        # Show upstream and downstream
                        st.markdown("**Dependencies:**")
                        dep_col1, dep_col2 = st.columns(2)
                        
                        with dep_col1:
                            st.markdown("**⬆️ Upstream (Reads From):**")
                            upstream = graph.get_upstream(node_id, max_depth=1)
                            if upstream:
                                for up_id in upstream[:10]:
                                    if up_id in graph.nodes:
                                        st.markdown(f"- {graph.nodes[up_id].name}")
                            else:
                                st.markdown("_Source (no dependencies)_")
                        
                        with dep_col2:
                            st.markdown("**⬇️ Downstream (Writes To):**")
                            downstream = graph.get_downstream(node_id, max_depth=1)
                            if downstream:
                                for down_id in downstream[:10]:
                                    if down_id in graph.nodes:
                                        st.markdown(f"- {graph.nodes[down_id].name}")
                            else:
                                st.markdown("_Consumer (no outputs)_")
                        
                        break
        else:
            st.info("No scripts found matching filter")
    
    with tab3:
        st.markdown("**Data Flow Connections**")
        
        connection_data = []
        for edge in graph.edges:
            source_node = graph.nodes.get(edge.source_id)
            target_node = graph.nodes.get(edge.target_id)
            
            if source_node and target_node:
                connection_data.append({
                    "From": source_node.name,
                    "To": target_node.name,
                    "Type": edge.edge_type,
                    "Via Script": edge.script or "N/A"
                })
        
        if connection_data:
            df = pd.DataFrame(connection_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No connections found")
    
    # Export options
    st.markdown("---")
    st.markdown("### 📤 Export Lineage Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Graph (JSON)", use_container_width=True):
            graph_dict = graph.to_dict()
            json_str = json.dumps(graph_dict, indent=2)
            
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"lineage_graph_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Download as Excel", use_container_width=True):
            # Create Excel with multiple sheets
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Nodes sheet
                if location_data:
                    pd.DataFrame(location_data).to_excel(writer, sheet_name='Data_Locations', index=False)
                if script_data:
                    pd.DataFrame(script_data).to_excel(writer, sheet_name='Scripts', index=False)
                if connection_data:
                    pd.DataFrame(connection_data).to_excel(writer, sheet_name='Connections', index=False)
            
            output.seek(0)
            
            st.download_button(
                label="Download Excel",
                data=output,
                file_name=f"lineage_graph_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col3:
        if st.button("📈 Download as Mermaid", use_container_width=True):
            visualizer = LineageVisualizer()
            mermaid_diagram = visualizer.generate_mermaid_diagram(graph.to_dict())
            
            st.download_button(
                label="Download Mermaid",
                data=mermaid_diagram,
                file_name=f"lineage_mermaid_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
