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
from services.multi_collection_indexer import MultiCollectionIndexer
from services.ai_script_analyzer import AIScriptAnalyzer

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
