"""
STAG - Smart Transform and Analysis Generator
==============================================
Comprehensive Streamlit frontend for CodebaseIntelligence RAG System

Features:
- Multi-system codebase chat with conversation memory
- Intelligent query routing and structured responses
- On-the-fly file indexing (upload Ab Initio, Autosys, PDFs)
- Cross-system comparison with Excel/PDF export
- Configurable AI parameters (temperature, top-k, top-p)
- Confidence scoring and source attribution
- Fuzzy matching and typo handling
- Adaptive learning within session
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import io

# Core services
from services.multi_collection_indexer import MultiCollectionIndexer
from services.query_router import QueryRouter
from services.response_formatter import ResponseFormatter
from services.logic_comparator import LogicComparator
from services.azure_embeddings import create_embedding_client

# Parsers
from parsers.abinitio.parser import AbInitioParser
from parsers.hadoop.parser import HadoopParser
from parsers.databricks.parser import DatabricksParser
from parsers.autosys.parser import AutosysParser
from parsers.documents.document_parser import DocumentParser

# RAG chatbot
from services.rag_chatbot_integrated import CodebaseRAGChatbot

# AI Analyzer
from services.ai_script_analyzer import AIScriptAnalyzer

# UI Components
from ui.lineage_tab import render_lineage_tab

# Chat orchestration
from services.chat.chat_orchestrator import create_chat_orchestrator, UpdateType

# Utilities
from loguru import logger
import tempfile
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()  # Load .env file to make Azure OpenAI credentials available


# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="STAG - Smart Transform Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# Session State Initialization
# ============================================

def initialize_session_state():
    """Initialize all session state variables"""

    # Conversation history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # RAG components
    if 'indexer' not in st.session_state:
        st.session_state.indexer = None

    if 'router' not in st.session_state:
        st.session_state.router = None

    if 'formatter' not in st.session_state:
        st.session_state.formatter = None

    if 'comparator' not in st.session_state:
        st.session_state.comparator = None

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None

    if 'ai_analyzer' not in st.session_state:
        st.session_state.ai_analyzer = None

    if 'chat_orchestrator' not in st.session_state:
        st.session_state.chat_orchestrator = None

    # Settings
    if 'model_name' not in st.session_state:
        st.session_state.model_name = "gpt-4"

    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.3

    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5

    if 'top_p' not in st.session_state:
        st.session_state.top_p = 0.9

    # Query history
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    # Comparison mode
    if 'comparison_mode' not in st.session_state:
        st.session_state.comparison_mode = False

    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = None

    # Indexed files tracking
    if 'indexed_files' not in st.session_state:
        st.session_state.indexed_files = {
            'abinitio': [],
            'hadoop': [],
            'databricks': [],
            'autosys': [],
            'documents': []
        }

    # Statistics
    if 'stats' not in st.session_state:
        st.session_state.stats = {}

    # Adaptive learning cache
    if 'learned_queries' not in st.session_state:
        st.session_state.learned_queries = {}


def initialize_rag_components():
    """Initialize RAG components if not already initialized"""

    if st.session_state.indexer is None:
        with st.spinner("Initializing RAG system..."):
            try:
                # Multi-collection indexer
                st.session_state.indexer = MultiCollectionIndexer(
                    vector_db_path="./outputs/vector_db"
                )

                # Query router
                st.session_state.router = QueryRouter()

                # Response formatter
                st.session_state.formatter = ResponseFormatter()

                # Logic comparator
                st.session_state.comparator = LogicComparator()

                # AI Analyzer (for intelligent analysis)
                st.session_state.ai_analyzer = AIScriptAnalyzer()

                # Chatbot (optional, for conversational mode)
                st.session_state.chatbot = CodebaseRAGChatbot(
                    use_local_search=True,
                    vector_db_path="./outputs/vector_db"
                )

                # Chat orchestrator (for agent-based streaming)
                st.session_state.chat_orchestrator = create_chat_orchestrator(
                    ai_analyzer=st.session_state.ai_analyzer,
                    indexer=st.session_state.indexer,
                    vector_store=None  # Not using vector_store directly
                )

                # Get initial stats
                st.session_state.stats = st.session_state.indexer.get_stats()

                logger.info("✓ STAG RAG components initialized")

            except Exception as e:
                st.error(f"Error initializing RAG components: {e}")
                logger.error(f"RAG initialization error: {e}")


# ============================================
# Sidebar - Configuration & Controls
# ============================================

def render_sidebar():
    """Render sidebar with configuration and controls"""

    st.sidebar.title("🚀 STAG Configuration")

    # Model selection
    st.sidebar.subheader("🤖 AI Model")
    model_options = ["gpt-4", "gpt-4o", "gpt-35-turbo"]
    st.session_state.model_name = st.sidebar.selectbox(
        "Model",
        options=model_options,
        index=model_options.index(st.session_state.model_name),
        help="Select Azure OpenAI model"
    )

    # AI Parameters
    st.sidebar.subheader("⚙️ AI Parameters")

    st.session_state.temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Higher = more creative, Lower = more deterministic"
    )

    st.session_state.top_k = st.sidebar.slider(
        "Top K Results",
        min_value=1,
        max_value=20,
        value=st.session_state.top_k,
        step=1,
        help="Number of search results to retrieve"
    )

    st.session_state.top_p = st.sidebar.slider(
        "Top P (Nucleus)",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.top_p,
        step=0.05,
        help="Nucleus sampling threshold"
    )

    st.sidebar.divider()

    # File upload section
    st.sidebar.subheader("📁 Index Files")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Ab Initio, Autosys, or Documents",
        accept_multiple_files=True,
        type=['mp', 'jil', 'pdf', 'xlsx', 'docx', 'txt', 'md'],
        help="Upload files for on-the-fly indexing"
    )

    if uploaded_files:
        if st.sidebar.button("Index Uploaded Files"):
            index_uploaded_files(uploaded_files)

    st.sidebar.divider()

    # System filters
    st.sidebar.subheader("🔍 System Filters")

    all_systems = st.sidebar.checkbox("Search All Systems", value=True)

    if not all_systems:
        st.sidebar.multiselect(
            "Select Systems",
            options=["Ab Initio", "Hadoop", "Databricks", "Autosys", "Documents"],
            default=["Ab Initio", "Hadoop"]
        )

    st.sidebar.divider()

    # Statistics
    st.sidebar.subheader("📊 Database Stats")

    if st.session_state.stats:
        for collection, stats in st.session_state.stats.items():
            doc_count = stats.get('total_documents', 0)
            st.sidebar.metric(
                collection.replace('_collection', '').title(),
                f"{doc_count:,} docs"
            )

    st.sidebar.divider()

    # Actions
    st.sidebar.subheader("🛠️ Actions")

    if st.sidebar.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    if st.sidebar.button("Refresh Stats"):
        if st.session_state.indexer:
            st.session_state.stats = st.session_state.indexer.get_stats()
            st.rerun()

    if st.sidebar.button("Export Chat History"):
        export_chat_history()


# ============================================
# File Indexing
# ============================================

def index_uploaded_files(uploaded_files):
    """Index uploaded files on-the-fly"""

    if not st.session_state.indexer:
        st.error("Indexer not initialized")
        return

    with st.spinner(f"Indexing {len(uploaded_files)} files..."):
        abinitio_files = []
        autosys_files = []
        document_files = []

        # Save files to temp directory
        temp_dir = Path(tempfile.mkdtemp())

        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name

            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Categorize by extension
            if uploaded_file.name.endswith('.mp'):
                abinitio_files.append(file_path)
            elif uploaded_file.name.endswith('.jil'):
                autosys_files.append(file_path)
            else:
                document_files.append(file_path)

        # Index Ab Initio
        if abinitio_files:
            try:
                parser = AbInitioParser()
                for file_path in abinitio_files:
                    result = parser.parse_file(str(file_path))
                    if result.get('processes'):
                        st.session_state.indexer.index_abinitio(
                            processes=result['processes'],
                            components=result.get('components', [])
                        )
                        st.session_state.indexed_files['abinitio'].append(file_path.name)

                st.success(f"✓ Indexed {len(abinitio_files)} Ab Initio files")
            except Exception as e:
                st.error(f"Error indexing Ab Initio: {e}")

        # Index Autosys
        if autosys_files:
            try:
                parser = AutosysParser()
                for file_path in autosys_files:
                    result = parser.parse_file(str(file_path))
                    if result.get('components'):
                        jobs_dict = [job.__dict__ for job in result['components']]
                        st.session_state.indexer.index_autosys(jobs=jobs_dict)
                        st.session_state.indexed_files['autosys'].append(file_path.name)

                st.success(f"✓ Indexed {len(autosys_files)} Autosys files")
            except Exception as e:
                st.error(f"Error indexing Autosys: {e}")

        # Index documents
        if document_files:
            try:
                parser = DocumentParser()
                docs = []
                for file_path in document_files:
                    doc = parser.parse_file(str(file_path))
                    if doc:
                        docs.append(doc)

                if docs:
                    st.session_state.indexer.index_documents(docs)
                    st.session_state.indexed_files['documents'].extend(
                        [f.name for f in document_files]
                    )

                st.success(f"✓ Indexed {len(document_files)} documents")
            except Exception as e:
                st.error(f"Error indexing documents: {e}")

        # Refresh stats
        st.session_state.stats = st.session_state.indexer.get_stats()


# ============================================
# Chat Interface
# ============================================

def render_chat_interface():
    """Render main chat interface with agent-based streaming"""

    st.title("💬 STAG - Smart Transform Analysis")
    st.caption("🤖 Powered by AI agents with visible thinking process")

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display metadata if available
            if "metadata" in message and message["metadata"]:
                with st.expander("📊 Query Details"):
                    # Show thinking process if available
                    if message["metadata"].get("thinking"):
                        st.markdown("**🧠 Thinking Process:**")
                        for thought in message["metadata"]["thinking"]:
                            st.caption(f"• {thought}")

                    # Show task plan if available
                    if message["metadata"].get("task_plan"):
                        st.markdown("**📋 Task Plan:**")
                        for i, task in enumerate(message["metadata"]["task_plan"], 1):
                            st.caption(f"{i}. {task}")

                    # Show agent execution if available
                    if message["metadata"].get("agent_execution"):
                        st.markdown("**🔧 Agent Execution:**")
                        for agent_log in message["metadata"]["agent_execution"]:
                            st.caption(f"• {agent_log}")

                    # Show data
                    if message["metadata"].get("data"):
                        st.json(message["metadata"]["data"])

    # Chat input
    if prompt := st.chat_input("Ask about your codebase..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response with streaming
        with st.chat_message("assistant"):
            render_streaming_response(prompt)


def render_streaming_response(query: str):
    """Render response with streaming agent updates"""

    # Check if orchestrator is available
    if not st.session_state.chat_orchestrator:
        st.error("Chat orchestrator not initialized. Please refresh the page.")
        return

    # Create containers for streaming updates
    thinking_container = st.container()
    task_plan_container = st.container()
    agent_execution_container = st.container()
    answer_container = st.container()

    # Track metadata for history
    thinking_logs = []
    task_plan = []
    agent_logs = []
    final_data = {}
    final_answer = ""

    try:
        # Stream updates from orchestrator
        for update in st.session_state.chat_orchestrator.process_query_stream(
            query=query,
            context=None,
            conversation_history=st.session_state.messages
        ):
            if update.type == UpdateType.THINKING:
                # Show thinking process
                with thinking_container:
                    st.caption(f"💭 {update.content}")
                thinking_logs.append(update.content)

            elif update.type == UpdateType.TASK_PLAN:
                # Show task plan
                with task_plan_container:
                    st.markdown("### 📋 Task Plan:")
                    if update.data and update.data.get("tasks"):
                        for i, task in enumerate(update.data["tasks"], 1):
                            st.caption(f"{i}. {task}")
                        task_plan = update.data["tasks"]

            elif update.type == UpdateType.TASK_START:
                # Show task start
                with agent_execution_container:
                    st.info(f"⚙️ {update.content}")
                agent_logs.append(f"[START] {update.content}")

            elif update.type == UpdateType.TASK_PROGRESS:
                # Show task progress
                with agent_execution_container:
                    st.caption(f"  ↳ {update.content}")
                agent_logs.append(f"[PROGRESS] {update.content}")

            elif update.type == UpdateType.TASK_COMPLETE:
                # Show task completion
                with agent_execution_container:
                    st.success(f"✅ {update.content}")
                agent_logs.append(f"[COMPLETE] {update.content}")

            elif update.type == UpdateType.AGENT_START:
                # Show agent start
                with agent_execution_container:
                    st.info(f"🤖 {update.content}")
                agent_logs.append(f"[AGENT START] {update.content}")

            elif update.type == UpdateType.AGENT_PROGRESS:
                # Show agent progress
                with agent_execution_container:
                    st.caption(f"  → {update.content}")
                agent_logs.append(f"[AGENT PROGRESS] {update.content}")

            elif update.type == UpdateType.AGENT_COMPLETE:
                # Show agent completion
                with agent_execution_container:
                    st.success(f"✅ {update.content}")
                agent_logs.append(f"[AGENT COMPLETE] {update.content}")

            elif update.type == UpdateType.FINAL_ANSWER:
                # Show final answer
                final_answer = update.content
                if update.data:
                    final_data = update.data

                with answer_container:
                    st.markdown("---")
                    st.markdown("### 💡 Answer:")
                    st.markdown(final_answer)

                    # Show additional data if available
                    if final_data.get("sources"):
                        with st.expander(f"📚 Sources ({len(final_data['sources'])} found)"):
                            for i, source in enumerate(final_data["sources"], 1):
                                if isinstance(source, dict):
                                    st.caption(f"{i}. {source.get('source', 'Unknown source')}")
                                else:
                                    st.caption(f"{i}. {str(source)[:100]}...")

            elif update.type == UpdateType.ERROR:
                # Show error
                with agent_execution_container:
                    st.error(f"❌ {update.content}")
                agent_logs.append(f"[ERROR] {update.content}")

        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer if final_answer else "No response generated",
            "metadata": {
                "thinking": thinking_logs,
                "task_plan": task_plan,
                "agent_execution": agent_logs,
                "data": final_data,
                "timestamp": datetime.now().isoformat()
            }
        })

        # Add to query history
        st.session_state.query_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in streaming response: {e}")
        st.error(f"Error generating response: {e}")


def generate_response(query: str) -> Dict[str, Any]:
    """Generate response using RAG system"""

    try:
        # Check adaptive learning cache
        if query in st.session_state.learned_queries:
            logger.info(f"Using cached response for: {query[:50]}...")
            return st.session_state.learned_queries[query]

        # Fuzzy matching for typos
        corrected_query = apply_fuzzy_matching(query)
        if corrected_query != query:
            st.info(f"Did you mean: *{corrected_query}*?")
            query = corrected_query

        # Route query
        routing = st.session_state.router.route_query(query)

        # Search collections
        results = st.session_state.indexer.search_multi_collection(
            query=query,
            collections=routing["collections"],
            top_k=st.session_state.top_k
        )

        # Format response
        formatted_response = st.session_state.formatter.format_response(
            results_by_collection=results,
            query=query,
            intent=routing.get("intent", "search")
        )

        # Generate answer using chatbot (if available)
        if st.session_state.chatbot:
            # Prepare context from results
            context_sources = []
            for coll, docs in results.items():
                context_sources.extend(docs)

            # Generate answer
            answer = st.session_state.chatbot.generate_answer(
                query=query,
                sources=context_sources,
                temperature=st.session_state.temperature
            )
        else:
            # Fallback: use formatted response
            answer = formatted_response["formatted_text"]

        response_data = {
            "answer": answer,
            "sources": context_sources if st.session_state.chatbot else [],
            "routing": routing,
            "formatted_response": formatted_response
        }

        # Cache in adaptive learning
        st.session_state.learned_queries[query] = response_data

        return response_data

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            "answer": f"Error generating response: {e}",
            "sources": [],
            "routing": {},
            "formatted_response": {}
        }


def render_sources(sources: List[Dict[str, Any]]):
    """Render source documents with confidence scores"""

    for i, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})

        # Calculate confidence score (based on relevance score)
        confidence = source.get("score", 0.5) * 100

        # Determine color
        if confidence >= 80:
            color = "green"
        elif confidence >= 60:
            color = "orange"
        else:
            color = "red"

        st.markdown(f"**{i}. {source.get('title', 'Untitled')}** "
                   f":{color}[{confidence:.0f}% confident]")

        st.markdown(f"*{source.get('content', '')[:200]}...*")

        # Metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"System: {metadata.get('system', 'N/A')}")
        with col2:
            st.caption(f"Type: {metadata.get('doc_type', 'N/A')}")
        with col3:
            st.caption(f"Collection: {source.get('collection', 'N/A')}")

        st.divider()


# ============================================
# Comparison Mode
# ============================================

def render_comparison_mode():
    """Render comparison mode interface"""

    st.subheader("🔄 Cross-System Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("System 1 (e.g., Ab Initio graph)", key="system1_input")

    with col2:
        st.text_input("System 2 (e.g., Hadoop script)", key="system2_input")

    if st.button("Compare Logic"):
        if st.session_state.system1_input and st.session_state.system2_input:
            run_comparison(
                st.session_state.system1_input,
                st.session_state.system2_input
            )

    # Display comparison results
    if st.session_state.comparison_results:
        render_comparison_results(st.session_state.comparison_results)


def run_comparison(system1_name: str, system2_name: str):
    """Run cross-system logic comparison"""

    with st.spinner("Comparing systems..."):
        try:
            # Search for system1
            results1 = st.session_state.indexer.search_multi_collection(
                query=system1_name,
                collections=["abinitio_collection", "hadoop_collection"],
                top_k=1
            )

            # Search for system2
            results2 = st.session_state.indexer.search_multi_collection(
                query=system2_name,
                collections=["abinitio_collection", "hadoop_collection"],
                top_k=1
            )

            # Extract top results
            system1_data = None
            system2_data = None

            for coll, docs in results1.items():
                if docs:
                    system1_data = {
                        "system_name": coll.replace('_collection', '').title(),
                        "name": docs[0].get('title'),
                        "code": docs[0].get('content'),
                        "description": docs[0].get('metadata', {})
                    }
                    break

            for coll, docs in results2.items():
                if docs:
                    system2_data = {
                        "system_name": coll.replace('_collection', '').title(),
                        "name": docs[0].get('title'),
                        "code": docs[0].get('content'),
                        "description": docs[0].get('metadata', {})
                    }
                    break

            if system1_data and system2_data:
                # Compare using LogicComparator
                comparison = st.session_state.comparator.compare_logic(
                    system1=system1_data,
                    system2=system2_data
                )

                st.session_state.comparison_results = comparison
            else:
                st.error("Could not find both systems for comparison")

        except Exception as e:
            st.error(f"Comparison error: {e}")


def render_comparison_results(comparison: Dict[str, Any]):
    """Render comparison results in structured format"""

    st.success("✓ Comparison Complete")

    # Summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Similarity Score", f"{comparison.get('similarity_score', 0):.2%}")
    with col2:
        equiv = "✓ Yes" if comparison.get('are_equivalent') else "✗ No"
        st.metric("Equivalent?", equiv)

    # Semantic summary
    st.markdown("### 📝 Summary")
    st.info(comparison.get('semantic_summary', 'N/A'))

    # Differences table
    if comparison.get('differences'):
        st.markdown("### 🔍 Key Differences")

        diff_df = pd.DataFrame(comparison['differences'])
        st.dataframe(diff_df, use_container_width=True)

    # Migration recommendation
    if comparison.get('migration_recommendation'):
        st.markdown("### 🚀 Migration Recommendation")

        migration = comparison['migration_recommendation']

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Difficulty", migration.get('difficulty', 'N/A').upper())
        with col2:
            st.metric("Effort Estimate", migration.get('effort_estimate', 'N/A'))

        st.markdown("**Approach:**")
        st.write(migration.get('approach', 'N/A'))

    # Export options
    st.markdown("### 📤 Export")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export to Excel"):
            export_comparison_to_excel(comparison)

    with col2:
        if st.button("Export to PDF"):
            export_comparison_to_pdf(comparison)


# ============================================
# Export Functions
# ============================================

def export_comparison_to_excel(comparison: Dict[str, Any]):
    """Export comparison to Excel"""

    try:
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([{
                'Similarity Score': comparison.get('similarity_score', 0),
                'Are Equivalent': comparison.get('are_equivalent', False),
                'Summary': comparison.get('semantic_summary', '')
            }])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Differences sheet
            if comparison.get('differences'):
                diff_df = pd.DataFrame(comparison['differences'])
                diff_df.to_excel(writer, sheet_name='Differences', index=False)

            # Migration sheet
            if comparison.get('migration_recommendation'):
                migration_df = pd.DataFrame([comparison['migration_recommendation']])
                migration_df.to_excel(writer, sheet_name='Migration', index=False)

        output.seek(0)

        st.download_button(
            label="Download Excel",
            data=output,
            file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Export error: {e}")


def export_comparison_to_pdf(comparison: Dict[str, Any]):
    """Export comparison to PDF (placeholder)"""
    st.info("PDF export functionality coming soon!")


def export_chat_history():
    """Export chat history to JSON"""

    try:
        history_json = json.dumps(st.session_state.messages, indent=2)

        st.sidebar.download_button(
            label="Download History",
            data=history_json,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    except Exception as e:
        st.sidebar.error(f"Export error: {e}")


# ============================================
# Fuzzy Matching & Typo Handling
# ============================================

def apply_fuzzy_matching(query: str) -> str:
    """Apply fuzzy matching to correct common typos"""

    # Common corrections
    corrections = {
        'abinito': 'abinitio',
        'ab initio': 'abinitio',
        'hadop': 'hadoop',
        'sparc': 'spark',
        'autosys': 'autosys',
        'databrick': 'databricks',
    }

    query_lower = query.lower()

    for typo, correction in corrections.items():
        if typo in query_lower:
            query = query.replace(typo, correction)
            query = query.replace(typo.title(), correction.title())

    return query


# ============================================
# Database Management Interface
# ============================================

def render_database_management():
    """Render database management interface"""

    st.subheader("⚙️ Vector Database Management")

    st.markdown("""
    Manage your vector database collections, check status, clear data, and re-index your codebase.
    """)

    # Database status section
    st.markdown("---")
    st.markdown("### 📊 Database Status")

    col1, col2, col3 = st.columns(3)

    # Calculate total stats
    total_docs = 0
    total_collections = 0
    db_path = Path("./outputs/vector_db")
    db_size_mb = 0

    if db_path.exists():
        # Calculate size
        total_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
        db_size_mb = total_size / (1024 * 1024)

        # Count documents
        if st.session_state.stats:
            for stats in st.session_state.stats.values():
                total_docs += stats.get('total_documents', 0)
                total_collections += 1

    with col1:
        st.metric("Total Documents", f"{total_docs:,}")
    with col2:
        st.metric("Collections", total_collections)
    with col3:
        st.metric("Database Size", f"{db_size_mb:.2f} MB")

    # Refresh button
    if st.button("🔄 Refresh Status", use_container_width=True):
        if st.session_state.indexer:
            st.session_state.stats = st.session_state.indexer.get_stats()
            st.success("✓ Status refreshed!")
            st.rerun()

    # Collection details
    st.markdown("---")
    st.markdown("### 📈 Collection Details")

    if st.session_state.stats:
        for collection_name, collection_stats in st.session_state.stats.items():
            with st.expander(f"📁 {collection_name.replace('_collection', '').title()}", expanded=False):
                doc_count = collection_stats.get('total_documents', 0)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Documents", f"{doc_count:,}")

                # Show any errors
                if 'error' in collection_stats:
                    st.error(f"⚠️ Error: {collection_stats['error']}")
    else:
        st.info("No database statistics available. Initialize the indexer first.")

    # Management operations
    st.markdown("---")
    st.markdown("### 🛠️ Management Operations")

    operation = st.selectbox(
        "Select Operation",
        [
            "View Status Only",
            "Clear Specific Collection",
            "Clear All Collections",
            "Re-index Ab Initio",
            "Re-index Hadoop",
            "Re-index Databricks",
            "Re-index Autosys",
            "Re-index Documents",
            "Re-index Everything",
            "Export Statistics"
        ]
    )

    # Operation-specific UI
    if operation == "Clear Specific Collection":
        render_clear_collection_ui()

    elif operation == "Clear All Collections":
        render_clear_all_ui()

    elif operation == "Re-index Ab Initio":
        render_reindex_abinitio_ui()

    elif operation == "Re-index Hadoop":
        render_reindex_hadoop_ui()

    elif operation == "Re-index Databricks":
        render_reindex_databricks_ui()

    elif operation == "Re-index Autosys":
        render_reindex_autosys_ui()

    elif operation == "Re-index Documents":
        render_reindex_documents_ui()

    elif operation == "Re-index Everything":
        render_reindex_all_ui()

    elif operation == "Export Statistics":
        render_export_stats_ui()


def render_clear_collection_ui():
    """UI for clearing specific collection"""
    st.markdown("#### 🗑️ Clear Specific Collection")

    collection_map = {
        'Ab Initio': 'abinitio_collection',
        'Hadoop': 'hadoop_collection',
        'Databricks': 'databricks_collection',
        'Autosys': 'autosys_collection',
        'Cross-System Links': 'cross_system_links',
        'Documents': 'documents_collection',
    }

    selected_display = st.selectbox("Select Collection to Clear", list(collection_map.keys()))
    selected_collection = collection_map[selected_display]

    # Show current count
    if st.session_state.stats and selected_collection in st.session_state.stats:
        current_count = st.session_state.stats[selected_collection].get('total_documents', 0)
        st.info(f"Current documents in {selected_display}: **{current_count:,}**")

    st.warning("⚠️ This will permanently delete all documents in this collection!")

    confirm = st.checkbox("I understand this action cannot be undone")

    if st.button("Clear Collection", type="primary", disabled=not confirm):
        with st.spinner(f"Clearing {selected_display}..."):
            try:
                # Delete collection directory
                collection_path = Path("./outputs/vector_db") / selected_collection
                if collection_path.exists():
                    import shutil
                    shutil.rmtree(collection_path)
                    st.success(f"✓ {selected_display} collection cleared successfully!")

                    # Refresh stats
                    if st.session_state.indexer:
                        st.session_state.stats = st.session_state.indexer.get_stats()

                    st.rerun()
                else:
                    st.warning(f"Collection {selected_display} doesn't exist")
            except Exception as e:
                st.error(f"Error clearing collection: {e}")


def render_clear_all_ui():
    """UI for clearing all collections"""
    st.markdown("#### 🗑️ Clear All Collections")

    st.error("⚠️ **DANGER ZONE** - This will delete the ENTIRE vector database!")

    total_docs = sum(stats.get('total_documents', 0)
                     for stats in st.session_state.stats.values()) if st.session_state.stats else 0

    st.warning(f"This will permanently delete **{total_docs:,} documents** across all collections!")

    confirm1 = st.checkbox("I understand this will delete all data")
    confirm2 = st.text_input("Type 'DELETE ALL' to confirm")

    if st.button("Clear All Collections", type="primary", disabled=(not confirm1 or confirm2 != "DELETE ALL")):
        with st.spinner("Clearing all collections..."):
            try:
                import shutil
                db_path = Path("./outputs/vector_db")
                if db_path.exists():
                    shutil.rmtree(db_path)
                    st.success("✓ All collections cleared successfully!")

                    # Reinitialize indexer
                    st.session_state.indexer = None
                    initialize_rag_components()

                    st.rerun()
                else:
                    st.warning("Database doesn't exist")
            except Exception as e:
                st.error(f"Error clearing database: {e}")


def render_reindex_abinitio_ui():
    """UI for re-indexing Ab Initio"""
    st.markdown("#### 🔄 Re-index Ab Initio")

    st.info("Index Ab Initio .mp files with FAWN-enhanced parser")

    # Option 1: Directory path
    st.markdown("**Option 1: Index from Directory**")
    directory_path = st.text_input("Ab Initio Directory Path", placeholder="/path/to/abinitio")

    # Option 2: File upload
    st.markdown("**Option 2: Upload Files**")
    uploaded_files = st.file_uploader(
        "Upload .mp files",
        accept_multiple_files=True,
        type=['mp'],
        key="abinitio_upload"
    )

    if st.button("Start Indexing", type="primary"):
        if directory_path and Path(directory_path).exists():
            reindex_abinitio_from_directory(directory_path)
        elif uploaded_files:
            reindex_abinitio_from_upload(uploaded_files)
        else:
            st.error("Please provide a directory path or upload files")


def render_reindex_autosys_ui():
    """UI for re-indexing Autosys"""
    st.markdown("#### 🔄 Re-index Autosys")

    st.info("Index Autosys .jil files with AI-powered analysis")

    # Option 1: Directory path
    st.markdown("**Option 1: Index from Directory**")
    directory_path = st.text_input("Autosys Directory Path", placeholder="/path/to/autosys")

    # Option 2: File upload
    st.markdown("**Option 2: Upload Files**")
    uploaded_files = st.file_uploader(
        "Upload .jil files",
        accept_multiple_files=True,
        type=['jil'],
        key="autosys_upload"
    )

    if st.button("Start Indexing", type="primary"):
        if directory_path and Path(directory_path).exists():
            reindex_autosys_from_directory(directory_path)
        elif uploaded_files:
            reindex_autosys_from_upload(uploaded_files)
        else:
            st.error("Please provide a directory path or upload files")


def render_reindex_hadoop_ui():
    """UI for re-indexing Hadoop"""
    st.markdown("#### 🔄 Re-index Hadoop")

    st.info("Index Hadoop workflows (Pig, Hive, Oozie XML)")

    # Option 1: Directory path
    st.markdown("**Option 1: Index from Directory**")
    directory_path = st.text_input("Hadoop Directory Path", placeholder="/path/to/hadoop")

    # Option 2: File upload
    st.markdown("**Option 2: Upload Files**")
    uploaded_files = st.file_uploader(
        "Upload Hadoop files",
        accept_multiple_files=True,
        type=['pig', 'hql', 'xml', 'py'],
        key="hadoop_upload"
    )

    if st.button("Start Indexing", type="primary"):
        if directory_path and Path(directory_path).exists():
            reindex_hadoop_from_directory(directory_path)
        elif uploaded_files:
            reindex_hadoop_from_upload(uploaded_files)
        else:
            st.error("Please provide a directory path or upload files")


def render_reindex_databricks_ui():
    """UI for re-indexing Databricks"""
    st.markdown("#### 🔄 Re-index Databricks")

    st.info("Index Databricks notebooks (Python, Scala, SQL)")

    # Option 1: Directory path
    st.markdown("**Option 1: Index from Directory**")
    directory_path = st.text_input("Databricks Directory Path", placeholder="/path/to/databricks")

    # Option 2: File upload
    st.markdown("**Option 2: Upload Files**")
    uploaded_files = st.file_uploader(
        "Upload Databricks notebooks",
        accept_multiple_files=True,
        type=['py', 'scala', 'sql', 'ipynb'],
        key="databricks_upload"
    )

    if st.button("Start Indexing", type="primary"):
        if directory_path and Path(directory_path).exists():
            reindex_databricks_from_directory(directory_path)
        elif uploaded_files:
            reindex_databricks_from_upload(uploaded_files)
        else:
            st.error("Please provide a directory path or upload files")


def render_reindex_documents_ui():
    """UI for re-indexing documents"""
    st.markdown("#### 🔄 Re-index Documents")

    st.info("Index PDF, Excel, Word, and Markdown documents")

    # Option 1: Directory path
    st.markdown("**Option 1: Index from Directory**")
    directory_path = st.text_input("Documents Directory Path", placeholder="/path/to/documents")
    recursive = st.checkbox("Include subdirectories", value=True)

    # Option 2: File upload
    st.markdown("**Option 2: Upload Files**")
    uploaded_files = st.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=['pdf', 'xlsx', 'xls', 'docx', 'txt', 'md'],
        key="docs_upload"
    )

    if st.button("Start Indexing", type="primary"):
        if directory_path and Path(directory_path).exists():
            reindex_documents_from_directory(directory_path, recursive)
        elif uploaded_files:
            reindex_documents_from_upload(uploaded_files)
        else:
            st.error("Please provide a directory path or upload files")


def render_reindex_all_ui():
    """UI for re-indexing everything"""
    st.markdown("#### 🔄 Re-index Everything")

    st.warning("⚠️ This will clear all existing data and re-index from scratch")

    st.markdown("**Provide paths for each system (leave blank to skip):**")

    abinitio_path = st.text_input("Ab Initio Directory", placeholder="/path/to/abinitio or leave blank")
    autosys_path = st.text_input("Autosys Directory", placeholder="/path/to/autosys or leave blank")
    documents_path = st.text_input("Documents Directory", placeholder="/path/to/documents or leave blank")

    confirm = st.checkbox("Clear existing data and re-index")

    if st.button("Start Full Re-index", type="primary", disabled=not confirm):
        reindex_all(abinitio_path, autosys_path, documents_path)


def render_export_stats_ui():
    """UI for exporting statistics"""
    st.markdown("#### 📤 Export Statistics")

    st.info("Export database statistics to JSON file")

    if st.button("Export Stats", type="primary"):
        try:
            import json

            output = {
                "timestamp": datetime.now().isoformat(),
                "total_documents": sum(stats.get('total_documents', 0)
                                     for stats in st.session_state.stats.values()),
                "collections": st.session_state.stats
            }

            output_json = json.dumps(output, indent=2)

            st.download_button(
                label="Download JSON",
                data=output_json,
                file_name=f"vector_db_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

            st.success("✓ Stats ready for download!")

        except Exception as e:
            st.error(f"Export error: {e}")


# Indexing helper functions

def reindex_abinitio_from_directory(directory_path: str):
    """Re-index Ab Initio from directory"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Parsing Ab Initio files...")
        progress_bar.progress(20)

        parser = AbInitioParser()
        result = parser.parse_directory(directory_path)

        progress_bar.progress(50)
        status_text.text("Indexing graphs and components...")

        if st.session_state.indexer:
            stats = st.session_state.indexer.index_abinitio(
                processes=result.get("processes", []),
                components=result.get("components", [])
            )

            progress_bar.progress(100)
            status_text.empty()

            st.success(f"✓ Indexed {stats.get('graphs', 0)} graphs and {stats.get('components', 0)} components")

            # Refresh stats
            st.session_state.stats = st.session_state.indexer.get_stats()
            st.session_state.indexed_files['abinitio'].append(f"Directory: {directory_path}")

    except Exception as e:
        st.error(f"Error indexing Ab Initio: {e}")
    finally:
        progress_bar.empty()


def reindex_abinitio_from_upload(uploaded_files):
    """Re-index Ab Initio from uploaded files"""
    # Reuse existing file upload logic
    index_uploaded_files(uploaded_files)


def reindex_autosys_from_directory(directory_path: str):
    """Re-index Autosys from directory"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Parsing Autosys files...")
        progress_bar.progress(20)

        parser = AutosysParser()
        result = parser.parse_directory(directory_path)

        progress_bar.progress(50)
        status_text.text("Indexing jobs...")

        if st.session_state.indexer:
            jobs_dict = [job.__dict__ for job in result.get("components", [])]
            stats = st.session_state.indexer.index_autosys(jobs=jobs_dict)

            progress_bar.progress(100)
            status_text.empty()

            st.success(f"✓ Indexed {stats.get('jobs', 0)} Autosys jobs")

            # Refresh stats
            st.session_state.stats = st.session_state.indexer.get_stats()
            st.session_state.indexed_files['autosys'].append(f"Directory: {directory_path}")

    except Exception as e:
        st.error(f"Error indexing Autosys: {e}")
    finally:
        progress_bar.empty()


def reindex_autosys_from_upload(uploaded_files):
    """Re-index Autosys from uploaded files"""
    index_uploaded_files(uploaded_files)


def reindex_hadoop_from_directory(directory_path: str):
    """Re-index Hadoop from directory"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Parsing Hadoop workflows...")
        progress_bar.progress(20)

        parser = HadoopParser()
        result = parser.parse_directory(directory_path)

        progress_bar.progress(50)
        status_text.text("Indexing workflows and scripts...")

        if st.session_state.indexer and result.get("processes"):
            # Create documents from parsed processes/components
            documents = []
            for process in result.get("processes", []):
                doc = {
                    "id": process.id,
                    "content": f"{process.name}\n{process.description or ''}",
                    "doc_type": "hadoop_workflow",
                    "system": "hadoop",
                    "metadata": {
                        "process_name": process.name,
                        "source_path": process.source_path
                    }
                }
                documents.append(doc)

            # Index documents
            st.session_state.indexer.collections["hadoop_collection"].index_documents(documents)

            progress_bar.progress(100)
            status_text.empty()

            st.success(f"✓ Indexed {len(documents)} Hadoop workflows")

            # Refresh stats
            st.session_state.stats = st.session_state.indexer.get_stats()
            st.session_state.indexed_files['hadoop'] = st.session_state.indexed_files.get('hadoop', [])
            st.session_state.indexed_files['hadoop'].append(f"Directory: {directory_path}")
        else:
            st.warning("No Hadoop workflows found in the directory")

    except Exception as e:
        st.error(f"Error indexing Hadoop: {e}")
        logger.error(f"Hadoop indexing error: {e}", exc_info=True)
    finally:
        progress_bar.empty()


def reindex_hadoop_from_upload(uploaded_files):
    """Re-index Hadoop from uploaded files"""
    index_uploaded_files(uploaded_files)


def reindex_databricks_from_directory(directory_path: str):
    """Re-index Databricks from directory"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Parsing Databricks notebooks...")
        progress_bar.progress(20)

        parser = DatabricksParser()
        result = parser.parse_directory(directory_path)

        progress_bar.progress(50)
        status_text.text("Indexing notebooks...")

        if st.session_state.indexer and result.get("processes"):
            # Create documents from parsed processes/components
            documents = []
            for process in result.get("processes", []):
                doc = {
                    "id": process.id,
                    "content": f"{process.name}\n{process.description or ''}",
                    "doc_type": "databricks_notebook",
                    "system": "databricks",
                    "metadata": {
                        "process_name": process.name,
                        "source_path": process.source_path
                    }
                }
                documents.append(doc)

            # Index documents
            st.session_state.indexer.collections["databricks_collection"].index_documents(documents)

            progress_bar.progress(100)
            status_text.empty()

            st.success(f"✓ Indexed {len(documents)} Databricks notebooks")

            # Refresh stats
            st.session_state.stats = st.session_state.indexer.get_stats()
            st.session_state.indexed_files['databricks'] = st.session_state.indexed_files.get('databricks', [])
            st.session_state.indexed_files['databricks'].append(f"Directory: {directory_path}")
        else:
            st.warning("No Databricks notebooks found in the directory")

    except Exception as e:
        st.error(f"Error indexing Databricks: {e}")
        logger.error(f"Databricks indexing error: {e}", exc_info=True)
    finally:
        progress_bar.empty()


def reindex_databricks_from_upload(uploaded_files):
    """Re-index Databricks from uploaded files"""
    index_uploaded_files(uploaded_files)


def reindex_documents_from_directory(directory_path: str, recursive: bool = True):
    """Re-index documents from directory"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Parsing documents...")
        progress_bar.progress(20)

        parser = DocumentParser()
        docs = parser.parse_directory(directory_path, recursive=recursive)

        progress_bar.progress(50)
        status_text.text("Indexing document chunks...")

        if st.session_state.indexer:
            stats = parser.index_documents(docs, st.session_state.indexer)

            progress_bar.progress(100)
            status_text.empty()

            st.success(f"✓ Indexed {stats.get('files', 0)} documents ({stats.get('total', 0)} chunks)")

            # Refresh stats
            st.session_state.stats = st.session_state.indexer.get_stats()
            st.session_state.indexed_files['documents'].append(f"Directory: {directory_path}")

    except Exception as e:
        st.error(f"Error indexing documents: {e}")
    finally:
        progress_bar.empty()


def reindex_documents_from_upload(uploaded_files):
    """Re-index documents from uploaded files"""
    index_uploaded_files(uploaded_files)


def reindex_all(abinitio_path: str = "", autosys_path: str = "", documents_path: str = ""):
    """Re-index all systems"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Clear database
        status_text.text("Step 1/4: Clearing existing database...")
        progress_bar.progress(10)

        import shutil
        db_path = Path("./outputs/vector_db")
        if db_path.exists():
            shutil.rmtree(db_path)

        # Step 2: Reinitialize
        status_text.text("Step 2/4: Initializing fresh indexer...")
        progress_bar.progress(20)

        st.session_state.indexer = MultiCollectionIndexer()

        # Step 3: Index each system
        total_steps = sum([bool(abinitio_path), bool(autosys_path), bool(documents_path)])
        current_step = 0

        if abinitio_path and Path(abinitio_path).exists():
            current_step += 1
            status_text.text(f"Step 3/4: Indexing Ab Initio ({current_step}/{total_steps})...")
            progress_bar.progress(30 + (current_step * 20))
            reindex_abinitio_from_directory(abinitio_path)

        if autosys_path and Path(autosys_path).exists():
            current_step += 1
            status_text.text(f"Step 3/4: Indexing Autosys ({current_step}/{total_steps})...")
            progress_bar.progress(30 + (current_step * 20))
            reindex_autosys_from_directory(autosys_path)

        if documents_path and Path(documents_path).exists():
            current_step += 1
            status_text.text(f"Step 3/4: Indexing Documents ({current_step}/{total_steps})...")
            progress_bar.progress(30 + (current_step * 20))
            reindex_documents_from_directory(documents_path)

        # Step 4: Finalize
        status_text.text("Step 4/4: Finalizing...")
        progress_bar.progress(90)

        st.session_state.stats = st.session_state.indexer.get_stats()

        progress_bar.progress(100)
        status_text.empty()

        st.success("✓ Full re-index complete!")
        st.balloons()

    except Exception as e:
        st.error(f"Error during re-indexing: {e}")
    finally:
        progress_bar.empty()


# ============================================
# Main Application
# ============================================

def main():
    """Main application entry point"""

    # Initialize session state
    initialize_session_state()

    # Initialize RAG components
    initialize_rag_components()

    # Render sidebar
    render_sidebar()

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "🔄 Compare", "📊 Analytics", "⚙️ Database", "🔗 Lineage"])

    with tab1:
        render_chat_interface()

    with tab2:
        render_comparison_mode()

    with tab3:
        render_analytics_dashboard()

    with tab4:
        render_database_management()

    with tab5:
        render_lineage_tab()


def render_analytics_dashboard():
    """Render analytics and query history"""

    st.subheader("📊 Analytics Dashboard")

    # Query history
    if st.session_state.query_history:
        st.markdown("### 📜 Query History")

        history_df = pd.DataFrame(st.session_state.query_history)
        st.dataframe(history_df, use_container_width=True)

    # Collection statistics
    st.markdown("### 📈 Collection Statistics")

    if st.session_state.stats:
        stats_data = []
        for coll, stats in st.session_state.stats.items():
            stats_data.append({
                "Collection": coll.replace('_collection', '').title(),
                "Documents": stats.get('total_documents', 0)
            })

        stats_df = pd.DataFrame(stats_data)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(stats_df.set_index("Collection"))
        with col2:
            st.dataframe(stats_df, use_container_width=True)

    # Indexed files
    st.markdown("### 📁 Indexed Files")

    for system, files in st.session_state.indexed_files.items():
        if files:
            with st.expander(f"{system.title()} ({len(files)} files)"):
                for file in files:
                    st.write(f"- {file}")


# ============================================
# Run Application
# ============================================

if __name__ == "__main__":
    main()
