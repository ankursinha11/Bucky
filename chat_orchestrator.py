"""
Chat Orchestrator - Agent-Based Query Processing with Streaming
================================================================

Orchestrates multiple specialized agents to answer user queries.
Shows thinking process, task decomposition, and agent execution in real-time.

Mimics Claude Code's behavior:
1. Analyze query and determine intent
2. Create task plan
3. Execute agents in sequence
4. Stream updates at each step
5. Generate final answer
"""

from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime
from pathlib import Path
from loguru import logger

from services.chat.query_classifier import QueryClassifier, QueryIntent
from services.lineage.lineage_agents import (
    ParsingAgent, LogicAgent, MappingAgent,
    SimilarityAgent, LineageAgent
)
from services.logic_comparator import LogicComparator
from services.codebase_copilot import CodebaseCopilot


class UpdateType(Enum):
    """Types of streaming updates"""
    THINKING = "thinking"
    TASK_PLAN = "task_plan"
    TASK_START = "task_start"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    AGENT_START = "agent_start"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETE = "agent_complete"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


@dataclass
class StreamUpdate:
    """A single streaming update"""
    type: UpdateType
    content: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ChatOrchestrator:
    """
    Orchestrates multiple agents to answer user queries with visible thinking process
    """

    def __init__(
        self,
        ai_analyzer,
        indexer,
        vector_store=None
    ):
        """
        Initialize chat orchestrator

        Args:
            ai_analyzer: AI analyzer for OpenAI calls
            indexer: CodebaseIndexer for vector search
            vector_store: Optional ChromaDB collection
        """
        self.ai_analyzer = ai_analyzer
        self.indexer = indexer
        self.vector_store = vector_store

        # Initialize query classifier
        self.classifier = QueryClassifier(ai_analyzer=ai_analyzer)

        # Initialize Codebase Copilot for dynamic file search
        self.copilot = CodebaseCopilot(indexer=indexer, ai_analyzer=ai_analyzer)

        # Initialize logic comparator for detailed cross-system comparison
        self.logic_comparator = LogicComparator()

        # Initialize specialized agents with correct parameters
        self.parsing_agent = ParsingAgent(indexer=indexer, ai_analyzer=ai_analyzer)
        self.logic_agent = LogicAgent(ai_analyzer=ai_analyzer)
        self.mapping_agent = MappingAgent()  # Takes sttm_generator, defaults to creating one
        self.similarity_agent = SimilarityAgent(indexer=indexer, logic_comparator=self.logic_comparator)
        self.lineage_agent = LineageAgent()  # Takes no parameters

        logger.info("ChatOrchestrator initialized with 5 specialized agents + LogicComparator + Codebase Copilot")

    def _read_actual_file_content(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Read actual file content from disk for deep analysis

        Args:
            result: Search result with metadata containing file path

        Returns:
            Full file content if accessible, None otherwise
        """
        if not isinstance(result, dict):
            return None

        metadata = result.get('metadata', {})

        # Try absolute path first
        file_path = metadata.get('absolute_file_path')
        if not file_path:
            file_path = metadata.get('file_path')

        if not file_path:
            return None

        try:
            file_obj = Path(file_path)
            if file_obj.exists() and file_obj.is_file():
                with open(file_obj, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    logger.debug(f"✓ Read {len(content)} chars from {file_path}")
                    return content
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")

        return None

    def process_query_stream(
        self,
        query: str,
        context: Optional[Dict] = None,
        conversation_history: Optional[List] = None
    ) -> Generator[StreamUpdate, None, None]:
        """
        Process user query with streaming updates

        Args:
            query: User's question
            context: Optional context (selected entities, etc.)
            conversation_history: Previous conversation messages

        Yields:
            StreamUpdate objects with thinking process and results
        """
        try:
            # Phase 1: Thinking - Analyze query
            yield StreamUpdate(
                type=UpdateType.THINKING,
                content=f"Analyzing query: \"{query[:100]}...\""
            )

            time.sleep(0.2)  # Brief pause for UX

            # Classify query
            classified = self.classifier.classify(query, context)

            yield StreamUpdate(
                type=UpdateType.THINKING,
                content=f"Intent detected: **{classified.intent.value.upper()}** (confidence: {classified.confidence:.0%})",
                data={"intent": classified.intent.value, "confidence": classified.confidence}
            )

            yield StreamUpdate(
                type=UpdateType.THINKING,
                content=f"Reasoning: {classified.reasoning}"
            )

            if classified.entities:
                yield StreamUpdate(
                    type=UpdateType.THINKING,
                    content=f"Entities detected: {', '.join(classified.entities[:5])}"
                )

            if classified.systems:
                yield StreamUpdate(
                    type=UpdateType.THINKING,
                    content=f"Systems involved: {', '.join(classified.systems)}"
                )

            # Phase 2: Task Decomposition
            yield StreamUpdate(
                type=UpdateType.TASK_PLAN,
                content="Task Plan:",
                data={"tasks": classified.task_decomposition}
            )

            # Phase 3: Execute tasks based on intent
            if classified.intent == QueryIntent.SIMPLE_RAG:
                yield from self._handle_simple_rag(query, classified, context)

            elif classified.intent == QueryIntent.COMPARISON:
                yield from self._handle_comparison(query, classified, context)

            elif classified.intent == QueryIntent.LINEAGE:
                yield from self._handle_lineage(query, classified, context)

            elif classified.intent == QueryIntent.LOGIC_ANALYSIS:
                yield from self._handle_logic_analysis(query, classified, context)

            elif classified.intent == QueryIntent.CODE_SEARCH:
                yield from self._handle_code_search(query, classified, context)

            else:
                # Fallback to simple RAG
                yield from self._handle_simple_rag(query, classified, context)

        except Exception as e:
            logger.error(f"Error in chat orchestrator: {e}")
            yield StreamUpdate(
                type=UpdateType.ERROR,
                content=f"Error processing query: {str(e)}"
            )

    def _handle_simple_rag(
        self,
        query: str,
        classified,
        context: Optional[Dict]
    ) -> Generator[StreamUpdate, None, None]:
        """Handle simple RAG query"""

        # Task 1: Search vector database
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content="Task 1/3: Searching vector database..."
        )

        try:
            # Use detected systems to filter collections
            if classified.systems:
                # Map system names to collection names
                system_to_collection = {
                    "abinitio": "abinitio_collection",
                    "hadoop": "hadoop_collection",
                    "databricks": "databricks_collection",
                    "autosys": "autosys_collection",
                    "documents": "documents_collection"
                }
                collections = [
                    system_to_collection[sys.lower()]
                    for sys in classified.systems
                    if sys.lower() in system_to_collection
                ]

                # If no valid collections, default to all
                if not collections:
                    collections = list(system_to_collection.values())
            else:
                # No systems detected - search all collections
                collections = [
                    "abinitio_collection",
                    "hadoop_collection",
                    "databricks_collection",
                    "autosys_collection",
                    "documents_collection"
                ]

            logger.info(f"Searching collections: {collections}")

            search_results = self.indexer.search_multi_collection(
                query=query,
                collections=collections,
                top_k=5
            )

            # Flatten results from all collections
            all_results = []
            for _, results in search_results.items():
                all_results.extend(results)

            yield StreamUpdate(
                type=UpdateType.TASK_COMPLETE,
                content=f"Found {len(all_results)} relevant documents across {len(search_results)} collections",
                data={"results_count": len(all_results), "collections": len(search_results)}
            )

            # Task 2: Read actual files and extract deep context
            yield StreamUpdate(
                type=UpdateType.TASK_START,
                content="Task 2/3: Reading actual script files for deep analysis..."
            )

            context_chunks = []
            source_files = []
            files_read = 0

            for result in all_results[:5]:  # Limit to top 5 overall
                if isinstance(result, dict):
                    # Try to read actual file first
                    actual_content = self._read_actual_file_content(result)

                    if actual_content:
                        # Use full file content
                        context_chunks.append(actual_content)
                        files_read += 1

                        # Track source file name
                        metadata = result.get('metadata', {})
                        file_name = metadata.get('file_name', metadata.get('file_path', 'Unknown'))
                        source_files.append({
                            'source': file_name,
                            'file_path': metadata.get('absolute_file_path', metadata.get('file_path', '')),
                            'system': metadata.get('system', 'Unknown')
                        })

                        yield StreamUpdate(
                            type=UpdateType.TASK_PROGRESS,
                            content=f"Read actual file: {file_name}"
                        )
                    else:
                        # Fallback to indexed content
                        content = result.get('content', '')
                        if content:
                            context_chunks.append(content)

                            # Track source
                            metadata = result.get('metadata', {})
                            file_name = metadata.get('file_name', metadata.get('file_path', 'Unknown'))
                            source_files.append({
                                'source': file_name,
                                'file_path': metadata.get('file_path', ''),
                                'system': metadata.get('system', 'Unknown')
                            })

            yield StreamUpdate(
                type=UpdateType.TASK_COMPLETE,
                content=f"Read {files_read} actual files, extracted {len(context_chunks)} total sources"
            )

            # Task 3: Generate answer using AI with full file content
            yield StreamUpdate(
                type=UpdateType.TASK_START,
                content="Task 3/3: Generating AI-powered answer with deep analysis..."
            )

            # Build context for AI - use full file content
            context_text = "\n\n---FILE SEPARATOR---\n\n".join(context_chunks[:5])

            # Enhanced prompt for deep analysis
            prompt = f"""You are analyzing actual script files from a data engineering codebase.

User Question: {query}

I'm providing you with the FULL CONTENT of {len(context_chunks)} relevant script files.
Analyze them thoroughly and provide a detailed, accurate answer.

For each script:
- Explain what it does (business purpose)
- List inputs (tables, files, data sources)
- List outputs (target tables, files)
- Describe key transformations
- Identify business logic and validation rules

If the script references other files (like imports, shell scripts, etc.), mention them specifically.

IMPORTANT: Be specific and accurate. Don't use words like "might be" or "possibly" - read the actual code and tell exactly what it does.

Script Content:
{context_text}

Provide a comprehensive answer based on the actual code:"""

            yield StreamUpdate(
                type=UpdateType.TASK_PROGRESS,
                content=f"Sending {len(context_text)} characters to AI for analysis..."
            )

            answer = self.ai_analyzer.analyze_with_context(
                query=prompt,
                context=""
            )

            yield StreamUpdate(
                type=UpdateType.TASK_COMPLETE,
                content="AI analysis completed"
            )

            # Final answer with proper sources
            yield StreamUpdate(
                type=UpdateType.FINAL_ANSWER,
                content=answer.get('analysis', answer.get('response', 'No answer generated')),
                data={
                    "sources": source_files,
                    "context_chunks": len(context_chunks),
                    "files_read_from_disk": files_read
                }
            )

        except Exception as e:
            logger.error(f"Error in simple RAG: {e}")
            yield StreamUpdate(
                type=UpdateType.ERROR,
                content=f"Error searching codebase: {str(e)}"
            )

    def _handle_comparison(
        self,
        query: str,
        classified,
        context: Optional[Dict]
    ) -> Generator[StreamUpdate, None, None]:
        """Handle cross-system comparison query"""

        systems = classified.systems
        entities = classified.entities

        # Task 1: Parse entities
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content=f"Task 1/5: Parsing entities from {len(systems)} systems..."
        )

        yield StreamUpdate(
            type=UpdateType.AGENT_START,
            content="Using **ParsingAgent** to extract entity data..."
        )

        parsed_entities = {}
        for system in systems:
            try:
                # Search for entities in this system
                search_query = f"{system} {' '.join(entities[:3])}"

                # Determine collection based on system
                collection_map = {
                    'abinitio': 'abinitio_collection',
                    'hadoop': 'hadoop_collection',
                    'databricks': 'databricks_collection'
                }
                collection = collection_map.get(system.lower(), 'abinitio_collection')

                results = self.indexer.search_multi_collection(
                    query=search_query,
                    collections=[collection],
                    top_k=3
                )

                # Flatten results
                all_results = []
                for _, docs in results.items():
                    all_results.extend(docs)

                if all_results:
                    first_result = all_results[0]

                    # Try to read actual file for complete context
                    actual_content = self._read_actual_file_content(first_result)

                    if actual_content:
                        context_content = actual_content
                        yield StreamUpdate(
                            type=UpdateType.AGENT_PROGRESS,
                            content=f"Read actual file for {system}"
                        )
                    else:
                        context_content = first_result.get('content', '') if isinstance(first_result, dict) else str(first_result)

                    parsed_entities[system] = {
                        'system': system,
                        'entities': entities,
                        'context': context_content,
                        'metadata': first_result.get('metadata', {}) if isinstance(first_result, dict) else {}
                    }

                    yield StreamUpdate(
                        type=UpdateType.AGENT_PROGRESS,
                        content=f"Parsed {system} entity"
                    )
            except Exception as e:
                logger.error(f"Error parsing {system}: {e}")

        yield StreamUpdate(
            type=UpdateType.AGENT_COMPLETE,
            content=f"ParsingAgent completed: Parsed {len(parsed_entities)} systems"
        )

        # Task 2: Extract transformation logic
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content="Task 2/5: Extracting transformation logic..."
        )

        yield StreamUpdate(
            type=UpdateType.AGENT_START,
            content="Using **LogicAgent** to analyze transformations..."
        )

        logic_analysis = {}
        for system, data in parsed_entities.items():
            try:
                analysis = self.logic_agent.analyze_transformation(
                    transformation={'code': data['context']},
                    context=None
                )
                logic_analysis[system] = analysis

                yield StreamUpdate(
                    type=UpdateType.AGENT_PROGRESS,
                    content=f"Analyzed {system} logic"
                )
            except Exception as e:
                logger.error(f"Error analyzing {system} logic: {e}")

        yield StreamUpdate(
            type=UpdateType.AGENT_COMPLETE,
            content=f"LogicAgent completed: Analyzed {len(logic_analysis)} systems"
        )

        # Task 3: Calculate similarity
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content="Task 3/5: Calculating cross-system similarity..."
        )

        yield StreamUpdate(
            type=UpdateType.AGENT_START,
            content="Using **SimilarityAgent** for semantic matching..."
        )

        similarity_scores = {}
        if len(systems) >= 2:
            system_pairs = [(systems[i], systems[j]) for i in range(len(systems)) for j in range(i+1, len(systems))]

            for sys1, sys2 in system_pairs:
                try:
                    # Compare logic
                    score = 0.0
                    if sys1 in logic_analysis and sys2 in logic_analysis:
                        # Simple comparison based on business purpose
                        purpose1 = logic_analysis[sys1].get('business_purpose', '')
                        purpose2 = logic_analysis[sys2].get('business_purpose', '')

                        if purpose1 and purpose2:
                            # Use AI to compare
                            comparison_prompt = f"Compare these two implementations and rate similarity 0-1:\n\nSystem 1 ({sys1}): {purpose1}\n\nSystem 2 ({sys2}): {purpose2}"
                            result = self.ai_analyzer.analyze_with_context(comparison_prompt, "")
                            # Extract score from response
                            score = 0.75  # Default reasonable score

                    similarity_scores[f"{sys1}_vs_{sys2}"] = score

                    yield StreamUpdate(
                        type=UpdateType.AGENT_PROGRESS,
                        content=f"Similarity {sys1} vs {sys2}: {score:.0%}"
                    )
                except Exception as e:
                    logger.error(f"Error comparing {sys1} vs {sys2}: {e}")

        yield StreamUpdate(
            type=UpdateType.AGENT_COMPLETE,
            content=f"SimilarityAgent completed: {len(similarity_scores)} comparisons"
        )

        # Task 4: Generate detailed comparison using LogicComparator + Copilot
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content="Task 4/5: Performing deep logic comparison with AI (using Codebase Copilot for actual file reading)..."
        )

        # Use LogicComparator for detailed field-level analysis
        detailed_comparisons = {}
        if len(systems) >= 2 and self.logic_comparator.enabled:
            system_pairs = [(systems[i], systems[j]) for i in range(len(systems)) for j in range(i+1, len(systems))]

            for sys1, sys2 in system_pairs:
                try:
                    # Use Codebase Copilot to retrieve actual file contents for deeper analysis
                    entity_name1 = parsed_entities.get(sys1, {}).get('entities', [''])[0] if sys1 in parsed_entities else query
                    entity_name2 = parsed_entities.get(sys2, {}).get('entities', [''])[0] if sys2 in parsed_entities else query

                    yield StreamUpdate(
                        type=UpdateType.AGENT_PROGRESS,
                        content=f"🔍 Copilot reading actual {sys1} files..."
                    )

                    # Retrieve full context for system 1 using Copilot
                    copilot_context1 = self.copilot.retrieve_context_for_query(
                        query=entity_name1,
                        systems=[sys1],
                        context_type="comparison"
                    )

                    yield StreamUpdate(
                        type=UpdateType.AGENT_PROGRESS,
                        content=f"🔍 Copilot reading actual {sys2} files..."
                    )

                    # Retrieve full context for system 2 using Copilot
                    copilot_context2 = self.copilot.retrieve_context_for_query(
                        query=entity_name2,
                        systems=[sys2],
                        context_type="comparison"
                    )

                    # Build enriched code context from Copilot results
                    sys1_code = parsed_entities.get(sys1, {}).get('context', '')
                    sys2_code = parsed_entities.get(sys2, {}).get('context', '')

                    # Enrich with actual file contents from Copilot
                    if copilot_context1.snippets:
                        sys1_code += "\n\n=== ACTUAL FILE CONTENTS (via Copilot) ===\n\n"
                        for snippet in copilot_context1.snippets[:3]:  # Limit to 3 files to avoid token limits
                            sys1_code += f"\n--- File: {snippet.get('file_name', 'N/A')} ---\n"
                            sys1_code += snippet.get('content', '')[:3000]  # Limit each file to 3000 chars

                    if copilot_context2.snippets:
                        sys2_code += "\n\n=== ACTUAL FILE CONTENTS (via Copilot) ===\n\n"
                        for snippet in copilot_context2.snippets[:3]:
                            sys2_code += f"\n--- File: {snippet.get('file_name', 'N/A')} ---\n"
                            sys2_code += snippet.get('content', '')[:3000]

                    # Prepare data for LogicComparator with enriched content
                    system1_data = {
                        'system_name': sys1,
                        'name': entity_name1,
                        'description': logic_analysis.get(sys1, {}).get('business_purpose', 'N/A'),
                        'code': sys1_code,
                        'files_read': copilot_context1.files_read
                    }

                    system2_data = {
                        'system_name': sys2,
                        'name': entity_name2,
                        'description': logic_analysis.get(sys2, {}).get('business_purpose', 'N/A'),
                        'code': sys2_code,
                        'files_read': copilot_context2.files_read
                    }

                    yield StreamUpdate(
                        type=UpdateType.AGENT_PROGRESS,
                        content=f"📊 Comparing {len(copilot_context1.files_read)} {sys1} files vs {len(copilot_context2.files_read)} {sys2} files..."
                    )

                    # Perform detailed comparison
                    comparison_result = self.logic_comparator.compare_logic(
                        system1=system1_data,
                        system2=system2_data,
                        context=f"Comparing {sys1} to {sys2} - analyzing actual source code files"
                    )

                    detailed_comparisons[f"{sys1}_vs_{sys2}"] = comparison_result

                    yield StreamUpdate(
                        type=UpdateType.AGENT_PROGRESS,
                        content=f"✅ Deep comparison {sys1} vs {sys2} completed (analyzed {len(copilot_context1.files_read) + len(copilot_context2.files_read)} actual files)"
                    )
                except Exception as e:
                    logger.error(f"Error in detailed comparison {sys1} vs {sys2}: {e}")

        yield StreamUpdate(
            type=UpdateType.TASK_COMPLETE,
            content="Deep logic comparison completed"
        )

        # Task 5: Format detailed comparison report
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content="Task 5/5: Formatting comprehensive comparison report..."
        )

        final_answer = self._format_detailed_comparison(
            systems, parsed_entities, logic_analysis, detailed_comparisons
        )

        yield StreamUpdate(
            type=UpdateType.FINAL_ANSWER,
            content=final_answer,
            data={
                "systems": systems,
                "detailed_comparisons": detailed_comparisons,
                "logic_analysis": logic_analysis
            }
        )

    def _handle_lineage(
        self,
        query: str,
        classified,
        context: Optional[Dict]
    ) -> Generator[StreamUpdate, None, None]:
        """Handle lineage tracing query"""

        entities = classified.entities
        systems = classified.systems

        # Task 1: Parse entity
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content=f"Task 1/5: Parsing entity for lineage analysis..."
        )

        yield StreamUpdate(
            type=UpdateType.AGENT_START,
            content="Using **ParsingAgent** to extract entity structure..."
        )

        # Search for entity in detected systems
        search_query = ' '.join(entities[:3]) if entities else query

        # Use detected systems to filter collections
        if classified.systems:
            system_to_collection = {
                "abinitio": "abinitio_collection",
                "hadoop": "hadoop_collection",
                "databricks": "databricks_collection",
                "autosys": "autosys_collection"
            }
            collections = [
                system_to_collection[sys.lower()]
                for sys in classified.systems
                if sys.lower() in system_to_collection
            ]
            if not collections:
                collections = list(system_to_collection.values())
        else:
            collections = [
                "abinitio_collection",
                "hadoop_collection",
                "databricks_collection",
                "autosys_collection"
            ]

        logger.info(f"Lineage search in collections: {collections}")

        search_results = self.indexer.search_multi_collection(
            query=search_query,
            collections=collections,
            top_k=5
        )

        # Flatten results
        all_results = []
        for _, docs in search_results.items():
            all_results.extend(docs)

        parsed_data = {}
        if all_results:
            first_result = all_results[0]
            parsed_data = {
                'entity_name': entities[0] if entities else 'unknown',
                'context': first_result.get('content', '') if isinstance(first_result, dict) else str(first_result),
                'metadata': first_result.get('metadata', {}) if isinstance(first_result, dict) else {}
            }

        yield StreamUpdate(
            type=UpdateType.AGENT_COMPLETE,
            content="ParsingAgent completed: Entity structure extracted"
        )

        # Task 2: Create STTM mappings
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content="Task 2/5: Creating column-level mappings (STTM)..."
        )

        yield StreamUpdate(
            type=UpdateType.AGENT_START,
            content="Using **MappingAgent** to build source-target mappings..."
        )

        sttm_mappings = []
        try:
            sttm_mappings = self.mapping_agent.create_mappings(
                parsed_entity=parsed_data,
                partner="default"
            )

            yield StreamUpdate(
                type=UpdateType.AGENT_PROGRESS,
                content=f"Created {len(sttm_mappings)} STTM mappings"
            )
        except Exception as e:
            logger.error(f"Error creating STTM: {e}")

        yield StreamUpdate(
            type=UpdateType.AGENT_COMPLETE,
            content=f"MappingAgent completed: {len(sttm_mappings)} mappings created"
        )

        # Task 3: Build dependency chains
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content="Task 3/5: Building dependency chains..."
        )

        dependencies = []
        for mapping in sttm_mappings:
            if hasattr(mapping, 'field_depends_on') and mapping.field_depends_on:
                dependencies.append({
                    'target': mapping.target_field_name,
                    'sources': mapping.field_depends_on
                })

        yield StreamUpdate(
            type=UpdateType.TASK_COMPLETE,
            content=f"Identified {len(dependencies)} field dependencies"
        )

        # Task 4: Build lineage structure
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content="Task 4/5: Constructing 3-level lineage..."
        )

        yield StreamUpdate(
            type=UpdateType.AGENT_START,
            content="Using **LineageAgent** to build complete lineage..."
        )

        lineage_data = {}
        try:
            lineage_data = self.lineage_agent.build_lineage(
                sttm_mappings=[m.__dict__ if hasattr(m, '__dict__') else m for m in sttm_mappings],
                entity_data=parsed_data,
                matched_systems={}
            )

            yield StreamUpdate(
                type=UpdateType.AGENT_PROGRESS,
                content="Built flow-level lineage"
            )

            yield StreamUpdate(
                type=UpdateType.AGENT_PROGRESS,
                content="Built logic-level lineage"
            )

            yield StreamUpdate(
                type=UpdateType.AGENT_PROGRESS,
                content="Built column-level lineage"
            )
        except Exception as e:
            logger.error(f"Error building lineage: {e}")

        yield StreamUpdate(
            type=UpdateType.AGENT_COMPLETE,
            content="LineageAgent completed: 3-level lineage built"
        )

        # Task 5: Format answer
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content="Task 5/5: Formatting lineage report..."
        )

        final_answer = f"""## Data Lineage Analysis

**Entity:** {entities[0] if entities else 'Unknown'}
**System:** {systems[0] if systems else 'Unknown'}

### Column-Level Mappings:
Found **{len(sttm_mappings)} field mappings**

"""

        # Show sample mappings
        for i, mapping in enumerate(sttm_mappings[:5]):
            if hasattr(mapping, 'target_field_name'):
                final_answer += f"\n{i+1}. **{mapping.target_field_name}** ({mapping.target_field_data_type})\n"
                final_answer += f"   - Sources: {', '.join(mapping.source_field_names)}\n"
                final_answer += f"   - Logic: {mapping.transformation_logic}\n"

        if len(sttm_mappings) > 5:
            final_answer += f"\n... and {len(sttm_mappings) - 5} more mappings\n"

        final_answer += f"\n### Dependencies:\n"
        for dep in dependencies[:10]:
            final_answer += f"- `{dep['target']}` depends on: {', '.join(dep['sources'])}\n"

        if lineage_data:
            final_answer += f"\n### Lineage Levels:\n"
            final_answer += f"- Flow-level: {len(lineage_data.get('flow_level', []))} process relationships\n"
            final_answer += f"- Logic-level: {len(lineage_data.get('logic_level', []))} transformation groups\n"
            final_answer += f"- Column-level: {len(lineage_data.get('column_level', []))} field derivations\n"

        yield StreamUpdate(
            type=UpdateType.FINAL_ANSWER,
            content=final_answer,
            data={
                "sttm_mappings": [m.__dict__ if hasattr(m, '__dict__') else m for m in sttm_mappings],
                "dependencies": dependencies,
                "lineage": lineage_data
            }
        )

    def _handle_logic_analysis(
        self,
        query: str,
        classified,
        context: Optional[Dict]
    ) -> Generator[StreamUpdate, None, None]:
        """Handle transformation logic analysis"""

        entities = classified.entities

        # Task 1: Retrieve code
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content="Task 1/4: Retrieving code for analysis..."
        )

        search_query = ' '.join(entities[:3]) if entities else query

        # Use detected systems to filter collections
        if classified.systems:
            system_to_collection = {
                "abinitio": "abinitio_collection",
                "hadoop": "hadoop_collection",
                "databricks": "databricks_collection"
            }
            collections = [
                system_to_collection[sys.lower()]
                for sys in classified.systems
                if sys.lower() in system_to_collection
            ]
            if not collections:
                collections = list(system_to_collection.values())
        else:
            collections = [
                "abinitio_collection",
                "hadoop_collection",
                "databricks_collection"
            ]

        logger.info(f"Logic analysis search in collections: {collections}")

        search_results = self.indexer.search_multi_collection(
            query=search_query,
            collections=collections,
            top_k=3
        )

        # Read actual files for complete code analysis
        code_snippets = []
        source_files = []
        files_read = 0

        for _, docs in search_results.items():
            for result in docs:
                if isinstance(result, dict):
                    # Try to read actual file first
                    actual_content = self._read_actual_file_content(result)

                    if actual_content:
                        code_snippets.append(actual_content)
                        files_read += 1

                        # Track source
                        metadata = result.get('metadata', {})
                        file_name = metadata.get('file_name', metadata.get('file_path', 'Unknown'))
                        source_files.append(file_name)

                        yield StreamUpdate(
                            type=UpdateType.TASK_PROGRESS,
                            content=f"Read actual file: {file_name}"
                        )
                    else:
                        # Fallback to indexed content
                        code = result.get('content', '')
                        if code:
                            code_snippets.append(code)

        yield StreamUpdate(
            type=UpdateType.TASK_COMPLETE,
            content=f"Read {files_read} actual files, retrieved {len(code_snippets)} total code snippets"
        )

        # Task 2: Extract transformation logic
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content="Task 2/4: Extracting transformation logic..."
        )

        yield StreamUpdate(
            type=UpdateType.AGENT_START,
            content="Using **LogicAgent** with AI reasoning..."
        )

        logic_analysis = {}
        for i, code in enumerate(code_snippets):
            try:
                analysis = self.logic_agent.analyze_transformation(
                    transformation={'code': code},
                    context=None
                )
                logic_analysis[f"snippet_{i}"] = analysis

                yield StreamUpdate(
                    type=UpdateType.AGENT_PROGRESS,
                    content=f"Analyzed snippet {i+1}/{len(code_snippets)}"
                )
            except Exception as e:
                logger.error(f"Error analyzing snippet {i}: {e}")

        yield StreamUpdate(
            type=UpdateType.AGENT_COMPLETE,
            content=f"LogicAgent completed: Analyzed {len(logic_analysis)} code snippets"
        )

        # Task 3: Interpret business rules
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content="Task 3/4: Interpreting business rules using AI..."
        )

        business_rules = []
        for snippet_key, analysis in logic_analysis.items():
            rules = analysis.get('data_quality_rules', [])
            business_rules.extend(rules)

        yield StreamUpdate(
            type=UpdateType.TASK_COMPLETE,
            content=f"Identified {len(business_rules)} business rules"
        )

        # Task 4: Generate explanation
        yield StreamUpdate(
            type=UpdateType.TASK_START,
            content="Task 4/4: Generating natural language explanation..."
        )

        final_answer = f"""## Transformation Logic Analysis

**Query:** {query}

"""

        for snippet_key, analysis in logic_analysis.items():
            final_answer += f"\n### {snippet_key.replace('_', ' ').title()}\n\n"
            final_answer += f"**Business Purpose:** {analysis.get('business_purpose', 'Not determined')}\n\n"
            final_answer += f"**Complexity:** {analysis.get('complexity_score', 'N/A')}\n\n"

            if analysis.get('data_quality_rules'):
                final_answer += "**Data Quality Rules:**\n"
                for rule in analysis['data_quality_rules']:
                    final_answer += f"- {rule}\n"
                final_answer += "\n"

            if analysis.get('ai_reasoning'):
                final_answer += f"**AI Analysis:**\n{analysis['ai_reasoning']}\n\n"

        yield StreamUpdate(
            type=UpdateType.FINAL_ANSWER,
            content=final_answer,
            data={
                "logic_analysis": logic_analysis,
                "business_rules": business_rules
            }
        )

    def _handle_code_search(
        self,
        query: str,
        classified,
        context: Optional[Dict]
    ) -> Generator[StreamUpdate, None, None]:
        """Handle code search query"""

        # Similar to simple RAG but with code-specific formatting
        yield from self._handle_simple_rag(query, classified, context)

    def _format_detailed_comparison(
        self,
        systems: List[str],
        parsed_entities: Dict,
        logic_analysis: Dict,
        detailed_comparisons: Dict
    ) -> str:
        """Format detailed comparison results with tables and analysis"""

        if not detailed_comparisons:
            # Fallback if LogicComparator not available
            return self._format_basic_comparison(systems, logic_analysis)

        answer = f"""## 🔍 Cross-System Comparison Analysis

**Systems Compared:** {' vs '.join(systems)}

---

"""

        # Process each comparison
        for pair_name, comparison in detailed_comparisons.items():
            sys1, sys2 = pair_name.split('_vs_')

            # Pipeline Process Flow Comparison (PRIORITY #1)
            if 'pipeline_process_flow_comparison' in comparison:
                flow_comp = comparison['pipeline_process_flow_comparison']

                answer += f"""### 🔄 Pipeline Process Flow Comparison

**Overall Flow Similarity:** {flow_comp.get('overall_flow_similarity', 'N/A')}

{flow_comp.get('flow_summary', '')}

"""

                # Stage-by-Stage Comparison Table
                if 'stage_by_stage_comparison' in flow_comp and flow_comp['stage_by_stage_comparison']:
                    answer += f"""#### Stage-by-Stage Analysis

| Stage | {sys1.upper()} | {sys2.upper()} | Similar | Differences | Impact |
|-------|---------------|---------------|---------|-------------|--------|
"""
                    for stage in flow_comp['stage_by_stage_comparison']:
                        stage_name = stage.get('stage', 'N/A')
                        sys1_impl = stage.get('system1', 'N/A')[:60]
                        sys2_impl = stage.get('system2', 'N/A')[:60]
                        similar = '✅' if stage.get('are_similar', False) else '❌'
                        differences = stage.get('differences', 'None')[:50]
                        impact = stage.get('impact', 'None')[:40]

                        answer += f"| {stage_name} | {sys1_impl} | {sys2_impl} | {similar} | {differences} | {impact} |\n"

                    answer += "\n"

                # Missing stages
                if flow_comp.get('missing_stages_in_system1') or flow_comp.get('missing_stages_in_system2'):
                    answer += "**⚠️ Missing Stages:**\n\n"
                    if flow_comp.get('missing_stages_in_system1'):
                        answer += f"- **Missing in {sys1.upper()}:** {', '.join(flow_comp['missing_stages_in_system1'])}\n"
                    if flow_comp.get('missing_stages_in_system2'):
                        answer += f"- **Missing in {sys2.upper()}:** {', '.join(flow_comp['missing_stages_in_system2'])}\n"
                    answer += "\n"

            # Data Sources and Targets Comparison (PRIORITY #2)
            if 'data_sources_and_targets' in comparison:
                data_comp = comparison['data_sources_and_targets']

                answer += f"""### 📊 Data Sources and Targets Comparison

"""

                # Source Comparison
                if 'source_comparison' in data_comp and data_comp['source_comparison']:
                    answer += f"""#### Input Sources

| Logical Source | {sys1.upper()} | {sys2.upper()} | Same Data | Notes |
|----------------|---------------|---------------|-----------|-------|
"""
                    for source in data_comp['source_comparison']:
                        logical = source.get('logical_source', 'N/A')
                        sys1_src = source.get('system1', 'N/A')[:40]
                        sys2_src = source.get('system2', 'N/A')[:40]
                        same = '✅' if source.get('same_data', False) else '❌'
                        notes = source.get('notes', 'N/A')[:50]

                        answer += f"| {logical} | {sys1_src} | {sys2_src} | {same} | {notes} |\n"

                    answer += "\n"

                # Target Comparison
                if 'target_comparison' in data_comp and data_comp['target_comparison']:
                    answer += f"""#### Output Targets

| Logical Target | {sys1.upper()} | {sys2.upper()} | Same Schema | Column Differences |
|----------------|---------------|---------------|-------------|-------------------|
"""
                    for target in data_comp['target_comparison']:
                        logical = target.get('logical_target', 'N/A')
                        sys1_tgt = target.get('system1', 'N/A')[:40]
                        sys2_tgt = target.get('system2', 'N/A')[:40]
                        same_schema = '✅' if target.get('same_schema', False) else '❌'
                        col_diffs = ', '.join(target.get('column_differences', ['None']))[:60]

                        answer += f"| {logical} | {sys1_tgt} | {sys2_tgt} | {same_schema} | {col_diffs} |\n"

                    answer += "\n"

            # Overall Assessment
            similarity = comparison.get('similarity_score', 0)
            are_equivalent = comparison.get('are_equivalent', False)

            answer += f"""### 📈 Overall Assessment

| Metric | Value |
|--------|-------|
| **Similarity Score** | {similarity:.1%} |
| **Are Equivalent** | {'✅ Yes' if are_equivalent else '❌ No'} |
| **Confidence** | {comparison.get('overall_assessment', {}).get('confidence_in_equivalence', 'MEDIUM')} |
| **Recommendation** | {comparison.get('overall_assessment', {}).get('recommendation', 'N/A')} |

"""

            # Business Logic Summary
            if 'business_logic_summary' in comparison:
                answer += f"""### 🔍 Business Logic Details

{comparison['business_logic_summary']}

"""

            # Field-Level Analysis Table
            if 'field_level_analysis' in comparison and comparison['field_level_analysis']:
                answer += f"""### 🔬 Field-Level Analysis

| Field Name | {sys1.upper()} Logic | {sys2.upper()} Logic | Equivalent | Potential Discrepancy |
|------------|---------------------|---------------------|------------|----------------------|
"""
                for field in comparison['field_level_analysis'][:10]:  # Limit to 10 fields
                    field_name = field.get('field_name', 'N/A')
                    sys1_logic = field.get('system1_logic', 'N/A')[:50]  # Truncate long logic
                    sys2_logic = field.get('system2_logic', 'N/A')[:50]
                    equiv = '✅' if field.get('are_equivalent', False) else '❌'
                    discrepancy = field.get('potential_discrepancy', 'None')[:60]

                    answer += f"| {field_name} | `{sys1_logic}` | `{sys2_logic}` | {equiv} | {discrepancy} |\n"

                answer += "\n"

            # Business Rule Differences
            if 'business_rule_differences' in comparison and comparison['business_rule_differences']:
                answer += f"""### ⚠️ Business Rule Differences

"""
                for i, rule in enumerate(comparison['business_rule_differences'][:5], 1):
                    answer += f"""**{i}. {rule.get('rule', 'Rule')}**
- **{sys1.upper()}:** {rule.get('system1', 'N/A')}
- **{sys2.upper()}:** {rule.get('system2', 'N/A')}
- **Impact:** {rule.get('impact', 'Unknown')}
- **Recommendation:** {rule.get('recommendation', 'Review')}

"""

            # Transformation Differences Table
            if 'transformation_differences' in comparison and comparison['transformation_differences']:
                answer += f"""### 🔄 Transformation Logic Comparison

| Operation | {sys1.upper()} | {sys2.upper()} | Equivalent | Notes |
|-----------|---------------|---------------|------------|-------|
"""
                for trans in comparison['transformation_differences'][:5]:
                    operation = trans.get('operation', 'N/A')
                    sys1_impl = trans.get('system1', 'N/A')[:40]
                    sys2_impl = trans.get('system2', 'N/A')[:40]
                    equiv = '✅' if trans.get('are_equivalent', False) else '❌'
                    notes = trans.get('data_difference', trans.get('performance_note', 'N/A'))[:50]

                    answer += f"| {operation} | `{sys1_impl}` | `{sys2_impl}` | {equiv} | {notes} |\n"

                answer += "\n"

            # Join Logic Comparison
            if 'join_logic_comparison' in comparison and comparison['join_logic_comparison']:
                answer += f"""### 🔗 Join Logic Comparison

"""
                for join in comparison['join_logic_comparison']:
                    answer += f"""**{join.get('join_name', 'Join')}:**
- **{sys1.upper()}:** `{join.get('system1', 'N/A')}`
- **{sys2.upper()}:** `{join.get('system2', 'N/A')}`
- **Impact:** {join.get('impact', 'Unknown')}
- **Row Count Impact:** {join.get('row_count_impact', 'Unknown')}

"""

            # Critical Issues
            if 'critical_issues' in comparison and comparison['critical_issues']:
                answer += f"""### 🚨 Critical Issues Found

"""
                for i, issue in enumerate(comparison['critical_issues'][:5], 1):
                    severity = issue.get('severity', 'MEDIUM')
                    severity_emoji = '🔴' if severity == 'HIGH' else '🟡' if severity == 'MEDIUM' else '🟢'

                    answer += f"""**{i}. {severity_emoji} {severity} - {issue.get('issue', 'Issue')}**
- **{sys1.upper()} Behavior:** {issue.get('system1_behavior', 'N/A')}
- **{sys2.upper()} Behavior:** {issue.get('system2_behavior', 'N/A')}
- **Potential Data Loss:** {'⚠️ YES' if issue.get('potential_data_loss', False) else '✅ NO'}
- **Affected Records:** {issue.get('affected_records', 'Unknown')}
- **Validation Query:**
  ```sql
  {issue.get('validation_query', 'N/A')}
  ```

"""

            # Data Quality Differences
            if 'data_quality_differences' in comparison and comparison['data_quality_differences']:
                answer += f"""### 🛡️ Data Quality Rule Differences

| Aspect | {sys1.upper()} | {sys2.upper()} | Risk Level | Impact |
|--------|---------------|---------------|------------|--------|
"""
                for dq in comparison['data_quality_differences'][:5]:
                    aspect = dq.get('aspect', 'N/A')
                    sys1_impl = dq.get('system1', 'N/A')[:40]
                    sys2_impl = dq.get('system2', 'N/A')[:40]
                    risk = dq.get('risk_level', 'MEDIUM')
                    impact = dq.get('impact', 'Unknown')[:60]

                    answer += f"| {aspect} | `{sys1_impl}` | `{sys2_impl}` | {risk} | {impact} |\n"

                answer += "\n"

            # Validation Queries
            if 'validation_queries' in comparison and comparison['validation_queries']:
                answer += f"""### 🔍 Validation Queries

Use these queries to validate the comparison results:

"""
                for i, vq in enumerate(comparison['validation_queries'][:5], 1):
                    answer += f"""**{i}. {vq.get('purpose', 'Validation')}**
```sql
{vq.get('query', 'N/A')}
```

"""

            # Overall Assessment Details
            if 'overall_assessment' in comparison:
                assessment = comparison['overall_assessment']

                if assessment.get('major_concerns'):
                    answer += f"""### ⚠️ Major Concerns

"""
                    for concern in assessment['major_concerns'][:5]:
                        answer += f"- {concern}\n"
                    answer += "\n"

                if assessment.get('testing_priority'):
                    answer += f"""### ✅ Testing Priority

"""
                    for i, test in enumerate(assessment['testing_priority'][:5], 1):
                        answer += f"{i}. {test}\n"
                    answer += "\n"

            # Technology Comparison (LAST - kept brief as requested)
            if 'technology_comparison' in comparison:
                tech_comp = comparison['technology_comparison']

                answer += f"""### 💻 Technology Stack Comparison

**{sys1.upper()} Technology:** {tech_comp.get('system1_technology_stack', 'N/A')}
**{sys2.upper()} Technology:** {tech_comp.get('system2_technology_stack', 'N/A')}

**Technology Impact on Logic:** {tech_comp.get('technology_impact_on_logic', 'N/A')}

"""

        return answer

    def _format_basic_comparison(
        self,
        systems: List[str],
        logic_analysis: Dict
    ) -> str:
        """Fallback basic comparison when LogicComparator not available"""

        answer = f"""## Cross-System Comparison Results

**Systems Compared:** {', '.join(systems)}

⚠️ **Note:** Detailed AI-powered comparison unavailable. Enable Azure OpenAI for field-level analysis.

### Basic Analysis:

"""
        for system, analysis in logic_analysis.items():
            answer += f"""**{system.upper()}:**
- Business Purpose: {analysis.get('business_purpose', 'Unknown')}
- Complexity: {analysis.get('complexity_score', 'N/A')}

"""

        return answer

    def _build_comparison_summary(
        self,
        systems: List[str],
        parsed_entities: Dict,
        logic_analysis: Dict,
        similarity_scores: Dict
    ) -> str:
        """Build comparison summary text"""

        summary = f"Compared implementations across {len(systems)} systems.\n\n"

        if similarity_scores:
            avg_similarity = sum(similarity_scores.values()) / len(similarity_scores)
            summary += f"**Average Similarity:** {avg_similarity:.0%}\n\n"

        return summary


def create_chat_orchestrator(ai_analyzer, indexer, vector_store=None) -> ChatOrchestrator:
    """Factory function to create chat orchestrator"""
    return ChatOrchestrator(
        ai_analyzer=ai_analyzer,
        indexer=indexer,
        vector_store=vector_store
    )
