"""
Query Classifier for Intent Detection
======================================

Analyzes user queries to determine intent and route to appropriate agents.

Intent Types:
- SIMPLE_RAG: Direct question answerable from vector search
- COMPARISON: Comparing multiple systems/entities
- LINEAGE: Column-level or data flow questions
- LOGIC_ANALYSIS: Understanding transformation logic
- CODE_SEARCH: Finding specific code patterns
- MULTI_STEP: Complex queries requiring multiple agents
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
from loguru import logger


class QueryIntent(Enum):
    """Query intent types"""
    SIMPLE_RAG = "simple_rag"
    COMPARISON = "comparison"
    LINEAGE = "lineage"
    LOGIC_ANALYSIS = "logic_analysis"
    CODE_SEARCH = "code_search"
    MULTI_STEP = "multi_step"
    DOCUMENT_GENERATION = "document_generation"  # NEW: Generate docs, STTM, comparison sheets
    WORKFLOW_MAPPING = "workflow_mapping"  # NEW: Workflow replacement/migration queries
    UNKNOWN = "unknown"


@dataclass
class ClassifiedQuery:
    """Result of query classification"""
    intent: QueryIntent
    confidence: float
    entities: List[str]  # Entities mentioned (graphs, tables, fields)
    systems: List[str]  # Systems mentioned (abinitio, hadoop, databricks)
    keywords: List[str]  # Important keywords
    reasoning: str  # Why this intent was chosen
    suggested_agents: List[str]  # Which agents to use
    task_decomposition: List[str]  # Suggested task breakdown


class QueryClassifier:
    """
    Classifies user queries to determine intent and routing strategy
    """

    def __init__(self, ai_analyzer=None):
        """
        Initialize query classifier

        Args:
            ai_analyzer: Optional AI analyzer for semantic understanding
        """
        self.ai_analyzer = ai_analyzer

        # Intent detection patterns
        self.patterns = {
            QueryIntent.COMPARISON: [
                r'\b(compare|difference|versus|vs|similar|equivalent|match)\b',
                r'\b(both|between|across)\b.*\b(system|graph|workflow|pipeline)\b',
                r'\b(hadoop|databricks|abinitio).*\b(and|vs|versus)\b.*\b(hadoop|databricks|abinitio)\b',
            ],
            QueryIntent.LINEAGE: [
                r'\b(lineage|trace|track|flow|source|target|derive|origin)\b',
                r'\b(column|field).*\b(come from|source|derived|calculated)\b',
                r'\b(upstream|downstream|dependency|dependent)\b',
                r'\b(STTM|mapping|source.*target)\b',
            ],
            QueryIntent.LOGIC_ANALYSIS: [
                r'\b(logic|transformation|calculate|compute|process)\b',
                r'\b(how does|what does|explain|understand)\b.*\b(work|process|transform)\b',
                r'\b(business rule|rule|validation|check)\b',
                r'\b(aggregate|join|filter|group)\b',
            ],
            QueryIntent.CODE_SEARCH: [
                r'\b(find|search|locate|show me|where is)\b.*\b(code|function|component|graph)\b',
                r'\b(contains|uses|calls|references)\b',
                r'\b(all|list).*\b(graph|workflow|component|table)\b',
            ],
            QueryIntent.DOCUMENT_GENERATION: [
                r'\b(generate|create|export|produce|build)\b.*\b(document|doc|report|sheet|excel|sttm|mapping)\b',
                r'\b(generate|create).*\b(comparison|sttm|mapping)\b',
                r'\b(document|export)\b.*\b(for|about)\b.*\b(flow|workflow|pipeline)\b',
            ],
            QueryIntent.WORKFLOW_MAPPING: [
                r'\b(replaced|migration|unmapped|gap|mapping)\b.*\b(workflow|pipeline|flow)\b',
                r'\b(what.*replaced|which.*replaced)\b',
                r'\b(n:1|1:n|n-1|1-n|consolidation|split|merged|combined)\b',
                r'\b(show|list|find).*\b(mapping|unmapped)\b',
            ],
        }

        # System detection patterns
        self.system_patterns = {
            'abinitio': r'\b(ab ?initio|\.mp|\.dml|graph|component)\b',
            'hadoop': r'\b(hadoop|hive|pig|\.hql|\.pig|mapreduce)\b',
            'databricks': r'\b(databricks|spark|pyspark|notebook|\.py|\.ipynb)\b',
        }

        # Comparison keywords
        self.comparison_keywords = [
            'compare', 'difference', 'versus', 'vs', 'similar', 'equivalent',
            'match', 'both', 'between', 'across', 'different', 'same'
        ]

        # Lineage keywords
        self.lineage_keywords = [
            'lineage', 'trace', 'track', 'flow', 'source', 'target', 'derive',
            'origin', 'upstream', 'downstream', 'dependency', 'column', 'field'
        ]

    def classify(self, query: str, context: Optional[Dict] = None) -> ClassifiedQuery:
        """
        Classify user query to determine intent and routing

        Args:
            query: User's question
            context: Optional context (previous conversation, selected entities, etc.)

        Returns:
            ClassifiedQuery with intent, entities, and suggested approach
        """
        query_lower = query.lower()

        # Extract entities and systems
        entities = self._extract_entities(query)
        systems = self._extract_systems(query)
        keywords = self._extract_keywords(query)

        # Score each intent
        intent_scores = {}
        for intent, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score += 1
            intent_scores[intent] = score

        # Determine primary intent
        intent, confidence, reasoning = self._determine_intent(
            query_lower, intent_scores, entities, systems, keywords
        )

        # Suggest agents based on intent
        suggested_agents = self._suggest_agents(intent, systems, entities)

        # Create task decomposition
        task_decomposition = self._create_task_plan(intent, query, entities, systems)

        return ClassifiedQuery(
            intent=intent,
            confidence=confidence,
            entities=entities,
            systems=systems,
            keywords=keywords,
            reasoning=reasoning,
            suggested_agents=suggested_agents,
            task_decomposition=task_decomposition
        )

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entity names (graphs, tables, fields) from query"""
        entities = []

        # Pattern for quoted entities
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        entities.extend(quoted)

        # Pattern for capitalized entities or entities with underscores
        camel_or_snake = re.findall(r'\b([A-Z][a-zA-Z0-9_]*|[a-z]+_[a-z0-9_]+)\b', query)
        entities.extend(camel_or_snake)

        # Pattern for common entity keywords
        entity_patterns = [
            r'\bgraph\s+(\w+)',
            r'\btable\s+(\w+)',
            r'\bfield\s+(\w+)',
            r'\bcolumn\s+(\w+)',
            r'\bcomponent\s+(\w+)',
            r'\bworkflow\s+(\w+)',
        ]

        for pattern in entity_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)

        return list(set(entities))

    def _extract_systems(self, query: str) -> List[str]:
        """Extract system types mentioned in query"""
        systems = []

        for system, pattern in self.system_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                systems.append(system)

        return systems

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        keywords = []

        # Check for comparison keywords
        for kw in self.comparison_keywords:
            if kw in query.lower():
                keywords.append(kw)

        # Check for lineage keywords
        for kw in self.lineage_keywords:
            if kw in query.lower():
                keywords.append(kw)

        return keywords

    def _determine_intent(
        self,
        query_lower: str,
        intent_scores: Dict,
        entities: List[str],
        systems: List[str],
        keywords: List[str]
    ) -> tuple[QueryIntent, float, str]:
        """Determine primary intent with confidence and reasoning"""

        # Check for document generation (HIGH PRIORITY)
        if any(kw in query_lower for kw in ['generate document', 'generate sttm', 'create document', 'export', 'generate report']):
            return (
                QueryIntent.DOCUMENT_GENERATION,
                0.98,
                "Detected document generation request - will generate docs/STTM"
            )

        # Check for workflow mapping (HIGH PRIORITY)
        if any(kw in query_lower for kw in ['replaced', 'what replaced', 'n:1', '1:n', 'unmapped', 'migration', 'consolidation']):
            return (
                QueryIntent.WORKFLOW_MAPPING,
                0.96,
                "Detected workflow mapping query - will use intelligent workflow system"
            )

        # Check for multi-system comparison
        if len(systems) >= 2 and any(kw in query_lower for kw in self.comparison_keywords):
            return (
                QueryIntent.COMPARISON,
                0.95,
                f"Detected comparison query with {len(systems)} systems: {', '.join(systems)}"
            )

        # Check for lineage query
        if any(kw in query_lower for kw in ['lineage', 'trace', 'source', 'target', 'derive']):
            return (
                QueryIntent.LINEAGE,
                0.90,
                f"Detected lineage/tracing query with keywords: {', '.join(keywords)}"
            )

        # Check for logic analysis
        if any(kw in query_lower for kw in ['how does', 'explain', 'logic', 'transformation']):
            return (
                QueryIntent.LOGIC_ANALYSIS,
                0.85,
                "Detected logic analysis request - requires AI reasoning"
            )

        # Check for code search
        if any(kw in query_lower for kw in ['find', 'search', 'locate', 'show me', 'list all']):
            return (
                QueryIntent.CODE_SEARCH,
                0.80,
                "Detected code search request - requires vector search"
            )

        # Get highest scoring intent
        if intent_scores:
            max_intent = max(intent_scores.items(), key=lambda x: x[1])
            if max_intent[1] > 0:
                return (
                    max_intent[0],
                    0.70,
                    f"Matched {max_intent[1]} patterns for {max_intent[0].value}"
                )

        # Default to simple RAG
        return (
            QueryIntent.SIMPLE_RAG,
            0.60,
            "No specific patterns matched - using simple RAG search"
        )

    def _suggest_agents(self, intent: QueryIntent, systems: List[str], entities: List[str]) -> List[str]:
        """Suggest which agents to use based on intent"""
        agents = []

        if intent == QueryIntent.SIMPLE_RAG:
            agents = ["RAGAgent"]

        elif intent == QueryIntent.COMPARISON:
            agents = ["ParsingAgent", "SimilarityAgent", "ComparisonAgent"]

        elif intent == QueryIntent.LINEAGE:
            agents = ["ParsingAgent", "MappingAgent", "LineageAgent"]

        elif intent == QueryIntent.LOGIC_ANALYSIS:
            agents = ["ParsingAgent", "LogicAgent", "RAGAgent"]

        elif intent == QueryIntent.CODE_SEARCH:
            agents = ["RAGAgent", "ParsingAgent"]

        elif intent == QueryIntent.DOCUMENT_GENERATION:
            agents = ["DocumentGenerationAgent", "STTMGenerator", "ParsingAgent"]

        elif intent == QueryIntent.WORKFLOW_MAPPING:
            agents = ["WorkflowMappingAgent", "MigrationValidator"]

        elif intent == QueryIntent.MULTI_STEP:
            agents = ["RAGAgent", "ParsingAgent", "LogicAgent", "LineageAgent"]

        return agents

    def _create_task_plan(
        self,
        intent: QueryIntent,
        query: str,
        entities: List[str],
        systems: List[str]
    ) -> List[str]:
        """Create task decomposition based on intent"""
        tasks = []

        if intent == QueryIntent.SIMPLE_RAG:
            tasks = [
                f"Search vector database for: '{query[:50]}...'",
                "Retrieve top 5 relevant chunks",
                "Generate answer from context"
            ]

        elif intent == QueryIntent.COMPARISON:
            system_list = ', '.join(systems) if systems else "detected systems"
            tasks = [
                f"Parse entities from {system_list}",
                f"Extract transformations and logic",
                f"Compare implementations using similarity scoring",
                f"Identify differences and similarities",
                f"Generate comparison report"
            ]

        elif intent == QueryIntent.LINEAGE:
            entity_list = ', '.join(entities[:3]) if entities else "specified entity"
            tasks = [
                f"Parse entity: {entity_list}",
                f"Extract column-level mappings (STTM)",
                f"Build dependency chains",
                f"Trace lineage across transformations",
                f"Generate lineage visualization data"
            ]

        elif intent == QueryIntent.LOGIC_ANALYSIS:
            entity_list = ', '.join(entities[:2]) if entities else "target entity"
            tasks = [
                f"Retrieve code for: {entity_list}",
                f"Extract transformation logic",
                f"Use AI to interpret business rules",
                f"Explain logic in natural language",
                f"Document findings"
            ]

        elif intent == QueryIntent.CODE_SEARCH:
            tasks = [
                f"Search codebase for pattern",
                f"Filter by systems: {', '.join(systems) if systems else 'all'}",
                f"Rank results by relevance",
                f"Extract code snippets",
                f"Present findings"
            ]

        elif intent == QueryIntent.DOCUMENT_GENERATION:
            entity_list = ', '.join(entities[:2]) if entities else "specified entity"
            tasks = [
                f"Find workflow/pipeline: {entity_list}",
                f"Extract complete metadata and transformations",
                f"Generate STTM mappings if needed",
                f"Create document (Excel/JSON/Markdown)",
                f"Provide download link"
            ]

        elif intent == QueryIntent.WORKFLOW_MAPPING:
            entity_list = ', '.join(entities[:2]) if entities else "workflows"
            tasks = [
                f"Load workflow intelligence data",
                f"Search for workflow mappings: {entity_list}",
                f"Calculate similarities and detect patterns (N:1, 1:N)",
                f"Generate answer with mapping details",
                f"Provide actionable recommendations"
            ]

        return tasks


def create_query_classifier(ai_analyzer=None) -> QueryClassifier:
    """Factory function to create query classifier"""
    return QueryClassifier(ai_analyzer=ai_analyzer)
