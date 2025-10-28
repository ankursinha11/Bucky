"""
RAG Chatbot using LangChain and Azure OpenAI
Enables natural language queries over parsed codebase
"""

import os
from typing import List, Dict, Any, Optional
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from loguru import logger

from services.azure_search.search_client import CodebaseSearchClient


class CodebaseRAGChatbot:
    """RAG-based chatbot for codebase queries"""

    def __init__(self, search_client: Optional[CodebaseSearchClient] = None):
        self.search_client = search_client or CodebaseSearchClient()

        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
            temperature=0.1,
            max_tokens=2000,
        )

        # Conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        # System prompt
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for chatbot"""
        return """You are an expert code analyst for healthcare finance data pipelines.
You have access to parsed code from Ab Initio, Hadoop, and Azure Databricks systems.

Your capabilities:
- Explain what code components do
- Show source-to-target mappings (STTM)
- Identify gaps between systems
- Trace data lineage
- Compare logic across systems
- Provide recommendations

Guidelines:
- Always cite the specific system, process, and component you're referencing
- Use technical but clear language
- When showing STTM, include source→target column mappings
- When identifying gaps, explain the business impact
- Provide code snippets when relevant
- If you don't know, say so and suggest where to look

Current systems available: Ab Initio, Hadoop, Databricks
Focus areas: Lead generation, patient matching, coverage discovery, CDD processes
"""

    def query(
        self,
        question: str,
        context_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the codebase

        Args:
            question: User's question
            context_filters: Optional filters (doc_types, systems)

        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info(f"Query: {question}")

        # Search for relevant context
        doc_types = context_filters.get("doc_types") if context_filters else None
        systems = context_filters.get("systems") if context_filters else None

        search_results = self.search_client.hybrid_search(
            query=question,
            doc_types=doc_types,
            systems=systems,
            top=5,
        )

        if not search_results:
            return {
                "answer": "I couldn't find relevant information in the codebase to answer your question.",
                "sources": [],
                "confidence": "low",
            }

        # Build context from search results
        context = self._build_context(search_results)

        # Generate answer
        answer = self._generate_answer(question, context)

        return {
            "answer": answer,
            "sources": [
                {
                    "id": r.get("id"),
                    "type": r.get("doc_type"),
                    "name": r.get("name"),
                    "system": r.get("system"),
                }
                for r in search_results
            ],
            "confidence": "high" if len(search_results) >= 3 else "medium",
        }

    def _build_context(self, search_results: List[Dict]) -> str:
        """Build context string from search results"""
        context_parts = []

        for i, result in enumerate(search_results, 1):
            doc_type = result.get("doc_type", "unknown")
            name = result.get("name", "unknown")
            system = result.get("system", "unknown")
            content = result.get("content", "")
            description = result.get("description", "")

            context_part = f"""
[Source {i}] Type: {doc_type} | System: {system} | Name: {name}
Description: {description}
Content: {content[:500]}
---
"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM"""
        prompt = f"""{self.system_prompt}

Context from codebase:
{context}

User Question: {question}

Please provide a detailed answer based on the context above. Include specific references to processes, components, and systems mentioned in the context.

Answer:"""

        try:
            response = self.llm.predict(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"

    def chat(self, message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Multi-turn conversation

        Args:
            message: User message
            conversation_id: Optional conversation ID for tracking

        Returns:
            Dict with response and conversation metadata
        """
        # For now, treat each message independently
        # In production, would track conversation history
        return self.query(message)

    def ask_about_process(self, process_name: str, system: Optional[str] = None) -> Dict[str, Any]:
        """Ask about a specific process"""
        question = f"What does the process '{process_name}' do?"

        filters = {"doc_types": ["process", "component"]}
        if system:
            filters["systems"] = [system]

        return self.query(question, context_filters=filters)

    def find_sttm(
        self,
        source_table: Optional[str] = None,
        target_table: Optional[str] = None,
        column_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find source-to-target mappings"""
        question_parts = ["Show me source-to-target mappings"]

        if source_table:
            question_parts.append(f"from source table {source_table}")
        if target_table:
            question_parts.append(f"to target table {target_table}")
        if column_name:
            question_parts.append(f"for column {column_name}")

        question = " ".join(question_parts)

        return self.query(question, context_filters={"doc_types": ["sttm"]})

    def find_gaps(
        self,
        source_system: Optional[str] = None,
        target_system: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find gaps between systems"""
        question_parts = ["What gaps exist"]

        if source_system and target_system:
            question_parts.append(f"between {source_system} and {target_system}")
        elif source_system:
            question_parts.append(f"in {source_system}")

        if severity:
            question_parts.append(f"with {severity} severity")

        question = " ".join(question_parts) + "?"

        return self.query(question, context_filters={"doc_types": ["gap"]})

    def compare_processes(
        self,
        process1: str,
        system1: str,
        process2: str,
        system2: str,
    ) -> Dict[str, Any]:
        """Compare two processes from different systems"""
        question = f"Compare the process '{process1}' in {system1} with '{process2}' in {system2}. What are the differences?"

        # Search for both processes
        results1 = self.search_client.hybrid_search(
            query=process1,
            systems=[system1],
            doc_types=["process", "component"],
            top=3,
        )

        results2 = self.search_client.hybrid_search(
            query=process2,
            systems=[system2],
            doc_types=["process", "component"],
            top=3,
        )

        # Combine contexts
        combined_context = self._build_context(results1 + results2)

        answer = self._generate_answer(question, combined_context)

        return {
            "answer": answer,
            "sources": [
                {
                    "id": r.get("id"),
                    "type": r.get("doc_type"),
                    "name": r.get("name"),
                    "system": r.get("system"),
                }
                for r in results1 + results2
            ],
            "confidence": "high",
        }

    def explain_gap(self, gap_id: str) -> Dict[str, Any]:
        """Explain a specific gap in detail"""
        # Search for gap by ID
        results = self.search_client.search(
            query=gap_id,
            filters=f"id eq '{gap_id}'",
            top=1,
        )

        if not results:
            return {
                "answer": f"Gap {gap_id} not found in the database.",
                "sources": [],
                "confidence": "low",
            }

        gap_info = results[0]
        question = f"Explain this gap in detail: {gap_info.get('name')}. What is the impact and how can it be resolved?"

        context = self._build_context(results)
        answer = self._generate_answer(question, context)

        return {
            "answer": answer,
            "sources": [
                {
                    "id": gap_info.get("id"),
                    "type": "gap",
                    "name": gap_info.get("name"),
                }
            ],
            "confidence": "high",
        }
