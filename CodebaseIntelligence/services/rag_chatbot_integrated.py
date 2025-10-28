"""
RAG Chatbot - Integrated with FREE Local Search
Uses ChromaDB + sentence-transformers + Azure OpenAI (optional)
"""

import os
from typing import List, Dict, Any, Optional
from loguru import logger

# Try to import OpenAI, but make it optional
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not installed. Chatbot will work in search-only mode.")

from services.local_search.local_search_client import LocalSearchClient


class CodebaseRAGChatbot:
    """
    RAG-based chatbot for codebase queries
    Works with FREE local search (ChromaDB) and optional Azure OpenAI
    """

    def __init__(
        self,
        use_local_search: bool = True,
        vector_db_path: str = "./outputs/vector_db",
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize chatbot

        Args:
            use_local_search: Use local ChromaDB (default True)
            vector_db_path: Path to ChromaDB database
            openai_api_key: Optional Azure OpenAI API key
        """
        self.use_local_search = use_local_search

        # Initialize search client (FREE local search!)
        logger.info("Initializing FREE local search with ChromaDB...")
        self.search_client = LocalSearchClient(persist_directory=vector_db_path)

        # Initialize OpenAI if available and key provided
        self.llm = None
        self.openai_available = OPENAI_AVAILABLE

        if OPENAI_AVAILABLE and (openai_api_key or os.getenv("AZURE_OPENAI_API_KEY")):
            try:
                self.llm = AzureOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=openai_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                )
                self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
                logger.info("✓ Azure OpenAI initialized")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI: {e}. Using search-only mode.")
                self.llm = None
        else:
            logger.info("Running in search-only mode (no OpenAI)")

        # System prompt for LLM
        self.system_prompt = self._create_system_prompt()

        # Conversation history (simple in-memory for now)
        self.conversation_history: List[Dict[str, str]] = []

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

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents into vector database

        Args:
            documents: List of dicts with 'content', 'metadata' keys
        """
        logger.info(f"Indexing {len(documents)} documents into vector database...")

        try:
            self.search_client.index_documents(documents)
            logger.info(f"✓ Successfully indexed {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise

    def query(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the codebase

        Args:
            question: User's question
            top_k: Number of search results to retrieve
            filters: Optional filters for search

        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info(f"Query: {question}")

        # Search for relevant context
        try:
            search_results = self.search_client.search(
                query=question,
                top=top_k,
                filters=filters,
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                "answer": f"Search error: {str(e)}",
                "sources": [],
                "confidence": "low",
                "mode": "error",
            }

        if not search_results or "results" not in search_results:
            return {
                "answer": "I couldn't find relevant information in the codebase to answer your question. The vector database might be empty. Please index documents first.",
                "sources": [],
                "confidence": "low",
                "mode": "search_only",
            }

        results_list = search_results["results"]

        if not results_list:
            return {
                "answer": "No relevant documents found. Try rephrasing your question or check if the codebase has been indexed.",
                "sources": [],
                "confidence": "low",
                "mode": "search_only",
            }

        # If OpenAI is available, generate answer using LLM
        if self.llm:
            context = self._build_context(results_list)
            answer = self._generate_answer_with_llm(question, context)
            mode = "rag"
        else:
            # Search-only mode: return formatted search results
            answer = self._format_search_results(results_list)
            mode = "search_only"

        # Extract sources
        sources = []
        for result in results_list:
            metadata = result.get("metadata", {})
            sources.append({
                "id": result.get("id", ""),
                "content": result.get("content", "")[:200] + "...",
                "score": result.get("score", 0),
                "metadata": metadata,
            })

        return {
            "answer": answer,
            "sources": sources,
            "confidence": "high" if len(results_list) >= 3 else "medium",
            "mode": mode,
            "total_results": len(results_list),
        }

    def _build_context(self, search_results: List[Dict]) -> str:
        """Build context string from search results"""
        context_parts = []

        for i, result in enumerate(search_results, 1):
            content = result.get("content", "")
            metadata = result.get("metadata", {})

            doc_type = metadata.get("doc_type", "unknown")
            name = metadata.get("name", "unknown")
            system = metadata.get("system", "unknown")

            context_part = f"""
[Source {i}] Type: {doc_type} | System: {system} | Name: {name}
Content: {content[:500]}
---
"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _generate_answer_with_llm(self, question: str, context: str) -> str:
        """Generate answer using Azure OpenAI"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""Context from codebase:
{context}

User Question: {question}

Please provide a detailed answer based on the context above. Include specific references to processes, components, and systems mentioned in the context."""},
        ]

        try:
            response = self.llm.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}")
            return f"Error generating answer: {str(e)}\n\nFalling back to search results:\n{self._format_search_results([{'content': context}])}"

    def _format_search_results(self, results: List[Dict]) -> str:
        """Format search results for display (when no LLM available)"""
        output_parts = [
            "**Search Results (No LLM configured - showing raw results):**\n"
        ]

        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            score = result.get("score", 0)

            doc_type = metadata.get("doc_type", "unknown")
            name = metadata.get("name", "unknown")
            system = metadata.get("system", "unknown")

            output_parts.append(f"""
**Result {i}** (Score: {score:.3f})
- Type: {doc_type}
- System: {system}
- Name: {name}
- Content: {content[:300]}...

""")

        output_parts.append("\n**Tip:** Configure Azure OpenAI API key for AI-powered answers!")

        return "\n".join(output_parts)

    def chat(self, message: str) -> Dict[str, Any]:
        """
        Multi-turn conversation

        Args:
            message: User message

        Returns:
            Dict with response and conversation metadata
        """
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": message})

        # Get response
        response = self.query(message)

        # Add to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response["answer"]
        })

        # Keep only last 10 exchanges
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return response

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        try:
            search_stats = self.search_client.get_stats()

            return {
                "vector_db": search_stats,
                "openai_configured": self.llm is not None,
                "mode": "RAG" if self.llm else "search_only",
                "conversation_length": len(self.conversation_history) // 2,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "error": str(e),
                "openai_configured": self.llm is not None,
            }


# Convenience functions for quick queries

def ask_about_process(chatbot: CodebaseRAGChatbot, process_name: str, system: Optional[str] = None) -> Dict[str, Any]:
    """Ask about a specific process"""
    question = f"What does the process '{process_name}' do?"

    filters = {"doc_type": "process"} if system is None else {"doc_type": "process", "system": system}

    return chatbot.query(question, filters=filters)


def find_sttm(
    chatbot: CodebaseRAGChatbot,
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

    return chatbot.query(question, filters={"doc_type": "sttm"})


def find_gaps(
    chatbot: CodebaseRAGChatbot,
    source_system: Optional[str] = None,
    target_system: Optional[str] = None,
) -> Dict[str, Any]:
    """Find gaps between systems"""
    question_parts = ["What gaps exist"]

    if source_system and target_system:
        question_parts.append(f"between {source_system} and {target_system}")
    elif source_system:
        question_parts.append(f"in {source_system}")

    question = " ".join(question_parts) + "?"

    return chatbot.query(question, filters={"doc_type": "gap"})
