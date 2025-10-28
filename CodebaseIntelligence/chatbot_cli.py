#!/usr/bin/env python3
"""
RAG Chatbot CLI - Interactive command-line interface
Ask questions about your codebase using FREE local search + optional OpenAI
"""

import sys
import os
from pathlib import Path
from loguru import logger

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from services.rag_chatbot_integrated import CodebaseRAGChatbot
from services.codebase_indexer import CodebaseIndexer


def print_banner():
    """Print welcome banner"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║           Codebase Intelligence RAG Chatbot               ║
║                                                           ║
║  Ask questions about your Ab Initio, Hadoop, Databricks  ║
║                      codebases!                           ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

"""
    print(banner)


def print_help():
    """Print help message"""
    help_text = """
Available Commands:
------------------
  ask <question>       - Ask a question about the codebase
  process <name>       - Get info about a specific process
  sttm <table>         - Find STTM mappings for a table
  gaps                 - Find gaps in the codebase
  stats                - Show chatbot statistics
  clear                - Clear conversation history
  help                 - Show this help message
  exit/quit            - Exit the chatbot

Examples:
---------
  ask What does the patient matching process do?
  process 400_commGenIpa
  sttm patientacctspayercob
  gaps
  stats

Tips:
-----
  - Use FREE local search (ChromaDB) - no API keys needed!
  - Configure Azure OpenAI API key for AI-powered answers
  - Run indexing first to populate the vector database
"""
    print(help_text)


def setup_chatbot() -> CodebaseRAGChatbot:
    """Initialize chatbot"""
    print("\n🔧 Initializing chatbot...")

    # Check for OpenAI API key
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if api_key:
        print("✓ Azure OpenAI API key found - RAG mode enabled")
    else:
        print("⚠ No OpenAI API key found - running in search-only mode")
        print("  Set AZURE_OPENAI_API_KEY environment variable for AI-powered answers")

    # Initialize chatbot
    chatbot = CodebaseRAGChatbot(
        use_local_search=True,
        vector_db_path="./outputs/vector_db",
        openai_api_key=api_key,
    )

    # Check if vector database has data
    stats = chatbot.get_stats()
    db_stats = stats.get("vector_db", {})
    doc_count = db_stats.get("document_count", 0)

    if doc_count == 0:
        print("\n⚠ WARNING: Vector database is empty!")
        print("  You need to index your codebase first.")
        print("  Run: python3 index_codebase.py")
        print("  Or use run_analysis.py with --mode full")
        print()
    else:
        print(f"✓ Vector database loaded: {doc_count} documents indexed")

    print("\nChatbot ready! Type 'help' for commands or 'exit' to quit.\n")

    return chatbot


def handle_command(chatbot: CodebaseRAGChatbot, command: str) -> bool:
    """
    Handle user command

    Returns:
        True to continue, False to exit
    """
    command = command.strip()

    if not command:
        return True

    # Check for exit commands
    if command.lower() in ["exit", "quit", "q"]:
        print("\n👋 Goodbye!\n")
        return False

    # Help command
    if command.lower() == "help":
        print_help()
        return True

    # Clear history
    if command.lower() == "clear":
        chatbot.clear_history()
        print("✓ Conversation history cleared\n")
        return True

    # Stats command
    if command.lower() == "stats":
        stats = chatbot.get_stats()
        print("\n📊 Chatbot Statistics:")
        print(f"  Mode: {stats.get('mode', 'unknown')}")
        print(f"  OpenAI Configured: {stats.get('openai_configured', False)}")
        print(f"  Conversation Length: {stats.get('conversation_length', 0)} exchanges")

        db_stats = stats.get("vector_db", {})
        print(f"\n  Vector Database:")
        print(f"    Documents: {db_stats.get('document_count', 0)}")
        print(f"    Collections: {db_stats.get('collection_count', 0)}")
        print()
        return True

    # Parse command
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    # Process-specific query
    if cmd == "process":
        if not args:
            print("Usage: process <process_name>\n")
            return True

        print(f"\n🔍 Looking up process: {args}...")
        from services.rag_chatbot_integrated import ask_about_process
        result = ask_about_process(chatbot, args)
        print_result(result)
        return True

    # STTM query
    if cmd == "sttm":
        if not args:
            print("Usage: sttm <table_name>\n")
            return True

        print(f"\n🔍 Finding STTM mappings for: {args}...")
        from services.rag_chatbot_integrated import find_sttm
        result = find_sttm(chatbot, source_table=args)
        print_result(result)
        return True

    # Gaps query
    if cmd == "gaps":
        print("\n🔍 Finding gaps...")
        from services.rag_chatbot_integrated import find_gaps
        result = find_gaps(chatbot)
        print_result(result)
        return True

    # Ask question
    if cmd == "ask":
        if not args:
            print("Usage: ask <your question>\n")
            return True

        question = args
    else:
        # Treat entire command as question
        question = command

    # Query chatbot
    print(f"\n🤔 Thinking...\n")
    result = chatbot.chat(question)
    print_result(result)

    return True


def print_result(result: dict):
    """Print chatbot result"""
    print("=" * 70)
    print("📝 Answer:")
    print("=" * 70)
    print(result.get("answer", "No answer generated"))
    print()

    # Show sources
    sources = result.get("sources", [])
    if sources:
        print("=" * 70)
        print(f"📚 Sources ({len(sources)} results):")
        print("=" * 70)

        for i, source in enumerate(sources[:3], 1):  # Show top 3
            print(f"\n[{i}] {source.get('content', '')[:150]}...")
            metadata = source.get("metadata", {})
            print(f"    Type: {metadata.get('doc_type', 'N/A')}")
            print(f"    System: {metadata.get('system', 'N/A')}")
            print(f"    Score: {source.get('score', 0):.3f}")

        if len(sources) > 3:
            print(f"\n    ... and {len(sources) - 3} more results")
    else:
        print("⚠ No sources found - vector database might be empty\n")

    # Show metadata
    print("\n" + "-" * 70)
    print(f"Mode: {result.get('mode', 'unknown')} | "
          f"Confidence: {result.get('confidence', 'unknown')} | "
          f"Results: {result.get('total_results', 0)}")
    print("=" * 70 + "\n")


def interactive_mode(chatbot: CodebaseRAGChatbot):
    """Run chatbot in interactive mode"""
    print("Interactive mode - type your questions below:\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Handle command
            should_continue = handle_command(chatbot, user_input)

            if not should_continue:
                break

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
        except EOFError:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ Error: {e}\n")


def main():
    """Main function"""
    # Print banner
    print_banner()

    # Setup chatbot
    try:
        chatbot = setup_chatbot()
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        print(f"\n❌ Failed to initialize chatbot: {e}")
        print("\nPlease check:")
        print("  1. Vector database exists: ./outputs/vector_db/")
        print("  2. Dependencies installed: pip install -r requirements.txt")
        print("  3. Run indexing first: python3 index_codebase.py")
        sys.exit(1)

    # Check command line arguments
    if len(sys.argv) > 1:
        # Single question mode
        question = " ".join(sys.argv[1:])
        print(f"Question: {question}\n")
        result = chatbot.query(question)
        print_result(result)
    else:
        # Interactive mode
        interactive_mode(chatbot)


if __name__ == "__main__":
    main()
