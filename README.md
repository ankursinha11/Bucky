python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', os.getenv('AZURE_OPENAI_API_KEY')[:20] + '...' if os.getenv('AZURE_OPENAI_API_KEY') else 'NOT FOUND'); print('Endpoint:', os.getenv('AZURE_OPENAI_ENDPOINT'))"


"""
Quick Verification Script - Test STAG Initialization
====================================================

Run this to verify all components initialize correctly
"""

import sys
from loguru import logger

logger.info("=" * 60)
logger.info("STAG Initialization Verification")
logger.info("=" * 60)

# Test 1: Import all required modules
logger.info("\n[Test 1] Importing modules...")
try:
    from services.multi_collection_indexer import MultiCollectionIndexer
    from services.ai_script_analyzer import AIScriptAnalyzer
    from services.chat.chat_orchestrator import create_chat_orchestrator
    from services.chat.query_classifier import QueryClassifier
    from services.lineage.lineage_agents import (
        ParsingAgent, LogicAgent, MappingAgent, SimilarityAgent, LineageAgent
    )
    logger.info("✓ All imports successful")
except Exception as e:
    logger.error(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize MultiCollectionIndexer
logger.info("\n[Test 2] Initializing MultiCollectionIndexer...")
try:
    indexer = MultiCollectionIndexer(vector_db_path="./outputs/vector_db")
    stats = indexer.get_stats()
    logger.info(f"✓ Indexer initialized with {len(stats)} collections")

    # Show collection stats
    for collection, data in stats.items():
        doc_count = data.get('total_documents', 0)
        logger.info(f"  - {collection}: {doc_count} documents")
except Exception as e:
    logger.error(f"✗ Indexer initialization failed: {e}")
    sys.exit(1)

# Test 3: Initialize AI Analyzer
logger.info("\n[Test 3] Initializing AI Analyzer...")
try:
    ai_analyzer = AIScriptAnalyzer()
    if ai_analyzer.enabled:
        logger.info("✓ AI Analyzer initialized (Azure OpenAI enabled)")
    else:
        logger.warning("⚠ AI Analyzer initialized (No Azure OpenAI - search-only mode)")
except Exception as e:
    logger.error(f"✗ AI Analyzer initialization failed: {e}")
    sys.exit(1)

# Test 4: Initialize Chat Orchestrator
logger.info("\n[Test 4] Initializing Chat Orchestrator...")
try:
    orchestrator = create_chat_orchestrator(
        ai_analyzer=ai_analyzer,
        indexer=indexer,
        vector_store=None
    )
    logger.info("✓ Chat Orchestrator initialized")
    logger.info(f"  - Agents: ParsingAgent, LogicAgent, MappingAgent, SimilarityAgent, LineageAgent")
except Exception as e:
    logger.error(f"✗ Chat Orchestrator initialization failed: {e}")
    logger.error(f"  Error details: {type(e).__name__}: {str(e)}")
    sys.exit(1)

# Test 5: Initialize Individual Agents
logger.info("\n[Test 5] Testing individual agent initialization...")
try:
    logger.info("  - Testing ParsingAgent...")
    parsing_agent = ParsingAgent(indexer=indexer, ai_analyzer=ai_analyzer)
    logger.info("    ✓ ParsingAgent OK")

    logger.info("  - Testing LogicAgent...")
    logic_agent = LogicAgent(ai_analyzer=ai_analyzer)
    logger.info("    ✓ LogicAgent OK")

    logger.info("  - Testing MappingAgent...")
    mapping_agent = MappingAgent()
    logger.info("    ✓ MappingAgent OK")

    logger.info("  - Testing SimilarityAgent...")
    similarity_agent = SimilarityAgent(indexer=indexer)
    logger.info("    ✓ SimilarityAgent OK")

    logger.info("  - Testing LineageAgent...")
    lineage_agent = LineageAgent()
    logger.info("    ✓ LineageAgent OK")

    logger.info("✓ All agents initialized successfully")
except Exception as e:
    logger.error(f"✗ Agent initialization failed: {e}")
    sys.exit(1)

# Test 6: Test Query Classification
logger.info("\n[Test 6] Testing query classification...")
try:
    classifier = QueryClassifier(ai_analyzer=ai_analyzer)

    test_queries = [
        "What parsers are available?",
        "Compare Ab Initio customer_load with Hadoop customer_etl",
        "Trace lineage of customer_id field",
        "Explain how the aggregation works"
    ]

    for query in test_queries:
        classified = classifier.classify(query)
        logger.info(f"  - '{query[:40]}...'")
        logger.info(f"    Intent: {classified.intent.value} ({classified.confidence:.0%})")

    logger.info("✓ Query classification working")
except Exception as e:
    logger.error(f"✗ Query classification failed: {e}")
    sys.exit(1)

# Test 7: Verify analyze_with_context method exists
logger.info("\n[Test 7] Verifying AIScriptAnalyzer.analyze_with_context...")
try:
    if hasattr(ai_analyzer, 'analyze_with_context'):
        logger.info("✓ analyze_with_context method exists")

        # Test with AI disabled (should return graceful message)
        result = ai_analyzer.analyze_with_context("test query", "test context")
        if isinstance(result, dict) and ('analysis' in result or 'response' in result):
            logger.info("✓ Method returns correct format")
        else:
            logger.warning("⚠ Method returns unexpected format")
    else:
        logger.error("✗ analyze_with_context method missing")
        sys.exit(1)
except Exception as e:
    logger.error(f"✗ Error testing analyze_with_context: {e}")
    sys.exit(1)

# Test 8: Verify search_multi_collection method exists
logger.info("\n[Test 8] Verifying MultiCollectionIndexer.search_multi_collection...")
try:
    if hasattr(indexer, 'search_multi_collection'):
        logger.info("✓ search_multi_collection method exists")
    else:
        logger.error("✗ search_multi_collection method missing")
        sys.exit(1)
except Exception as e:
    logger.error(f"✗ Error checking search_multi_collection: {e}")
    sys.exit(1)

# Final Summary
logger.info("\n" + "=" * 60)
logger.info("VERIFICATION COMPLETE")
logger.info("=" * 60)
logger.info("\n✅ All tests passed! Your STAG system is ready to use.\n")
logger.info("Key findings:")
if ai_analyzer.enabled:
    logger.info("  ✓ Azure OpenAI: ENABLED (AI features available)")
else:
    logger.info("  ⚠ Azure OpenAI: DISABLED (search-only mode)")
    logger.info("    To enable AI features, add credentials to .env:")
    logger.info("    - AZURE_OPENAI_API_KEY")
    logger.info("    - AZURE_OPENAI_ENDPOINT")
    logger.info("    - AZURE_OPENAI_DEPLOYMENT_NAME")

total_docs = sum(data.get('total_documents', 0) for data in stats.values())
logger.info(f"  ✓ Total documents indexed: {total_docs}")
logger.info(f"  ✓ Collections available: {len(stats)}")

logger.info("\nNext steps:")
if total_docs == 0:
    logger.info("  1. Index your codebase: ./reindex.sh")
    logger.info("  2. Launch STAG: streamlit run stag_app.py")
else:
    logger.info("  1. Launch STAG: streamlit run stag_app.py")
    logger.info("  2. Try query: 'What parsers are available?'")
logger.info("  3. Open browser: http://localhost:8501")
logger.info("\n" + "=" * 60)
