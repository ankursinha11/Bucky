#!/usr/bin/env python3
"""
Codebase Indexer - Index parsed codebase into vector database
Populates ChromaDB for RAG chatbot queries
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from services.codebase_indexer import CodebaseIndexer
from services.deep_indexer import DeepIndexer
from parsers.abinitio import AbInitioParser
from parsers.hadoop import HadoopParser
from parsers.databricks import DatabricksParser
from parsers.custom import CustomDocumentParser
from parsers.hadoop.deep_parser_multi_repo import DeepHadoopParserMultiRepo
from parsers.databricks.deep_parser_multi_repo import DeepDatabricksParserMultiRepo
from parsers.abinitio.deep_parser import DeepAbInitioParser


def index_from_parser_deep(
    parser_type: str,
    source_path: str,
    vector_db_path: str = "./outputs/vector_db",
    use_ai: bool = True,
):
    """
    DEEP PARSING with AI analysis and 3-tier indexing

    Args:
        parser_type: Type of parser
        source_path: Path to source
        vector_db_path: Vector DB path
        use_ai: Whether to use AI analysis (default True)
    """
    print("\n" + "=" * 70)
    print("  🚀 DEEP CODEBASE ANALYSIS - 3-Tier + AI")
    print("=" * 70 + "\n")

    if not Path(source_path).exists():
        print(f"❌ Source path not found: {source_path}")
        sys.exit(1)

    # Select deep parser based on type
    deep_parser_map = {
        'hadoop': DeepHadoopParserMultiRepo,
        'databricks': DeepDatabricksParserMultiRepo,
        'abinitio': DeepAbInitioParser,
    }

    if parser_type not in deep_parser_map:
        print(f"⚠️  Deep parsing not yet supported for {parser_type}")
        print(f"   Supported: {', '.join(deep_parser_map.keys())}")
        print(f"   Falling back to standard parsing...")
        return index_from_parser(parser_type, source_path, vector_db_path)

    print(f"📂 Deep parsing {parser_type.upper()} repository: {source_path}\n")

    # Use deep parser
    deep_parser = deep_parser_map[parser_type](use_ai=use_ai)
    result = deep_parser.parse_directory(source_path)

    repository = result.get("repository")
    repositories = result.get("repositories", [])  # Multi-app support
    workflow_flows = result.get("workflow_flows", [])
    script_logics = result.get("script_logics", [])

    print(f"\n✓ Deep parsing complete!")

    # Show appropriate message for single vs multiple repositories
    if repositories and len(repositories) > 1:
        print(f"   📦 Applications: {len(repositories)}")
        for repo in repositories:
            print(f"      - {repo.name}")
    elif repository:
        print(f"   📊 Tier 1 (Repository): {repository.name}")

    print(f"   📊 Tier 2 (Workflows): {len(workflow_flows)} workflows")
    print(f"   📊 Tier 3 (Scripts): {len(script_logics)} scripts analyzed")
    print(f"   🔍 Total transformations: {sum(len(s.transformations) for s in script_logics)}")
    if use_ai:
        print(f"   🤖 AI analysis: ENABLED")

    # Index with deep indexer
    print("\n" + "=" * 70)
    print("📊 Indexing into vector database (3-tier structure)...")
    print("=" * 70)

    try:
        indexer = DeepIndexer(vector_db_path=vector_db_path)
        counts = indexer.index_deep_analysis(
            repository=repository,
            repositories=repositories if repositories else None,
            workflow_flows=workflow_flows,
            script_logics=script_logics,
        )

        print("\n✅ Deep indexing complete!")
        print(f"   📁 Tier 1 (Repository):      {counts.get('tier1_repository', 0)}")
        print(f"   📁 Tier 2 (Workflows):        {counts.get('tier2_workflows', 0)}")
        print(f"   📁 Tier 3 (Scripts):          {counts.get('tier3_scripts', 0)}")
        print(f"   📁 Tier 3 (Transformations):  {counts.get('tier3_transformations', 0)}")
        print(f"   📁 Tier 3 (Lineages):         {counts.get('tier3_lineage', 0)}")

        total_docs = sum(counts.values())
        print(f"\n   📊 Total documents: {total_docs}")

        stats = indexer.get_stats()
        db_stats = stats.get("vector_db", {})
        print(f"\n📁 Vector Database: {vector_db_path}")
        print(f"   Collections: {db_stats.get('collection_count', 0)}")
        print(f"   Documents:   {db_stats.get('document_count', 0)}")

        print("\n🚀 Ready for INTELLIGENT chatbot queries!")
        print("   The chatbot now understands:")
        print("   ✓ Repository-level architecture")
        print("   ✓ Workflow execution flows")
        print("   ✓ Script logic and transformations")
        print("   ✓ Business purposes (AI-analyzed)")
        print("   ✓ Column-level data lineage")
        print("\n   Run: python chatbot_cli.py")
        print()

    except Exception as e:
        logger.error(f"Indexing error: {e}")
        print(f"\n❌ Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def index_from_parser(
    parser_type: str,
    source_path: str,
    vector_db_path: str = "./outputs/vector_db",
):
    """
    Parse codebase using specified parser and index into vector database
    (BASIC parsing - use --deep for intelligent analysis)

    Args:
        parser_type: Type of parser (abinitio, hadoop, databricks, custom)
        source_path: Path to source files/directory
        vector_db_path: Path to vector database
    """
    print("\n" + "=" * 70)
    print("  Codebase Indexer - Populating Vector Database for RAG")
    print("=" * 70 + "\n")

    # Validate source path
    if not Path(source_path).exists():
        print(f"❌ Source path not found: {source_path}")
        sys.exit(1)

    # Initialize indexer
    indexer = CodebaseIndexer(vector_db_path=vector_db_path)

    # Select parser
    parser_map = {
        'abinitio': AbInitioParser,
        'hadoop': HadoopParser,
        'databricks': DatabricksParser,
        'custom': CustomDocumentParser,
    }

    if parser_type not in parser_map:
        print(f"❌ Unknown parser type: {parser_type}")
        print(f"   Supported: {', '.join(parser_map.keys())}")
        sys.exit(1)

    print(f"📂 Parsing with {parser_type.upper()} parser from: {source_path}\n")

    try:
        parser = parser_map[parser_type]()
        result = parser.parse_directory(source_path)

        processes = result["processes"]
        components = result["components"]

        print(f"✓ Parsed: {len(processes)} processes, {len(components)} components")

    except Exception as e:
        logger.error(f"Error parsing: {e}")
        print(f"❌ Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Check if we have anything to index
    if not processes and not components:
        print("\n❌ No data to index!")
        sys.exit(1)

    # Index all data
    print("\n" + "=" * 70)
    print("📊 Indexing into vector database...")
    print("=" * 70)

    try:
        counts = indexer.index_from_analysis(
            processes=processes,
            components=components,
        )

        print("\n✅ Indexing complete!")
        print(f"   Processes:  {counts['processes']}")
        print(f"   Components: {counts['components']}")
        print(f"   STTM:       {counts['sttm']}")
        print(f"   Gaps:       {counts['gaps']}")
        print(f"   Total:      {sum(counts.values())} documents")

        # Show stats
        stats = indexer.get_stats()
        db_stats = stats.get("vector_db", {})
        print(f"\n📁 Vector Database: {vector_db_path}")
        print(f"   Collections: {db_stats.get('collection_count', 0)}")
        print(f"   Documents:   {db_stats.get('document_count', 0)}")

        print("\n🚀 Ready for chatbot queries!")
        print("   Run: python chatbot_cli.py")
        print()

    except Exception as e:
        logger.error(f"Indexing error: {e}")
        print(f"\n❌ Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def index_from_parsers(
    abinitio_path: str = None,
    hadoop_path: str = None,
    databricks_path: str = None,
    custom_path: str = None,
    vector_db_path: str = "./outputs/vector_db",
):
    """
    Parse codebases and index into vector database

    Args:
        abinitio_path: Path to Ab Initio files
        hadoop_path: Path to Hadoop repository
        databricks_path: Path to Databricks notebooks
        vector_db_path: Path to vector database
    """
    print("\n" + "=" * 70)
    print("  Codebase Indexer - Populating Vector Database for RAG")
    print("=" * 70 + "\n")

    # Initialize indexer
    indexer = CodebaseIndexer(vector_db_path=vector_db_path)

    all_processes = []
    all_components = []

    # Parse Ab Initio
    if abinitio_path and Path(abinitio_path).exists():
        print(f"\n📂 Parsing Ab Initio files from: {abinitio_path}")
        try:
            parser = AbInitioParser()
            result = parser.parse_directory(abinitio_path)

            all_processes.extend(result["processes"])
            all_components.extend(result["components"])

            print(f"✓ Ab Initio: {len(result['processes'])} processes, "
                  f"{len(result['components'])} components")

        except Exception as e:
            logger.error(f"Error parsing Ab Initio: {e}")
            print(f"❌ Ab Initio parsing failed: {e}")
    elif abinitio_path:
        print(f"⚠ Ab Initio path not found: {abinitio_path}")

    # Parse Hadoop
    if hadoop_path and Path(hadoop_path).exists():
        print(f"\n📂 Parsing Hadoop repository from: {hadoop_path}")
        try:
            parser = HadoopParser()
            result = parser.parse_directory(hadoop_path)

            all_processes.extend(result["processes"])
            all_components.extend(result["components"])

            print(f"✓ Hadoop: {len(result['processes'])} processes, "
                  f"{len(result['components'])} components")

        except Exception as e:
            logger.error(f"Error parsing Hadoop: {e}")
            print(f"❌ Hadoop parsing failed: {e}")
    elif hadoop_path:
        print(f"⚠ Hadoop path not found: {hadoop_path}")

    # Parse Databricks
    if databricks_path and Path(databricks_path).exists():
        print(f"\n📂 Parsing Databricks notebooks from: {databricks_path}")
        try:
            parser = DatabricksParser()
            result = parser.parse_directory(databricks_path)

            all_processes.extend(result["processes"])
            all_components.extend(result["components"])

            print(f"✓ Databricks: {len(result['processes'])} processes, "
                  f"{len(result['components'])} components")

        except Exception as e:
            logger.error(f"Error parsing Databricks: {e}")
            print(f"❌ Databricks parsing failed: {e}")
    elif databricks_path:
        print(f"⚠ Databricks path not found: {databricks_path}")

    # Parse Custom Documents
    if custom_path and Path(custom_path).exists():
        print(f"\n📂 Parsing custom documents from: {custom_path}")
        try:
            parser = CustomDocumentParser()
            result = parser.parse_directory(custom_path)

            all_processes.extend(result["processes"])
            all_components.extend(result["components"])

            print(f"✓ Custom: {len(result['processes'])} documents, "
                  f"{len(result['components'])} components")

        except Exception as e:
            logger.error(f"Error parsing custom documents: {e}")
            print(f"❌ Custom documents parsing failed: {e}")
    elif custom_path:
        print(f"⚠ Custom path not found: {custom_path}")

    # Check if we have anything to index
    if not all_processes and not all_components:
        print("\n❌ No data to index!")
        print("   Provide at least one valid codebase path")
        sys.exit(1)

    # Index all data
    print("\n" + "=" * 70)
    print("📊 Indexing into vector database...")
    print("=" * 70)

    try:
        counts = indexer.index_from_analysis(
            processes=all_processes,
            components=all_components,
        )

        print("\n✅ Indexing complete!")
        print(f"   Processes:  {counts['processes']}")
        print(f"   Components: {counts['components']}")
        print(f"   STTM:       {counts['sttm']}")
        print(f"   Gaps:       {counts['gaps']}")
        print(f"   Total:      {sum(counts.values())} documents")

        # Show stats
        stats = indexer.get_stats()
        db_stats = stats.get("vector_db", {})
        print(f"\n📁 Vector Database: {vector_db_path}")
        print(f"   Collections: {db_stats.get('collection_count', 0)}")
        print(f"   Documents:   {db_stats.get('document_count', 0)}")

        print("\n🚀 Ready for chatbot queries!")
        print("   Run: python3 chatbot_cli.py")
        print()

    except Exception as e:
        logger.error(f"Indexing error: {e}")
        print(f"\n❌ Indexing failed: {e}")
        sys.exit(1)


def index_from_json(json_file: str, vector_db_path: str = "./outputs/vector_db"):
    """
    Index from JSON file (for testing)

    Args:
        json_file: Path to JSON file
        vector_db_path: Path to vector database
    """
    print(f"\n📄 Indexing from JSON file: {json_file}")

    indexer = CodebaseIndexer(vector_db_path=vector_db_path)

    try:
        counts = indexer.index_from_json(json_file)

        print(f"\n✅ Indexed {counts['total']} documents from JSON")
        print("\n🚀 Ready for chatbot queries!")
        print("   Run: python3 chatbot_cli.py")
        print()

    except Exception as e:
        logger.error(f"Error indexing JSON: {e}")
        print(f"\n❌ Failed to index JSON: {e}")
        sys.exit(1)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Index parsed codebase into vector database for RAG chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 🚀 DEEP PARSING with AI analysis (RECOMMENDED for Hadoop!)
  python index_codebase.py --parser hadoop --source /path/to/hadoop --deep

  This will:
    ✓ Parse workflow execution flows
    ✓ Extract script logic (Pig, Spark, Hive)
    ✓ Identify transformations (FILTER, JOIN, GROUP BY, etc.)
    ✓ Track column-level data lineage
    ✓ Use GPT-4 to understand business logic
    ✓ Generate flow diagrams
    → Result: INTELLIGENT chatbot that understands your code!

  # Deep parsing without AI (logic extraction only)
  python index_codebase.py --parser hadoop --source /path/to/hadoop --deep --no-ai

  # Basic parsing (fast but less intelligent)
  python index_codebase.py --parser hadoop --source /path/to/hadoop
  python index_codebase.py --parser databricks --source /path/to/notebooks
  python index_codebase.py --parser custom --source ./custom_documents

  # Advanced: Index multiple codebases at once
  python index_codebase.py --hadoop-path /path/to/hadoop --databricks-path /path/to/notebooks

  # Index from JSON
  python index_codebase.py --json-file output.json

Supported parsers:
  - abinitio: Ab Initio .mp files (supports --deep!)
  - hadoop: Oozie workflows, coordinators (supports --deep!)
  - databricks: Notebooks (.py, .sql, .ipynb) (supports --deep!)
  - custom: Excel, Word, PDF, CSV, text files

Deep parsing supports: Hadoop, Databricks, Ab Initio
        """
    )

    # Simple interface (new, recommended)
    parser.add_argument(
        "--parser",
        type=str,
        choices=['abinitio', 'hadoop', 'databricks', 'custom'],
        help="Parser type to use",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Source directory path (use with --parser)",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="🚀 Enable DEEP parsing with AI analysis (3-tier indexing, script logic extraction, AI insights)",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI analysis (use with --deep for logic extraction without AI)",
    )

    # Advanced interface (existing, for multiple codebases)
    parser.add_argument(
        "--abinitio-path",
        type=str,
        help="Path to Ab Initio files directory",
    )
    parser.add_argument(
        "--hadoop-path",
        type=str,
        help="Path to Hadoop repository",
    )
    parser.add_argument(
        "--databricks-path",
        type=str,
        help="Path to Databricks notebooks",
    )
    parser.add_argument(
        "--custom-path",
        type=str,
        help="Path to custom documents (Excel, Word, PDF, etc.)",
    )

    # JSON input (alternative)
    parser.add_argument(
        "--json-file",
        type=str,
        help="Path to JSON file with parsed data (alternative to parsing)",
    )

    # Vector database path
    parser.add_argument(
        "--vector-db-path",
        type=str,
        default="./outputs/vector_db",
        help="Path to vector database directory (default: ./outputs/vector_db)",
    )

    args = parser.parse_args()

    # Simple interface: --parser and --source
    if args.parser and args.source:
        # Check if deep parsing requested
        if args.deep:
            use_ai = not args.no_ai  # Use AI unless --no-ai specified
            index_from_parser_deep(
                parser_type=args.parser,
                source_path=args.source,
                vector_db_path=args.vector_db_path,
                use_ai=use_ai,
            )
        else:
            index_from_parser(
                parser_type=args.parser,
                source_path=args.source,
                vector_db_path=args.vector_db_path,
            )

    # JSON mode
    elif args.json_file:
        if not Path(args.json_file).exists():
            print(f"❌ JSON file not found: {args.json_file}")
            sys.exit(1)
        index_from_json(args.json_file, args.vector_db_path)

    # Advanced mode: multiple codebases
    elif args.abinitio_path or args.hadoop_path or args.databricks_path or args.custom_path:
        index_from_parsers(
            abinitio_path=args.abinitio_path,
            hadoop_path=args.hadoop_path,
            databricks_path=args.databricks_path,
            custom_path=args.custom_path,
            vector_db_path=args.vector_db_path,
        )

    # No input provided
    else:
        print("❌ No input provided!")
        print("\nQuick Start (Recommended):")
        print("  python index_codebase.py --parser hadoop --source /path/to/hadoop")
        print("  python index_codebase.py --parser custom --source ./custom_documents")
        print("\nFor more options:")
        print("  python index_codebase.py --help")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
