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
from parsers.abinitio import AbInitioParser
from parsers.hadoop import HadoopParser
from parsers.databricks import DatabricksParser


def index_from_parsers(
    abinitio_path: str = None,
    hadoop_path: str = None,
    databricks_path: str = None,
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
        description="Index parsed codebase into vector database for RAG chatbot"
    )

    # Codebase paths
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

    # Check if JSON mode or parser mode
    if args.json_file:
        # Index from JSON
        if not Path(args.json_file).exists():
            print(f"❌ JSON file not found: {args.json_file}")
            sys.exit(1)

        index_from_json(args.json_file, args.vector_db_path)

    elif args.abinitio_path or args.hadoop_path or args.databricks_path:
        # Index from parsers
        index_from_parsers(
            abinitio_path=args.abinitio_path,
            hadoop_path=args.hadoop_path,
            databricks_path=args.databricks_path,
            vector_db_path=args.vector_db_path,
        )

    else:
        # No input provided
        print("❌ No input provided!")
        print("\nUsage:")
        print("  Index from codebases:")
        print("    python3 index_codebase.py --abinitio-path /path/to/abinitio")
        print("    python3 index_codebase.py --hadoop-path /path/to/hadoop")
        print("    python3 index_codebase.py --abinitio-path ... --hadoop-path ...")
        print("\n  Or index from JSON:")
        print("    python3 index_codebase.py --json-file output.json")
        print()
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
