#!/usr/bin/env python3
"""
Complete Chatbot Diagnostics
Find out exactly why the chatbot isn't finding documents
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("CHATBOT DIAGNOSTICS")
print("=" * 80)
print()

# Step 1: Check current directory
print("Step 1: Environment Check")
print("-" * 80)
print(f"Current directory: {os.getcwd()}")
print(f"Script directory:  {Path(__file__).parent}")
print()

# Step 2: Find all possible vector databases
print("Step 2: Searching for Vector Databases")
print("-" * 80)

possible_locations = [
    "./outputs/vector_db",
    "outputs/vector_db",
    str(Path(__file__).parent / "outputs" / "vector_db"),
]

print("Checking these locations:")
for loc in possible_locations:
    print(f"  {loc}")
print()

found_databases = []

for location in possible_locations:
    full_path = Path(location).resolve()
    print(f"Checking: {full_path}")

    if full_path.exists():
        # Check if it has ChromaDB files
        has_chroma = any(
            item.name == "chroma.sqlite3" for item in full_path.rglob("chroma.sqlite3")
        )

        if has_chroma:
            print(f"  ✓ Found ChromaDB database!")
            found_databases.append(str(full_path))
        else:
            print(f"  ⚠️  Directory exists but no ChromaDB files found")
    else:
        print(f"  ✗ Does not exist")
    print()

if not found_databases:
    print("=" * 80)
    print("❌ NO VECTOR DATABASES FOUND")
    print("=" * 80)
    print()
    print("This means you need to run indexing first:")
    print("  python index_codebase.py --parser hadoop --source /path/to/hadoop")
    print()
    sys.exit(1)

# Step 3: Check what's in the database
print("Step 3: Checking Database Contents")
print("-" * 80)

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("❌ ChromaDB not installed!")
    print("   Install with: pip install chromadb sentence-transformers")
    sys.exit(1)

for db_path in found_databases:
    print(f"\nDatabase: {db_path}")
    print("-" * 80)

    try:
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False,
            ),
        )

        collections = client.list_collections()
        print(f"Collections: {len(collections)}")

        for coll in collections:
            count = coll.count()
            print(f"  - {coll.name}: {count} documents")

            if count > 0:
                # Sample a document
                sample = coll.peek(limit=1)
                if sample and "ids" in sample and sample["ids"]:
                    print(f"    Sample ID: {sample['ids'][0]}")
                if sample and "metadatas" in sample and sample["metadatas"]:
                    meta = sample["metadatas"][0]
                    print(f"    Sample type: {meta.get('doc_type', 'unknown')}")
                    print(f"    Sample system: {meta.get('system', 'unknown')}")

    except Exception as e:
        print(f"  ❌ Error: {e}")

# Step 4: Check chatbot configuration
print("\n" + "=" * 80)
print("Step 4: Chatbot Configuration")
print("-" * 80)

sys.path.insert(0, str(Path(__file__).parent))

try:
    from services.rag_chatbot_integrated import CodebaseRAGChatbot

    # Check what path the chatbot would use by default
    default_path = "./outputs/vector_db"
    resolved_path = Path(default_path).resolve()

    print(f"Chatbot default path:  {default_path}")
    print(f"Resolved to:           {resolved_path}")
    print()

    if str(resolved_path) in found_databases:
        print("✓ Chatbot is looking in the right place!")
    else:
        print("❌ MISMATCH!")
        print(f"   Chatbot looks in:  {resolved_path}")
        print(f"   Database exists in: {found_databases[0]}")
        print()
        print("Solution: Run chatbot from the CodebaseIntelligence directory:")
        print(f"  cd {Path(__file__).parent}")
        print("  python chatbot_cli.py")

except Exception as e:
    print(f"❌ Error loading chatbot: {e}")

# Step 5: Test actual search
print("\n" + "=" * 80)
print("Step 5: Testing Search")
print("-" * 80)

if found_databases:
    db_to_test = found_databases[0]
    print(f"Testing search in: {db_to_test}")
    print()

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from services.local_search.local_search_client import LocalSearchClient

        search_client = LocalSearchClient(persist_directory=db_to_test)
        search_client.create_index("codebase")

        print("Performing test search for 'workflow'...")
        results = search_client.search(query="workflow", top=3)

        if results and "results" in results and results["results"]:
            result_list = results["results"]
            print(f"✓ Found {len(result_list)} results!")
            print()
            for i, result in enumerate(result_list[:3], 1):
                print(f"  Result {i}:")
                print(f"    ID: {result.get('id', 'unknown')}")
                print(f"    Score: {result.get('score', 0):.3f}")
                metadata = result.get('metadata', {})
                print(f"    Type: {metadata.get('doc_type', 'unknown')}")
                print(f"    System: {metadata.get('system', 'unknown')}")
                print()
        else:
            print("❌ Search returned no results")
            print("   The database exists but has no documents!")

    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()

# Final Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if found_databases:
    print(f"✓ Found vector database at: {found_databases[0]}")
    print()
    print("To use the chatbot:")
    print(f"  1. cd {Path(__file__).parent}")
    print("  2. python chatbot_cli.py")
    print()
else:
    print("❌ No vector database found - run indexing first")
    print("  python index_codebase.py --parser hadoop --source /path/to/hadoop")

print()
