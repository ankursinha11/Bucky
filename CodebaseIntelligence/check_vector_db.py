#!/usr/bin/env python3
"""
Check Vector Database Status
Diagnose what's in the ChromaDB database
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import chromadb
    from chromadb.config import Settings
    print("✓ ChromaDB installed\n")
except ImportError:
    print("✗ ChromaDB not installed. Run: pip install chromadb")
    sys.exit(1)


def check_database(db_path: str):
    """Check database at given path"""
    print(f"Checking database at: {db_path}")
    print("=" * 70)

    db_path_obj = Path(db_path)

    # Check if path exists
    if not db_path_obj.exists():
        print(f"✗ Database path does not exist: {db_path}")
        return False

    print(f"✓ Database path exists")

    # Check if it has any contents
    contents = list(db_path_obj.iterdir())
    print(f"✓ Database has {len(contents)} items")

    if len(contents) > 0:
        print("\nContents:")
        for item in contents[:5]:
            print(f"  - {item.name}")
        if len(contents) > 5:
            print(f"  ... and {len(contents) - 5} more")

    # Try to connect
    print("\nAttempting to connect to database...")
    try:
        client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False,
            ),
        )
        print("✓ Connected to ChromaDB")

        # List collections
        collections = client.list_collections()
        print(f"\n✓ Found {len(collections)} collection(s):")

        for coll in collections:
            print(f"\n  Collection: {coll.name}")
            try:
                count = coll.count()
                print(f"    Documents: {count}")

                if count > 0:
                    # Peek at first document
                    sample = coll.peek(limit=1)
                    if sample and "ids" in sample and sample["ids"]:
                        print(f"    Sample ID: {sample['ids'][0]}")
                    if sample and "metadatas" in sample and sample["metadatas"]:
                        metadata = sample["metadatas"][0]
                        print(f"    Sample metadata: {metadata}")
            except Exception as e:
                print(f"    Error getting details: {e}")

        return len(collections) > 0

    except Exception as e:
        print(f"✗ Error connecting to database: {e}")
        return False


def main():
    print("=" * 70)
    print("Vector Database Diagnostic Tool")
    print("=" * 70)
    print()

    # Check common locations
    locations_to_check = [
        "./outputs/vector_db",
        "./chroma_db",
        "../outputs/vector_db",
        "outputs/vector_db",
    ]

    found = False

    for location in locations_to_check:
        print()
        if check_database(location):
            found = True
            print("\n" + "=" * 70)
            print(f"✓ FOUND POPULATED DATABASE at: {location}")
            print("=" * 70)
            break
        print()

    if not found:
        print("\n" + "=" * 70)
        print("⚠️ NO POPULATED DATABASE FOUND")
        print("=" * 70)
        print("\nPossible issues:")
        print("1. Database was created in a different location")
        print("2. Indexing didn't complete successfully")
        print("3. Database was deleted or moved")
        print()
        print("Solution: Re-run indexing:")
        print("  python index_codebase.py --parser hadoop --source /path/to/hadoop")
        print()

    # Look for any ChromaDB directories
    print("\nSearching for any ChromaDB directories...")
    current_dir = Path(".")
    chroma_dirs = []

    for path in current_dir.rglob("chroma.sqlite3"):
        chroma_dir = path.parent
        chroma_dirs.append(str(chroma_dir))
        print(f"  Found: {chroma_dir}")

    if chroma_dirs:
        print(f"\n✓ Found {len(chroma_dirs)} potential ChromaDB location(s)")
        print("\nTry checking these locations specifically:")
        for d in chroma_dirs:
            print(f"  python check_vector_db.py {d}")
    else:
        print("  No ChromaDB databases found in current directory")


if __name__ == "__main__":
    # Allow custom path as argument
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        check_database(db_path)
    else:
        main()
