#!/usr/bin/env python3
"""
Reset Specific ChromaDB Collection
===================================
Selectively reset/repair a specific collection without losing others

Usage:
    python reset_collection.py                    # Interactive - choose collection
    python reset_collection.py abinitio           # Reset abinitio collection
    python reset_collection.py --list             # List all collections
"""

import sys
from pathlib import Path
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    logger.error("ChromaDB not installed. Run: pip install chromadb")
    sys.exit(1)


def list_collections(db_path: str = "./outputs/vector_db"):
    """List all collections in the database"""
    db_path = Path(db_path)

    if not db_path.exists():
        logger.warning(f"Database doesn't exist at {db_path}")
        return []

    try:
        client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        collections = client.list_collections()

        print("\n" + "="*60)
        print("Available Collections")
        print("="*60)

        collection_info = []
        for coll in collections:
            try:
                count = coll.count()
                collection_info.append({
                    'name': coll.name,
                    'count': count,
                    'metadata': coll.metadata
                })
                print(f"\n📦 {coll.name}")
                print(f"   Documents: {count}")
                print(f"   Metadata: {coll.metadata}")
            except Exception as e:
                collection_info.append({
                    'name': coll.name,
                    'count': -1,
                    'error': str(e)
                })
                print(f"\n📦 {coll.name}")
                print(f"   ⚠️ ERROR: {e}")

        print("\n" + "="*60)
        return collection_info

    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return []


def reset_collection(collection_name: str, db_path: str = "./outputs/vector_db", force: bool = False):
    """
    Reset a specific collection

    Args:
        collection_name: Name of collection to reset
        db_path: Path to vector database
        force: Skip confirmation
    """
    db_path = Path(db_path)

    if not db_path.exists():
        logger.error(f"Database doesn't exist at {db_path}")
        return False

    try:
        client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Check if collection exists
        collections = [c.name for c in client.list_collections()]

        if collection_name not in collections:
            logger.error(f"Collection '{collection_name}' not found")
            logger.info(f"Available collections: {', '.join(collections)}")
            return False

        # Get collection info
        collection = client.get_collection(collection_name)
        try:
            count = collection.count()
            logger.info(f"Collection '{collection_name}' has {count} documents")
        except Exception as e:
            logger.warning(f"Could not read collection (may be corrupted): {e}")
            count = -1

        # Confirm deletion
        if not force:
            print("\n" + "="*60)
            print(f"⚠️  WARNING: Deleting collection '{collection_name}'")
            print("="*60)
            if count >= 0:
                print(f"\nDocuments to delete: {count}")
            else:
                print(f"\nCollection appears corrupted")
            print(f"\nOther collections will NOT be affected.")
            print(f"You will need to re-index {collection_name} data after this.")
            print("\nAre you sure? (yes/no): ", end="")

            response = input().strip().lower()
            if response != "yes":
                logger.info("❌ Deletion cancelled")
                return False

        # Delete collection
        logger.info(f"Deleting collection '{collection_name}'...")
        client.delete_collection(collection_name)
        logger.info(f"✅ Successfully deleted collection '{collection_name}'")

        # Show remaining collections
        remaining = [c.name for c in client.list_collections()]
        logger.info(f"\n📦 Remaining collections: {', '.join(remaining) if remaining else 'None'}")

        print("\n📋 Next steps:")
        print(f"1. Restart Streamlit (if running)")
        print(f"2. Re-index {collection_name} data:")
        print(f"   - Go to Index Management tab")
        print(f"   - Index {collection_name} repository")
        print(f"3. Other collections (Hadoop, Databricks, etc.) are still intact ✓")

        return True

    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        return False


def interactive_reset(db_path: str = "./outputs/vector_db"):
    """Interactive collection selection and reset"""
    print("\n" + "="*60)
    print("ChromaDB Collection Reset Tool")
    print("="*60)

    # List collections
    collections = list_collections(db_path)

    if not collections:
        logger.error("No collections found or database is corrupted")
        logger.warning("You may need to delete the entire database:")
        logger.warning(f"  rm -rf {db_path}")
        return False

    # Get user choice
    print("\nWhich collection do you want to reset?")
    for i, coll in enumerate(collections, 1):
        status = "✓" if coll.get('count', -1) >= 0 else "⚠️ CORRUPTED"
        print(f"  {i}. {coll['name']} ({status})")
    print(f"  0. Cancel")

    try:
        choice = int(input("\nEnter number: "))
        if choice == 0:
            logger.info("Cancelled")
            return False
        if choice < 1 or choice > len(collections):
            logger.error("Invalid choice")
            return False

        selected = collections[choice - 1]
        return reset_collection(selected['name'], db_path, force=False)

    except ValueError:
        logger.error("Invalid input")
        return False
    except KeyboardInterrupt:
        logger.info("\nCancelled")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reset specific ChromaDB collection without affecting others",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reset_collection.py                      # Interactive mode
  python reset_collection.py --list               # List all collections
  python reset_collection.py abinitio             # Reset abinitio collection
  python reset_collection.py hadoop --force       # Force reset hadoop
  python reset_collection.py --all                # Reset ALL collections
        """
    )

    parser.add_argument(
        "collection",
        nargs="?",
        help="Collection name to reset (e.g., 'abinitio', 'hadoop', 'databricks')"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all collections and exit"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Reset ALL collections (dangerous!)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )

    parser.add_argument(
        "--path",
        type=str,
        default="./outputs/vector_db",
        help="Path to vector database (default: ./outputs/vector_db)"
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        list_collections(args.path)
        sys.exit(0)

    # Reset all mode
    if args.all:
        print("\n⚠️⚠️⚠️  WARNING: This will delete ALL collections! ⚠️⚠️⚠️")
        if not args.force:
            response = input("Are you ABSOLUTELY sure? (type 'DELETE ALL'): ")
            if response != "DELETE ALL":
                print("Cancelled")
                sys.exit(0)

        collections = list_collections(args.path)
        for coll in collections:
            reset_collection(coll['name'], args.path, force=True)
        sys.exit(0)

    # Reset specific collection
    if args.collection:
        success = reset_collection(args.collection, args.path, args.force)
        sys.exit(0 if success else 1)

    # Interactive mode
    interactive_reset(args.path)
