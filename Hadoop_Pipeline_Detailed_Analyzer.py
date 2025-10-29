import sys
import os
from pathlib import Path

print("="*80)
print("SIMPLE PATH TEST")
print("="*80)

# Show current directory
print("\n1. Current working directory:")
print(f"   {os.getcwd()}")

# Show what was passed as argument
print("\n2. Command line arguments:")
if len(sys.argv) > 1:
    print(f"   Argument provided: {sys.argv[1]}")
    
    # Try to resolve the path
    path = Path(sys.argv[1])
    print(f"\n3. Path resolution:")
    print(f"   Input path: {sys.argv[1]}")
    print(f"   Absolute path: {path.absolute()}")
    print(f"   Path exists: {path.exists()}")
    
    if path.exists():
        print(f"\n4. Contents of {sys.argv[1]}:")
        try:
            for item in path.iterdir():
                if item.is_dir():
                    print(f"   [DIR]  {item.name}")
                else:
                    print(f"   [FILE] {item.name}")
        except Exception as e:
            print(f"   ERROR: {e}")
    else:
        print(f"\n   Path does NOT exist!")
        print(f"\n   Try one of these:")
        print(f"   - python test_simple.py hadoop_repos")
        print(f"   - python test_simple.py ./hadoop_repos")
        print(f"   - python test_simple.py ../hadoop_repos")
else:
    print("   No argument provided")
    print("\n   Usage: python test_simple.py <path>")

print("\n" + "="*80)
print("DONE")
print("="*80)

