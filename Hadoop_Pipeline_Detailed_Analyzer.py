#!/usr/bin/env python3
"""
Hadoop Pipeline Analyzer - DEBUG VERSION
Run this to see what's happening step by step
"""

import os
from pathlib import Path

def test_analyzer(hadoop_repos_path):
    """Test the analyzer with detailed output"""
    
    print("="*80)
    print("HADOOP PIPELINE ANALYZER - DEBUG MODE")
    print("="*80)
    
    # Check if path exists
    path = Path(hadoop_repos_path)
    print(f"\n1. Checking path: {path.absolute()}")
    
    if not path.exists():
        print(f"   ❌ ERROR: Path does not exist!")
        print(f"   Please check the path and try again.")
        return
    
    print(f"   ✓ Path exists")
    
    # List what's in the folder
    print(f"\n2. Contents of {hadoop_repos_path}:")
    try:
        items = list(path.iterdir())
        print(f"   Found {len(items)} items:")
        for item in items:
            if item.is_dir():
                print(f"   📁 {item.name}")
            else:
                print(f"   📄 {item.name}")
    except Exception as e:
        print(f"   ❌ ERROR: Cannot read directory: {e}")
        return
    
    # Check for repositories (folders with workflows/ or .xml files)
    print(f"\n3. Looking for Hadoop repositories...")
    repos = []
    for item in path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it has workflows directory
            workflows_dir = item / "workflows"
            xml_files = list(item.rglob("*.xml"))
            
            if workflows_dir.exists():
                print(f"   ✓ {item.name} - has workflows/ directory")
                repos.append(item)
            elif xml_files:
                print(f"   ✓ {item.name} - has {len(xml_files)} XML files")
                repos.append(item)
            else:
                print(f"   ✗ {item.name} - no workflows or XML files")
    
    if not repos:
        print(f"\n   ❌ ERROR: No Hadoop repositories found!")
        print(f"   Make sure the folder contains Hadoop repos with:")
        print(f"      - A 'workflows/' directory, OR")
        print(f"      - XML workflow files")
        return
    
    print(f"\n4. Found {len(repos)} Hadoop repositories:")
    for repo in repos:
        print(f"   - {repo.name}")
    
    print(f"\n5. Next step: Run the full analyzer")
    print(f"\n   Command:")
    print(f"   python Hadoop_Pipeline_Analyzer_Enhanced.py \"{hadoop_repos_path}\"")
    print(f"\n   OR if that doesn't work:")
    print(f"   python Hadoop_Pipeline_Analyzer_Enhanced.py \"{hadoop_repos_path}\" --output \"Hadoop_Analysis.xlsx\"")
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage: python hadoop_pipe_analyzer_debug.py <path_to_hadoop_repos>")
        print("\nExample:")
        print("  python hadoop_pipe_analyzer_debug.py ./hadoop_repos")
        print("  python hadoop_pipe_analyzer_debug.py \"C:/Users/YourName/hadoop_repos\"")
        sys.exit(1)
    
    test_analyzer(sys.argv[1])

