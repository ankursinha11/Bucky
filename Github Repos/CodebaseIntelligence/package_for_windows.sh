#!/bin/bash
# Package updated files for transfer to Windows

echo "=================================="
echo "Packaging Updated Files for Windows"
echo "=================================="

OUTPUT_DIR="updated_files_for_windows"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Create directory structure
mkdir -p "$OUTPUT_DIR/parsers/hadoop"
mkdir -p "$OUTPUT_DIR/parsers/abinitio"
mkdir -p "$OUTPUT_DIR/parsers/databricks"
mkdir -p "$OUTPUT_DIR/core/models"
mkdir -p "$OUTPUT_DIR/services"

# Copy updated parser files
echo "Copying parsers..."
cp parsers/__init__.py "$OUTPUT_DIR/parsers/"
cp parsers/hadoop/__init__.py "$OUTPUT_DIR/parsers/hadoop/"
cp parsers/hadoop/parser.py "$OUTPUT_DIR/parsers/hadoop/"
cp parsers/hadoop/oozie_parser.py "$OUTPUT_DIR/parsers/hadoop/"
cp parsers/abinitio/__init__.py "$OUTPUT_DIR/parsers/abinitio/"
cp parsers/abinitio/parser.py "$OUTPUT_DIR/parsers/abinitio/"
cp parsers/abinitio/mp_file_parser.py "$OUTPUT_DIR/parsers/abinitio/"
cp parsers/abinitio/patterns.py "$OUTPUT_DIR/parsers/abinitio/"
cp parsers/databricks/__init__.py "$OUTPUT_DIR/parsers/databricks/"
cp parsers/databricks/parser.py "$OUTPUT_DIR/parsers/databricks/"

# Copy updated model files
echo "Copying models..."
cp core/models/component.py "$OUTPUT_DIR/core/models/"

# Copy updated service files
echo "Copying services..."
cp services/codebase_indexer.py "$OUTPUT_DIR/services/"

# Copy verification script
echo "Copying verification script..."
cp verify_parser_version.py "$OUTPUT_DIR/"

# Create instructions file
cat > "$OUTPUT_DIR/COPY_INSTRUCTIONS.txt" << 'EOF'
INSTRUCTIONS FOR WINDOWS
========================

1. Copy all files from this directory to your Windows CodebaseIntelligence folder
2. Maintain the same directory structure (parsers/, core/models/, services/)
3. Run the verification script:

   cd CodebaseIntelligence
   python verify_parser_version.py

4. You should see all checkmarks (✓) if files were copied correctly
5. Then try running the indexer again

Files included:
- parsers/hadoop/parser.py (CRITICAL - unique ID fix)
- parsers/hadoop/oozie_parser.py (coordinator support)
- parsers/abinitio/* (all Ab Initio parsers)
- parsers/databricks/* (all Databricks parsers)
- core/models/component.py (OOZIE_COORDINATOR type)
- services/codebase_indexer.py (document ID structure fix)
- verify_parser_version.py (verification script)

Key Changes:
- Process IDs now include file hash for uniqueness
- Component IDs now include index for uniqueness
- Parser finds all workflow and coordinator files (not just workflow.xml)
- Coordinators properly supported and tagged
- Document IDs at top level (not in metadata)
EOF

echo ""
echo "✓ Packaged files to: $OUTPUT_DIR/"
echo ""
echo "File count:"
find "$OUTPUT_DIR" -type f | wc -l

echo ""
echo "Next steps:"
echo "1. Copy the '$OUTPUT_DIR' folder to Windows"
echo "2. Follow instructions in COPY_INSTRUCTIONS.txt"
