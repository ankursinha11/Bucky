#!/bin/bash
# Setup script for Codebase Intelligence standalone parser

echo "Setting up virtual environment for Codebase Intelligence..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install loguru pandas openpyxl lxml pyyaml

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use the standalone parser:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the parser:"
echo "   python run_parser_standalone.py hadoop /path/to/your/repo"
echo ""
echo "Example:"
echo "   python run_parser_standalone.py hadoop ~/repos/app-cdd"

