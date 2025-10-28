#!/bin/bash

# Installation script for Ubuntu VM

echo "=================================================="
echo "Codebase Intelligence Platform - Installation"
echo "=================================================="

# Check Python version
echo -e "\n[1/5] Checking Python version..."
python3 --version

if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Installing..."
    sudo apt update
    sudo apt install python3 python3-pip -y
fi

# Check pip
echo -e "\n[2/5] Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo "pip3 not found. Installing..."
    sudo apt install python3-pip -y
fi

# Install dependencies
echo -e "\n[3/5] Installing Python dependencies..."
pip3 install --user -r requirements.txt

# Create output directories
echo -e "\n[4/5] Creating output directories..."
mkdir -p outputs/reports
mkdir -p outputs/logs
mkdir -p outputs/test_reports

# Make scripts executable
echo -e "\n[5/5] Making scripts executable..."
chmod +x run_analysis.py
chmod +x test_system.py

echo -e "\n=================================================="
echo "✓ Installation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Run test: python3 test_system.py"
echo "  2. Check outputs in: outputs/test_reports/"
echo "  3. Read: QUICKSTART.md for usage"
echo ""
echo "=================================================="
