# Codebase Intelligence Platform - Client Deployment Guide

**Version:** 2.0
**Date:** October 28, 2025
**For:** Client VM Deployment with Visual Studio Code

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation Steps](#installation-steps)
4. [VS Code Setup](#vs-code-setup)
5. [Configuration](#configuration)
6. [Usage Guide](#usage-guide)
7. [Features](#features)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)
10. [Support & Contact](#support--contact)

---

## Overview

The Codebase Intelligence Platform analyzes legacy codebases (Ab Initio, Hadoop, Databricks) and generates:

- **STTM (Source-to-Target Mapping)** - Column-level mappings with transformation rules
- **Gap Analysis** - Missing components and migration recommendations
- **Clean Excel Reports** - FAWN-style output with readable parameters
- **RAG Chatbot** - AI-powered codebase Q&A with FREE local search

### What's New in v2.0

✅ **FAWN-Based Ab Initio Parser** - Clean, readable parameter extraction
✅ **FREE Local Vector Search** - No Azure AI Search required! (ChromaDB + sentence-transformers)
✅ **Enhanced Hadoop Parser** - Follows sub-workflows, resolves variables
✅ **GraphFlow Extraction** - Complete data lineage tracking
✅ **VS Code Integration** - Full IDE support with Python extension

---

## System Requirements

### Minimum Requirements
- **OS:** Windows 10+, Linux, or macOS
- **RAM:** 8 GB (16 GB recommended)
- **Disk Space:** 10 GB free
- **Python:** 3.9 or higher
- **Network:** Internet for initial setup (optional afterward)

### Recommended Requirements
- **RAM:** 16 GB
- **CPU:** 4+ cores
- **SSD:** For better performance
- **Python:** 3.10 or 3.11

---

## Installation Steps

### Step 1: Install Python

#### Windows
1. Download Python from https://www.python.org/downloads/
2. **IMPORTANT:** Check "Add Python to PATH" during installation
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.10 python3-pip python3-venv -y
python3 --version
pip3 --version
```

#### macOS
```bash
brew install python@3.10
python3 --version
pip3 --version
```

### Step 2: Install Visual Studio Code

1. **Download VS Code:**
   - Visit: https://code.visualstudio.com/download
   - Choose your OS version
   - Install with default settings

2. **Install Python Extension:**
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X or Cmd+Shift+X on Mac)
   - Search for "Python" by Microsoft
   - Click Install

3. **Install Additional Extensions (Optional but Recommended):**
   - **Jupyter** - For notebook support
   - **YAML** - For config files
   - **Excel Viewer** - For viewing output files
   - **GitLens** - For version control

### Step 3: Copy Project to Client VM

#### Method 1: USB/File Transfer
```bash
# Create project directory
mkdir -p ~/CodebaseIntelligence
cd ~/CodebaseIntelligence

# Copy all files from USB/transfer location
cp -r /path/to/CodebaseIntelligence/* .
```

#### Method 2: Git Clone (if available)
```bash
git clone <repository-url> ~/CodebaseIntelligence
cd ~/CodebaseIntelligence
```

#### Method 3: Network Share
```bash
# Mount network share (Windows)
net use Z: \\server\share

# Copy files
xcopy Z:\CodebaseIntelligence C:\CodebaseIntelligence /E /I
```

### Step 4: Set Up Python Virtual Environment

```bash
# Navigate to project
cd ~/CodebaseIntelligence

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate

# You should see (venv) prefix in terminal
```

### Step 5: Install Dependencies

```bash
# Make sure virtual environment is activated (you should see (venv) prefix)

# Install all required packages
pip install --upgrade pip
pip install -r requirements.txt

# This installs:
# - pandas, openpyxl (Excel processing)
# - loguru (logging)
# - pyyaml (config files)
# - chromadb (FREE local vector search)
# - sentence-transformers (FREE embeddings)
# - openai (for RAG chatbot if API key available)
# - lxml, beautifulsoup4 (XML parsing)
```

**Installation Time:** 5-10 minutes depending on internet speed

### Step 6: Verify Installation

```bash
# Test Python imports
python3 -c "import pandas; import chromadb; print('✓ All dependencies installed!')"

# Test parsers
python3 -c "from parsers.abinitio import AbInitioParser; print('✓ Ab Initio parser ready')"
python3 -c "from parsers.hadoop import HadoopParser; print('✓ Hadoop parser ready')"

# Test local search
python3 -c "from services.local_search import LocalSearchClient; print('✓ FREE local search ready')"
```

---

## VS Code Setup

### Open Project in VS Code

```bash
# Open VS Code from project directory
cd ~/CodebaseIntelligence
code .
```

### Configure Python Interpreter

1. Open VS Code
2. Press **Ctrl+Shift+P** (or **Cmd+Shift+P** on Mac)
3. Type: "Python: Select Interpreter"
4. Choose: `./venv/bin/python` or `.\venv\Scripts\python.exe` (Windows)

### Recommended VS Code Settings

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "editor.formatOnSave": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true
    },
    "python.analysis.typeCheckingMode": "basic",
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "${workspaceFolder}"
    },
    "terminal.integrated.env.osx": {
        "PYTHONPATH": "${workspaceFolder}"
    },
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${workspaceFolder}"
    }
}
```

### Useful VS Code Keyboard Shortcuts

| Action | Windows/Linux | Mac |
|--------|--------------|-----|
| Open Terminal | Ctrl+` | Cmd+` |
| Run Python File | Ctrl+F5 | Cmd+F5 |
| Debug | F5 | F5 |
| Command Palette | Ctrl+Shift+P | Cmd+Shift+P |
| File Explorer | Ctrl+Shift+E | Cmd+Shift+E |
| Search | Ctrl+Shift+F | Cmd+Shift+F |

---

## Configuration

### Directory Structure

```
CodebaseIntelligence/
├── parsers/                    # Parsers for all platforms
│   ├── abinitio/              # Ab Initio parser (FAWN-based)
│   │   ├── parser.py          # Main parser
│   │   ├── mp_file_parser.py  # FAWN-based MP parser
│   │   └── patterns.yaml      # Parser patterns
│   ├── hadoop/                # Hadoop parser (enhanced)
│   │   └── oozie_parser.py    # Sub-workflow support
│   └── databricks/            # Databricks parser
├── services/                  # Services
│   └── local_search/          # FREE local vector search
│       └── local_search_client.py
├── core/                      # Core logic
│   ├── sttm_generator.py      # STTM generation
│   ├── gap_analyzer.py        # Gap analysis
│   └── models.py              # Data models
├── outputs/                   # Output directory
│   ├── reports/               # Excel reports
│   ├── vector_db/             # Local vector database
│   └── logs/                  # Application logs
├── run_analysis.py            # Main runner
├── requirements.txt           # Python dependencies
└── config/                    # Configuration files
    └── config.yaml            # Main config
```

### Configure Local Paths

Edit `config/config.yaml` or create one:

```yaml
# Codebase Intelligence Platform Configuration

# Input paths (update these for your VM)
abinitio_path: "/path/to/abinitio/files"  # Update this
hadoop_path: "/path/to/hadoop/repository"  # Update this
databricks_path: "/path/to/databricks/notebooks"  # Update this

# Output paths
output_dir: "./outputs/reports"
log_dir: "./outputs/logs"

# Local Search (FREE - no API keys needed!)
use_local_search: true
vector_db_path: "./outputs/vector_db"

# Optional: Azure AI Search (if you have access later)
azure_search_endpoint: ""  # Leave empty to use FREE local search
azure_search_key: ""        # Leave empty to use FREE local search

# Optional: OpenAI for RAG chatbot
openai_api_key: ""  # Leave empty if not available

# Logging
log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR

# Performance
max_workers: 4  # Number of parallel workers
chunk_size: 1000  # Chunk size for large files
```

---

## Usage Guide

### Method 1: Run from VS Code Terminal

1. **Open Terminal in VS Code:**
   - Press **Ctrl+`** (or **Cmd+`** on Mac)

2. **Activate Virtual Environment** (if not already activated):
   ```bash
   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Run Parser:**

#### Parse Ab Initio Files (with CLEAN output!)
```bash
python3 run_analysis.py \
    --abinitio-path "/path/to/Ab-Initio" \
    --mode parse \
    --output-dir "./outputs/reports"
```

**Output:**
- `outputs/reports/AbInitio_Parsed_Output.xlsx`
- 4 sheets: DataSet, Component&Fields, GraphParameters (CLEAN!), GraphFlow

#### Parse Hadoop Repository (with sub-workflow resolution!)
```bash
python3 run_analysis.py \
    --hadoop-path "/path/to/hadoop/repo" \
    --mode parse \
    --output-dir "./outputs/reports"
```

**Output:**
- `outputs/reports/Hadoop_Parsed_Output.json`
- Includes resolved sub-workflows and variables

#### Full Analysis (Parse + STTM + Gap Analysis)
```bash
python3 run_analysis.py \
    --abinitio-path "/path/to/Ab-Initio" \
    --hadoop-path "/path/to/hadoop/repo" \
    --databricks-path "/path/to/databricks/notebooks" \
    --mode full \
    --output-dir "./outputs/reports"
```

**Output:**
- All parsed outputs
- STTM mappings
- Gap analysis report
- Combined Excel report

### Method 2: Run via VS Code Run Button

1. Open `run_analysis.py` in VS Code
2. Click the **▶️ Run** button in top-right corner
3. View output in Terminal panel

### Method 3: Create Launch Configuration

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Parse Ab Initio",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/run_analysis.py",
            "args": [
                "--abinitio-path", "/path/to/Ab-Initio",
                "--mode", "parse",
                "--output-dir", "./outputs/reports"
            ],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Full Analysis",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/run_analysis.py",
            "args": [
                "--abinitio-path", "/path/to/Ab-Initio",
                "--hadoop-path", "/path/to/hadoop",
                "--databricks-path", "/path/to/databricks",
                "--mode", "full",
                "--output-dir", "./outputs/reports"
            ],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        }
    ]
}
```

Then press **F5** to run with debugging!

---

## Features

### 1. FAWN-Based Ab Initio Parser

**What it does:**
- Extracts components, parameters, and flows from .mp files
- Produces **CLEAN, READABLE** parameter name/value pairs
- No more messy `@@{{` or `{}` symbols!

**Example Output:**

GraphParameters sheet:
| Graph | Parameter | Value |
|-------|-----------|-------|
| 400_commGenIpa | userID | $(id -un) |
| 400_commGenIpa | referralDate | 2015-02-01 |
| 400_commGenIpa | inputDmlPath | $PUB_ESCAN_DML |

**How to use:**
```bash
python3 run_analysis.py --abinitio-path "/path/to/Ab-Initio" --mode parse
```

### 2. Enhanced Hadoop Parser

**What it does:**
- Parses Oozie workflows
- **Follows sub-workflows** (solves empty input/output issue!)
- **Resolves variables** like `${table}`
- Extracts actual table names from passed properties

**Example:**
```
Before: input_sources: []
After: input_sources: ["patientacctspayercob", "hospitaldata"]
```

**How to use:**
```bash
python3 run_analysis.py --hadoop-path "/path/to/hadoop" --mode parse
```

### 3. GraphFlow Extraction

**What it does:**
- Maps data flows between components
- Creates source-to-target lineage
- Generates GraphFlow sheet in Excel

**How to use:**
GraphFlow is automatically extracted during Ab Initio parsing.

### 4. FREE Local Vector Search

**What it does:**
- **NO Azure AI Search required!**
- Uses ChromaDB (local vector database)
- Uses sentence-transformers (free embeddings)
- Works 100% offline after initial model download

**How to use:**
```python
from services.local_search import LocalSearchClient

# Initialize (FREE!)
client = LocalSearchClient()

# Index documents
docs = [{"content": "Ab Initio parser docs", "metadata": {...}}]
client.index_documents(docs)

# Search
results = client.search("How to parse .mp files?", top=5)
```

### 5. STTM Generator

**What it does:**
- Creates Source-to-Target column mappings
- Extracts transformation logic
- Generates Excel report with mappings

**How to use:**
```bash
python3 run_analysis.py \
    --abinitio-path "/path/to/Ab-Initio" \
    --hadoop-path "/path/to/hadoop" \
    --mode sttm
```

### 6. Gap Analyzer

**What it does:**
- Identifies missing components
- Compares source vs target
- Provides migration recommendations

**How to use:**
```bash
python3 run_analysis.py \
    --abinitio-path "/path/to/Ab-Initio" \
    --hadoop-path "/path/to/hadoop" \
    --mode gap
```

---

## Troubleshooting

### Issue 1: "python3 not found"

**Solution:**
```bash
# Windows: Use 'python' instead of 'python3'
python --version
python run_analysis.py ...

# Linux/Mac: Create alias
alias python=python3
```

### Issue 2: "Module not found" errors

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue 3: "Permission denied" when creating outputs

**Solution:**
```bash
# Create output directories manually
mkdir -p outputs/reports outputs/logs outputs/vector_db

# Fix permissions (Linux/Mac)
chmod -R 755 outputs/
```

### Issue 4: Slow performance

**Solution:**
```bash
# Reduce max_workers in config.yaml
max_workers: 2  # Instead of 4

# Use smaller chunk_size
chunk_size: 500  # Instead of 1000

# Disable logging in production
log_level: "WARNING"  # Instead of "DEBUG"
```

### Issue 5: Excel files won't open

**Solution:**
- Install Excel or LibreOffice Calc
- Or use Python to read:
```python
import pandas as pd
df = pd.read_excel("output.xlsx", sheet_name="GraphParameters")
print(df.head())
```

### Issue 6: Out of memory errors

**Solution:**
```bash
# Process files one at a time instead of all together
python3 run_analysis.py --abinitio-path "/path/to/single/file.mp" --mode parse

# Or increase VM RAM
# Or use pagination in config:
enable_pagination: true
page_size: 100
```

---

## Performance Optimization

### For Large Codebases

1. **Use Parallel Processing:**
   ```yaml
   # config.yaml
   max_workers: 8  # Use more cores
   enable_parallel: true
   ```

2. **Enable Caching:**
   ```yaml
   enable_caching: true
   cache_dir: "./outputs/cache"
   ```

3. **Process in Batches:**
   ```bash
   # Process directory by directory
   for dir in $(ls /path/to/Ab-Initio); do
       python3 run_analysis.py --abinitio-path "/path/to/Ab-Initio/$dir" --mode parse
   done
   ```

### For Low-Resource VMs

1. **Reduce Memory Usage:**
   ```yaml
   max_workers: 2
   chunk_size: 100
   enable_streaming: true
   ```

2. **Use Disk-Based Processing:**
   ```yaml
   use_disk_storage: true
   temp_dir: "/tmp/codebase_intel"
   ```

---

## Support & Contact

### Getting Help

1. **Check Logs:**
   ```bash
   tail -f outputs/logs/app.log
   ```

2. **Enable Debug Mode:**
   ```bash
   python3 run_analysis.py --log-level DEBUG ...
   ```

3. **Review Documentation:**
   - `SEARCH_SETUP_GUIDE.md` - Local search setup
   - `FAWN_PARSER_SUCCESS.md` - Ab Initio parser details
   - `SESSION_COMPLETE_SUMMARY.md` - Feature overview

### Common Questions

**Q: Do I need internet access?**
A: Only for initial setup (pip install). After that, everything works offline!

**Q: Do I need API keys?**
A: No! Local search is 100% free. OpenAI is optional for RAG chatbot.

**Q: Can I use this on Windows?**
A: Yes! Fully supported on Windows, Linux, and macOS.

**Q: How long does parsing take?**
A: Depends on codebase size:
- Small (10-50 files): 1-5 minutes
- Medium (50-500 files): 10-30 minutes
- Large (500+ files): 1-2 hours

**Q: Where are outputs saved?**
A: `outputs/reports/` directory by default.

---

## Quick Start Checklist

- [ ] Python 3.9+ installed
- [ ] VS Code installed with Python extension
- [ ] Project copied to VM
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Paths configured in `config/config.yaml`
- [ ] Test run completed successfully
- [ ] Output Excel files verified

---

## Next Steps

After successful setup:

1. **Parse your Ab Initio files** to get clean Excel outputs
2. **Parse your Hadoop repository** with sub-workflow resolution
3. **Generate STTM** for column-level mappings
4. **Run gap analysis** to identify missing components
5. **Use local search** for RAG chatbot queries

---

**Happy Analyzing! 🚀**

For issues or questions, check the logs in `outputs/logs/app.log` or review the troubleshooting section above.
