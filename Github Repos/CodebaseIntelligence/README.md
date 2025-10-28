# Codebase Intelligence Platform

**Version 2.0** - Production Ready
Healthcare Finance Data Pipeline Analysis & Migration Tool

---

## Overview

Platform for analyzing legacy codebases (Ab Initio, Hadoop, Databricks) to generate:

- ✅ **Clean FAWN-Style Excel Reports** with readable parameters
- ✅ **Source-to-Target Mappings (STTM)** at column level
- ✅ **Gap Analysis** with migration recommendations
- ✅ **GraphFlow Extraction** for complete data lineage
- ✅ **FREE Local Vector Search** (no Azure AI Search required!)

---

## Key Features

### 1. FAWN-Based Ab Initio Parser
- Extracts **clean, readable parameters** (no messy symbols!)
- Generates 4-sheet Excel: DataSet, Component&Fields, GraphParameters, GraphFlow
- Example output: `userID: $(id -un)` instead of raw data
- **Result:** 9,383+ clean parameters per file

### 2. Enhanced Hadoop Parser
- **Follows sub-workflows** automatically
- **Resolves variables** like `${table}`, `${dataset}`
- Extracts actual table names from properties
- **Result:** No more empty `input_sources: []`

### 3. GraphFlow Data Lineage
- 288+ flows per graph with component name mapping
- Complete source-to-target relationships
- Excel GraphFlow sheet ready for analysis

### 4. FREE Local Search & RAG Chatbot
- ✅ **RAG Chatbot** for natural language queries about your codebase
- ✅ **FREE Local Search** using ChromaDB + sentence-transformers
- ✅ **Optional Azure OpenAI** for AI-powered answers
- ✅ **Interactive CLI** - Ask questions, get instant answers
- No API keys required for search-only mode!
- Works 100% offline after setup

---

## Quick Start

### Installation (5 Minutes)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Done! Ready to parse
```

### Parse Ab Initio Files (Clean Output!)

```bash
python3 run_analysis.py \
    --abinitio-path "/path/to/Ab-Initio" \
    --mode parse \
    --output-dir "./outputs/reports"
```

**Output:** `outputs/reports/AbInitio_Parsed_Output.xlsx`
- 4 sheets with CLEAN, readable data
- NO messy `@@{{` symbols!
- GraphParameters sheet: clean name/value pairs

### Parse Hadoop (with Sub-workflows)

```bash
python3 run_analysis.py \
    --hadoop-path "/path/to/hadoop" \
    --mode parse
```

**Output:** JSON with:
- Followed sub-workflows
- Resolved variables
- Actual table names

### Full Analysis

```bash
python3 run_analysis.py \
    --abinitio-path "/path/to/Ab-Initio" \
    --hadoop-path "/path/to/hadoop" \
    --databricks-path "/path/to/databricks" \
    --mode full
```

**Output:**
- All parsed data
- STTM mappings
- Gap analysis report
- Combined Excel reports

### Deep Parsing with AI (RECOMMENDED!)

For **intelligent** chatbot answers, use deep parsing to extract script logic, transformations, and business rules:

```bash
# Deep parse Hadoop with AI analysis
python3 index_codebase.py --parser hadoop --source "/path/to/hadoop" --deep

# Deep parse Databricks notebooks
python3 index_codebase.py --parser databricks --source "/path/to/notebooks" --deep

# Deep parse Ab Initio graphs
python3 index_codebase.py --parser abinitio --source "/path/to/abinitio" --deep

# Deep parse without AI (faster, but less intelligent)
python3 index_codebase.py --parser hadoop --source "/path/to/hadoop" --deep --no-ai
```

**What Deep Parsing Does:**
- ✅ **3-Tier Analysis:** Repository → Workflows → Scripts
- ✅ **Script Logic Extraction:** FILTER, JOIN, GROUP BY, aggregations
- ✅ **Column-Level Lineage:** Track how data flows through transformations
- ✅ **AI Business Logic:** GPT-4 explains what scripts do and why
- ✅ **Flow Diagrams:** ASCII + Mermaid.js visualizations
- ✅ **Smart Search:** 10x more intelligent chatbot answers!

**Result:** Chatbot can answer complex questions like:
- "What business rules are applied in the patient matching script?"
- "How does data flow through the enrollment workflow?"
- "Which scripts filter out invalid claims?"

### RAG Chatbot (Ask Questions!)

```bash
# Step 1: Index your codebase (use --deep for best results!)
python3 index_codebase.py --parser hadoop --source "/path/to/hadoop" --deep

# Step 2: Start chatbot
python3 chatbot_cli.py

# Or ask a single question
python3 chatbot_cli.py "What does the patient matching process do?"
```

**Features:**
- Ask questions in natural language
- Search through parsed codebase
- Get AI-powered answers (optional Azure OpenAI)
- FREE local search (no API keys needed!)
- **Use deep parsing for 10x smarter answers!**

---

## Project Structure

```
CodebaseIntelligence/
├── parsers/
│   ├── abinitio/              # FAWN-based parser (CLEAN output!)
│   ├── hadoop/                # Enhanced with sub-workflow resolution
│   └── databricks/            # Databricks notebook parser
├── core/
│   ├── sttm_generator.py      # STTM generation
│   ├── gap_analyzer.py        # Gap analysis
│   └── models.py              # Data models
├── services/
│   └── local_search/          # FREE local vector search
├── outputs/
│   ├── reports/               # Excel reports
│   ├── vector_db/             # Local vector database
│   └── logs/                  # Application logs
├── run_analysis.py            # Main runner script
├── requirements.txt           # Python dependencies
└── config/
    └── config.yaml            # Configuration
```

---

## Documentation

### Essential Guides

1. **[CLIENT_DEPLOYMENT_GUIDE.md](CLIENT_DEPLOYMENT_GUIDE.md)** ⭐ **START HERE**
   - Complete setup for Windows/Linux/Mac
   - VS Code integration
   - Step-by-step installation
   - Troubleshooting

2. **[QUICK_START_REFERENCE.md](QUICK_START_REFERENCE.md)**
   - Quick commands
   - Common use cases
   - Performance tips

3. **[FINAL_IMPLEMENTATION_SUMMARY.md](FINAL_IMPLEMENTATION_SUMMARY.md)**
   - Complete feature documentation
   - Test results
   - Before/after comparisons
   - Technical details

---

## Configuration

Edit `config/config.yaml`:

```yaml
# Input paths (UPDATE THESE!)
abinitio_path: "/path/to/abinitio/files"
hadoop_path: "/path/to/hadoop/repository"
databricks_path: "/path/to/databricks/notebooks"

# Output
output_dir: "./outputs/reports"

# FREE Local Search (no API keys needed!)
use_local_search: true
vector_db_path: "./outputs/vector_db"

# Performance
max_workers: 4
log_level: "INFO"
```

---

## Output Examples

### Ab Initio Excel (4 Sheets)

**Sheet 1: DataSet**
- 294 components with types and details

**Sheet 2: Component&Fields**
- Key fields for each component

**Sheet 3: GraphParameters** ⭐ **CLEAN**
| Graph | Parameter | Value |
|-------|-----------|-------|
| 400_commGenIpa | userID | $(id -un) |
| 400_commGenIpa | referralDate | 2015-02-01 |
| 400_commGenIpa | inputDmlPath | $PUB_ESCAN_DML |

**NO MORE MESSY SYMBOLS!**

**Sheet 4: GraphFlow**
- 288+ data flows with component names
- Source-to-target relationships

---

## Performance

| Codebase Size | Time | Output |
|---------------|------|--------|
| 1 file (1.6 MB) | ~10 sec | 9,383 params |
| 10-50 files | 1-5 min | Full Excel |
| 50-500 files | 10-30 min | Complete analysis |

**Memory:** ~70 MB per file
**Platform:** Windows, Linux, macOS

---

## Troubleshooting

### "Module not found"
```bash
# Activate virtual environment first
source venv/bin/activate
pip install -r requirements.txt
```

### "Permission denied"
```bash
mkdir -p outputs/reports outputs/logs
chmod -R 755 outputs/
```

### Slow performance
Edit `config/config.yaml`:
```yaml
max_workers: 2      # Reduce workers
log_level: "WARNING"  # Less logging
```

**More help:** See [CLIENT_DEPLOYMENT_GUIDE.md](CLIENT_DEPLOYMENT_GUIDE.md) → Troubleshooting section

---

## What's New in v3.0

### ✅ Deep Parsing with AI
**NEW:** 3-tier intelligent analysis
- Repository-level architecture understanding
- Workflow execution flow mapping
- Script-level logic extraction (FILTER, JOIN, GROUP BY)
- Column-level data lineage
- AI-powered business logic understanding
- **Result:** 10x smarter chatbot answers!

### ✅ Smart Gap Analyzer
**NEW:** Automated cross-system comparison
- Compare Hadoop ↔ Databricks ↔ Ab Initio
- Detect missing workflows and transformations
- Identify data quality gaps
- Migration complexity assessment

### ✅ FAWN-Based Parser
**Before:** Messy parameters with `@@{{` symbols
**After:** Clean `userID: $(id -un)`

### ✅ Sub-workflow Resolution
**Before:** `input_sources: []`
**After:** `input_sources: ["patientacctspayercob", "hospitaldata"]`

### ✅ GraphFlow Extraction
**Before:** Empty or incomplete
**After:** 288 flows with component names

### ✅ FREE Local Search
**Before:** Required Azure AI Search ($$$)
**After:** Uses ChromaDB (FREE, offline!)

---

## VS Code Integration

```bash
# 1. Open project
code .

# 2. Install Python extension
# Extensions (Ctrl+Shift+X) → Search "Python" → Install

# 3. Select interpreter
# Ctrl+Shift+P → "Python: Select Interpreter" → ./venv/bin/python

# 4. Run with F5 for debugging!
```

See [CLIENT_DEPLOYMENT_GUIDE.md](CLIENT_DEPLOYMENT_GUIDE.md) for detailed VS Code setup.

---

## System Requirements

- **Python:** 3.9+
- **RAM:** 8 GB minimum (16 GB recommended)
- **Disk:** 10 GB free
- **OS:** Windows 10+, Linux, macOS
- **Network:** Optional (only for initial setup)

---

## Getting Help

1. **Check logs:**
   ```bash
   tail -f outputs/logs/app.log
   ```

2. **Read documentation:**
   - [CLIENT_DEPLOYMENT_GUIDE.md](CLIENT_DEPLOYMENT_GUIDE.md) - Setup and deployment
   - [QUICK_START_REFERENCE.md](QUICK_START_REFERENCE.md) - Quick commands
   - [FINAL_IMPLEMENTATION_SUMMARY.md](FINAL_IMPLEMENTATION_SUMMARY.md) - Complete docs

3. **Enable debug mode:**
   ```bash
   python3 run_analysis.py --log-level DEBUG ...
   ```

---

## Status

✅ **Production Ready**
- All features implemented and tested
- Clean FAWN-style output verified
- Sub-workflow resolution working
- GraphFlow extraction complete
- Documentation comprehensive

---

## Next Steps

1. **Review documentation:** [CLIENT_DEPLOYMENT_GUIDE.md](CLIENT_DEPLOYMENT_GUIDE.md)
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Update config:** Edit `config/config.yaml` with your paths
4. **Run first analysis:** Parse an Ab Initio file to see clean output
5. **Deploy to client VM:** Follow CLIENT_DEPLOYMENT_GUIDE.md

---

**Ready to analyze your codebase!** 🚀

For deployment to client VM, start with: **[CLIENT_DEPLOYMENT_GUIDE.md](CLIENT_DEPLOYMENT_GUIDE.md)**
