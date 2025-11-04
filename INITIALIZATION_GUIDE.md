# Complete Initialization Guide - CodebaseIntelligence System

**Date:** November 4, 2025
**Status:** Production Ready

---

## Prerequisites

### 1. Python Environment
```bash
python --version  # Should be 3.9+
```

### 2. Required Environment Variables

Create a `.env` file in the project root:

```bash
# Azure OpenAI Configuration (Required for AI features)
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Azure OpenAI Embeddings (Required for vector search)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
```

**Note:** If you don't have Azure OpenAI, the system will work in **search-only mode** (no AI analysis, but vector search still works).

---

## Step-by-Step Initialization

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd /Users/ankurshome/Desktop/Hadoop_Parser/CodebaseIntelligence

# Install Python dependencies
pip install -r requirements.txt
```

**Key Dependencies Installed:**
- `streamlit` - Web UI
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings
- `openai` - Azure OpenAI
- `pandas`, `openpyxl` - Data processing
- `pdfplumber`, `python-docx` - Document parsing
- `loguru` - Logging

---

### Step 2: Verify Directory Structure

Ensure these directories exist:

```bash
# Create required directories
mkdir -p outputs/vector_db
mkdir -p outputs/logs
mkdir -p outputs/analysis
mkdir -p data/abinitio
mkdir -p data/autosys
mkdir -p data/hadoop
mkdir -p data/databricks
mkdir -p data/documents
```

---

### Step 3: Index Your Codebase

You have **3 options** for indexing:

#### Option A: Use Interactive Reindexing Script (Recommended)

```bash
# Make script executable
chmod +x reindex.sh

# Run interactive menu
./reindex.sh
```

**Interactive Menu:**
```
========================================
STAG - Interactive Reindexing
========================================

Choose an operation:
1) Re-index Ab Initio
2) Re-index Autosys
3) Re-index Hadoop
4) Re-index Databricks
5) Re-index Documents (PDFs/DOCX)
6) Re-index Everything
7) View Database Status
8) Clear All Collections
9) Exit

Select option [1-9]:
```

**Example: Index Ab Initio**
1. Select option `1`
2. Enter path: `/path/to/your/abinitio/files`
3. Wait for indexing to complete
4. See stats: "✓ Indexed 150 graphs, 500 components"

---

#### Option B: Use Python Indexing Script

```bash
# Index Ab Initio
python index_codebase.py \
  --system abinitio \
  --path /path/to/abinitio/files \
  --collection abinitio_collection

# Index Autosys
python index_codebase.py \
  --system autosys \
  --path /path/to/autosys/jil/files \
  --collection autosys_collection

# Index Hadoop
python index_codebase.py \
  --system hadoop \
  --path /path/to/hadoop/scripts \
  --collection hadoop_collection

# Index Documents
python index_codebase.py \
  --system documents \
  --path /path/to/documentation \
  --collection documents_collection
```

---

#### Option C: Use STAG UI Database Tab

1. **Launch STAG:**
   ```bash
   streamlit run stag_app.py
   ```

2. **Go to "⚙️ Database" tab**

3. **Select operation:** "Re-index Ab Initio"

4. **Enter path** or **upload files**

5. **Click "Start Indexing"**

6. **Watch progress** in real-time

---

### Step 4: Verify Indexing

#### Check Database Status

**Option 1: Via CLI**
```bash
python manage_vector_db.py --status
```

**Output:**
```
Vector Database Statistics:
==========================

Collection: abinitio_collection
  Total documents: 650
  Status: active

Collection: autosys_collection
  Total documents: 230
  Status: active

Collection: hadoop_collection
  Total documents: 120
  Status: active

Collection: sttm_abinitio_collection
  Total documents: 0
  Status: active (empty)

Total documents across all collections: 1000
Database location: ./outputs/vector_db
```

**Option 2: Via STAG UI**
1. Launch STAG: `streamlit run stag_app.py`
2. Go to "⚙️ Database" tab
3. View status automatically displayed

**Option 3: Via Python**
```python
from services.multi_collection_indexer import MultiCollectionIndexer

indexer = MultiCollectionIndexer()
stats = indexer.get_stats()

for collection, data in stats.items():
    print(f"{collection}: {data.get('total_documents', 0)} documents")
```

---

### Step 5: Test the System

#### Test 1: Simple Chat Query

```bash
# Launch STAG
streamlit run stag_app.py
```

1. Click **"💬 Chat"** tab
2. Type: `"What parsers are available?"`
3. Watch the agent-based response with thinking process
4. Should see: Ab Initio Parser, Autosys Parser, etc.

---

#### Test 2: Lineage Tracking

1. Click **"🔗 Lineage"** tab
2. Select system: **Ab Initio**
3. Enter entity name: `"customer_load"` (or any graph you indexed)
4. Click **"Analyze Lineage"**
5. Should see:
   - STTM mappings
   - Autosys context (if Autosys indexed)
   - Column-level lineage
   - AI reasoning

---

#### Test 3: Cross-System Comparison

1. Stay in **"💬 Chat"** tab
2. Type: `"Compare customer_load from Ab Initio with customer_etl from Hadoop"`
3. Watch agent orchestration:
   - Intent detected: COMPARISON
   - ParsingAgent extracts both entities
   - LogicAgent analyzes transformations
   - SimilarityAgent calculates similarity
   - Final comparison report generated

---

## Troubleshooting

### Issue 1: "No Azure OpenAI API key found"

**Solution:**
1. Check `.env` file exists in project root
2. Verify `AZURE_OPENAI_API_KEY` is set
3. Restart STAG after adding .env

**Workaround:**
- System works in search-only mode without API key
- Vector search still functions
- No AI analysis of logic/transformations

---

### Issue 2: "Collection not found"

**Solution:**
```bash
# Re-index the collection
./reindex.sh

# Or via Python
python index_codebase.py --system abinitio --path /path/to/files
```

---

### Issue 3: "No documents found"

**Cause:** Collections are empty

**Solution:**
1. Verify files exist at specified path
2. Check file extensions:
   - Ab Initio: `.mp`, `.dml`, `.ksh`
   - Autosys: `.jil`
   - Hadoop: `.hql`, `.pig`, `.sh`
   - Documents: `.pdf`, `.docx`, `.xlsx`
3. Re-run indexing

---

### Issue 4: "Memory error during indexing"

**Solution:**
```bash
# Index in batches
python index_codebase.py --system abinitio --path /path/to/files --batch-size 50

# Or clear and re-index
python manage_vector_db.py --clear all
./reindex.sh
```

---

### Issue 5: "ChromaDB error"

**Solution:**
```bash
# Delete and recreate vector DB
rm -rf outputs/vector_db
mkdir -p outputs/vector_db

# Re-index
./reindex.sh
```

---

## Directory Structure After Initialization

```
CodebaseIntelligence/
├── .env                          # ← Environment variables
├── stag_app.py                   # ← Main STAG UI
├── requirements.txt
├── outputs/
│   ├── vector_db/                # ← ChromaDB storage
│   │   ├── chroma.sqlite3
│   │   ├── abinitio_collection/
│   │   ├── autosys_collection/
│   │   ├── hadoop_collection/
│   │   ├── sttm_abinitio_collection/  # ← STTM auto-indexed here
│   │   └── ...
│   ├── logs/                     # ← Application logs
│   └── analysis/                 # ← Generated reports
├── data/                         # ← Your source files
│   ├── abinitio/
│   ├── autosys/
│   ├── hadoop/
│   └── documents/
├── services/
│   ├── multi_collection_indexer.py
│   ├── ai_script_analyzer.py
│   ├── chat/
│   │   ├── chat_orchestrator.py
│   │   └── query_classifier.py
│   └── lineage/
│       ├── lineage_agents.py
│       └── sttm_generator.py
└── ui/
    └── lineage_tab.py
```

---

## Quick Start Cheatsheet

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env with Azure OpenAI credentials
cat > .env << EOF
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
EOF

# 3. Create directories
mkdir -p outputs/vector_db data/{abinitio,autosys,hadoop,documents}

# 4. Index codebase
./reindex.sh
# Select option 6 (Re-index Everything)
# Enter paths for each system

# 5. Launch STAG
streamlit run stag_app.py

# 6. Open browser
# http://localhost:8501

# 7. Test it
# Go to Chat tab and ask: "What is available?"
```

---

## Advanced Configuration

### Custom Vector DB Path

```bash
# Edit stag_app.py
# Change this line:
vector_db_path="./outputs/vector_db"

# To:
vector_db_path="/custom/path/to/vector_db"
```

### Custom OpenAI Model

```bash
# In .env
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o  # Use GPT-4o instead
```

### Increase Context Window

```python
# In services/ai_script_analyzer.py
# Change this line:
script_content = script_logic.raw_content[:4000]

# To:
script_content = script_logic.raw_content[:8000]  # Larger context
```

---

## Verification Checklist

After initialization, verify these work:

- [ ] STAG UI launches without errors
- [ ] Database tab shows collection statistics
- [ ] Chat tab responds to queries
- [ ] Lineage tab can analyze entities
- [ ] STTM auto-indexing works
- [ ] Autosys context appears for Ab Initio graphs
- [ ] Agent orchestration shows thinking process
- [ ] Export functions work (Excel, JSON)

---

## Database Management Commands

### View Status
```bash
python manage_vector_db.py --status
```

### Clear Specific Collection
```bash
python manage_vector_db.py --clear abinitio
```

### Clear Everything
```bash
python manage_vector_db.py --clear all
```

### Re-index
```bash
python manage_vector_db.py --reindex abinitio --path /path/to/files
```

### Export Stats
```bash
python manage_vector_db.py --export stats.json
```

---

## Performance Tips

### 1. Index Incrementally
```bash
# Instead of indexing 1000 files at once
# Do batches of 100-200
python index_codebase.py --system abinitio --path /path/batch1 --batch-size 100
python index_codebase.py --system abinitio --path /path/batch2 --batch-size 100
```

### 2. Use SSD for Vector DB
```bash
# Store vector DB on fast SSD
mkdir -p /fast/ssd/vector_db
ln -s /fast/ssd/vector_db ./outputs/vector_db
```

### 3. Increase Memory
```bash
# Set environment variable for larger memory
export CHROMADB_MAX_MEMORY=4GB
```

---

## Next Steps After Initialization

1. **Index Your Actual Codebase**
   - Point indexer to real Ab Initio/Hadoop/Databricks files
   - Index Autosys JIL files
   - Index documentation (PDFs, DOCX)

2. **Test Lineage Tracking**
   - Pick an Ab Initio graph
   - Run lineage analysis
   - Verify STTM auto-indexed
   - Check Autosys context

3. **Try Agent Chat**
   - Ask comparison questions
   - Request lineage traces
   - Search for patterns

4. **Export Results**
   - Export STTM to Excel
   - Save lineage JSON
   - Generate reports

---

## Support

### Documentation
- **STAG Guide:** `STAG_README.md`
- **Agent Chat:** `AGENT_CHAT_GUIDE.md`
- **Lineage:** `LINEAGE_TRACKING_GUIDE.md`
- **Quick Commands:** `QUICK_REFERENCE.md`
- **STTM & Autosys:** `STTM_AND_AUTOSYS_CONTEXT_ENHANCEMENTS.md`

### Logs
Check logs for errors:
```bash
tail -f outputs/logs/stag.log
```

### Reset Everything
```bash
# Complete fresh start
rm -rf outputs/vector_db
mkdir -p outputs/vector_db
./reindex.sh
```

---

## Common Workflows

### Workflow 1: Daily Incremental Index

```bash
#!/bin/bash
# daily_index.sh

# Index new files only
python index_codebase.py \
  --system abinitio \
  --path /path/to/new/files \
  --incremental

python index_codebase.py \
  --system autosys \
  --path /path/to/new/jils \
  --incremental
```

### Workflow 2: Migration Analysis

```bash
# 1. Index source system (Ab Initio)
./reindex.sh  # Option 1: Ab Initio

# 2. Index target systems (Hadoop, Databricks)
./reindex.sh  # Option 3: Hadoop
./reindex.sh  # Option 4: Databricks

# 3. Launch STAG and use Chat
streamlit run stag_app.py
# Ask: "Find Ab Initio graphs similar to Hadoop workflows"
```

### Workflow 3: PII Audit

```bash
# 1. Index all systems
./reindex.sh  # Option 6: Everything

# 2. Use Python to search PII
python << EOF
from services.multi_collection_indexer import MultiCollectionIndexer

indexer = MultiCollectionIndexer()

# Search for PII in STTM
pii_fields = indexer.search_sttm(
    query="",
    filters={"contains_pii": True}
)

print(f"Found {len(pii_fields)} PII fields")
for field in pii_fields:
    print(f"- {field['metadata']['target_field']} in {field['metadata']['entity_name']}")
EOF
```

---

## Status Check

After following this guide, run this verification:

```bash
python << EOF
from services.multi_collection_indexer import MultiCollectionIndexer
from services.ai_script_analyzer import AIScriptAnalyzer
from services.chat.chat_orchestrator import create_chat_orchestrator

# Check indexer
print("Checking MultiCollectionIndexer...")
indexer = MultiCollectionIndexer()
stats = indexer.get_stats()
print(f"✓ Found {len(stats)} collections")

# Check AI analyzer
print("\nChecking AI Analyzer...")
ai_analyzer = AIScriptAnalyzer()
print(f"✓ AI enabled: {ai_analyzer.enabled}")

# Check chat orchestrator
print("\nChecking Chat Orchestrator...")
orchestrator = create_chat_orchestrator(ai_analyzer, indexer)
print("✓ Chat orchestrator initialized")

print("\n🎉 All systems operational!")
EOF
```

---

**Status:** ✅ READY TO USE

**Your system is now fully initialized and ready for production use!**

Launch STAG and start analyzing your codebase:
```bash
streamlit run stag_app.py
```

Then go to http://localhost:8501 and explore! 🚀
