# RAG Chatbot Guide

**Version:** 2.0
**Status:** ✅ Fully Integrated with FREE Local Search

---

## Overview

The RAG (Retrieval-Augmented Generation) Chatbot lets you ask natural language questions about your codebase:

- ✅ **FREE Local Search** using ChromaDB + sentence-transformers
- ✅ **Optional Azure OpenAI** for AI-powered answers
- ✅ **Works Offline** after initial setup
- ✅ **Automatic Indexing** from parsed codebase
- ✅ **Interactive CLI** for easy querying

---

## Quick Start

### Step 1: Index Your Codebase

```bash
# Index Ab Initio files
python3 index_codebase.py --abinitio-path "/path/to/Ab-Initio"

# Index Hadoop repository
python3 index_codebase.py --hadoop-path "/path/to/hadoop"

# Index multiple systems
python3 index_codebase.py \
    --abinitio-path "/path/to/Ab-Initio" \
    --hadoop-path "/path/to/hadoop" \
    --databricks-path "/path/to/databricks"
```

**Result:** Vector database populated at `./outputs/vector_db/`

### Step 2: Start Chatbot

```bash
# Interactive mode
python3 chatbot_cli.py

# Or ask a single question
python3 chatbot_cli.py "What does the patient matching process do?"
```

---

## How It Works

### Architecture

```
User Question
     ↓
[Chatbot CLI]
     ↓
[RAG Chatbot] ← → [ChromaDB Vector Search] ← [Indexed Codebase]
     ↓              (FREE local search!)
[Azure OpenAI]  (optional)
     ↓
Answer + Sources
```

### Two Modes

#### 1. Search-Only Mode (FREE, No API Keys!)
- Uses ChromaDB for semantic search
- Returns top relevant documents
- **Cost:** $0.00
- **Speed:** Fast (local)

#### 2. RAG Mode (with Azure OpenAI)
- Search + AI-powered answer generation
- Natural language responses
- **Cost:** Azure OpenAI API usage
- **Speed:** Moderate (API calls)

---

## Setup

### FREE Mode (No API Keys)

```bash
# 1. Install dependencies (if not already)
pip install -r requirements.txt

# 2. Index codebase
python3 index_codebase.py --abinitio-path "/path/to/Ab-Initio"

# 3. Run chatbot
python3 chatbot_cli.py
```

**You're done!** Chatbot works in search-only mode.

### RAG Mode (with Azure OpenAI)

```bash
# 1. Set environment variables
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"

# 2. Index codebase (if not already)
python3 index_codebase.py --abinitio-path "/path/to/Ab-Initio"

# 3. Run chatbot (will auto-detect OpenAI config)
python3 chatbot_cli.py
```

**You get AI-powered answers!**

---

## Usage Examples

### Interactive Mode

```bash
$ python3 chatbot_cli.py

╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║           Codebase Intelligence RAG Chatbot               ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

🔧 Initializing chatbot...
✓ Azure OpenAI API key found - RAG mode enabled
✓ Vector database loaded: 1250 documents indexed

Chatbot ready! Type 'help' for commands or 'exit' to quit.

You: What does the 400_commGenIpa process do?

🤔 Thinking...

======================================================================
📝 Answer:
======================================================================
The 400_commGenIpa process is an Ab Initio graph that handles commercial
generation for patient accounts. It processes patient data, generates
EDI communications, and matches partners for claim submissions...

======================================================================
📚 Sources (3 results):
======================================================================

[1] Process: 400_commGenIpa
    System: Ab Initio
    Type: Graph
    ...
    Type: process
    System: abinitio
    Score: 0.892

[2] Component: lkpEDIGenBCBSPartners
    Type: Lookup_File
    ...
    Score: 0.765

----------------------------------------------------------------------
Mode: rag | Confidence: high | Results: 5
======================================================================

You: help

Available Commands:
------------------
  ask <question>       - Ask a question about the codebase
  process <name>       - Get info about a specific process
  sttm <table>         - Find STTM mappings for a table
  gaps                 - Find gaps in the codebase
  stats                - Show chatbot statistics
  clear                - Clear conversation history
  help                 - Show this help message
  exit/quit            - Exit the chatbot

You: stats

📊 Chatbot Statistics:
  Mode: RAG
  OpenAI Configured: True
  Conversation Length: 1 exchanges

  Vector Database:
    Documents: 1250
    Collections: 1

You: exit

👋 Goodbye!
```

### Single Question Mode

```bash
# Ask one question and exit
python3 chatbot_cli.py "What parameters does 400_commGenIpa use?"

# Result is printed and script exits
```

---

## Commands Reference

### General Commands

| Command | Description | Example |
|---------|-------------|---------|
| `ask <question>` | Ask any question | `ask What does this process do?` |
| `<question>` | Direct question (no "ask") | `What are the input sources?` |
| `help` | Show help message | `help` |
| `exit` or `quit` | Exit chatbot | `exit` |
| `clear` | Clear conversation history | `clear` |
| `stats` | Show statistics | `stats` |

### Specialized Commands

| Command | Description | Example |
|---------|-------------|---------|
| `process <name>` | Get process info | `process 400_commGenIpa` |
| `sttm <table>` | Find STTM mappings | `sttm patientacctspayercob` |
| `gaps` | Find gaps | `gaps` |

---

## Indexing Guide

### What Gets Indexed?

When you run `index_codebase.py`, it indexes:

1. **Processes**
   - Process names, descriptions
   - Business functions
   - File paths
   - Parameters

2. **Components**
   - Component names, types
   - Input/output datasets
   - Transformation logic
   - DML definitions

3. **STTM Mappings** (if generated)
   - Source/target columns
   - Transformation rules
   - Data types

4. **Gaps** (if analyzed)
   - Gap descriptions
   - Severity levels
   - Recommendations

### Indexing Options

#### From Codebases (Recommended)
```bash
# Ab Initio only
python3 index_codebase.py --abinitio-path "/path/to/Ab-Initio"

# Hadoop only
python3 index_codebase.py --hadoop-path "/path/to/hadoop"

# All systems
python3 index_codebase.py \
    --abinitio-path "/path/to/Ab-Initio" \
    --hadoop-path "/path/to/hadoop" \
    --databricks-path "/path/to/databricks"

# Custom vector DB location
python3 index_codebase.py \
    --abinitio-path "/path/to/Ab-Initio" \
    --vector-db-path "/custom/path/vector_db"
```

#### From JSON File (Testing)
```bash
python3 index_codebase.py --json-file parsed_output.json
```

### Re-indexing

To re-index (update vector database):

```bash
# Delete old database
rm -rf outputs/vector_db/

# Re-index
python3 index_codebase.py --abinitio-path "/path/to/Ab-Initio"
```

---

## Example Queries

### General Questions
```
- What does the patient matching process do?
- How is data transformed in the lead generation workflow?
- What are the input sources for the CDD process?
- Show me all Hadoop workflows
```

### Process-Specific
```
- process 400_commGenIpa
- Tell me about the delta_lake_workflow process
- What parameters does 400_commGenIpa accept?
```

### STTM Queries
```
- sttm patientacctspayercob
- Show mappings from patientdata to targetdb
- What transformations are applied to the patient_id column?
```

### Gap Analysis
```
- gaps
- What gaps exist between Ab Initio and Hadoop?
- Are there any missing processes?
```

### Data Lineage
```
- Trace data flow for patient_id
- What components use the EDI lookup file?
- Show me the data pipeline for patient matching
```

---

## Configuration

### Environment Variables

```bash
# Optional: Azure OpenAI (for RAG mode)
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"

# Optional: Custom vector DB location
export VECTOR_DB_PATH="./custom/vector_db"
```

### Python Code

```python
from services.rag_chatbot_integrated import CodebaseRAGChatbot

# Initialize chatbot
chatbot = CodebaseRAGChatbot(
    use_local_search=True,  # Use FREE local search
    vector_db_path="./outputs/vector_db",
    openai_api_key="your-key"  # Optional
)

# Query
result = chatbot.query("What does process X do?")
print(result["answer"])
```

---

## Performance

### Indexing Time

| Codebase Size | Time | Documents |
|---------------|------|-----------|
| 10 files | ~30 sec | ~100 docs |
| 50 files | ~2 min | ~500 docs |
| 500 files | ~20 min | ~5000 docs |

### Query Time

| Mode | Time | Cost |
|------|------|------|
| Search-Only | <1 sec | $0 |
| RAG (with OpenAI) | 2-5 sec | ~$0.01 per query |

---

## Troubleshooting

### "Vector database is empty"

**Problem:** No documents indexed

**Solution:**
```bash
# Index your codebase first
python3 index_codebase.py --abinitio-path "/path/to/Ab-Initio"
```

### "OpenAI not installed"

**Problem:** OpenAI package not found

**Solution:**
```bash
pip install openai>=1.0.0

# Or reinstall all dependencies
pip install -r requirements.txt
```

### "No relevant documents found"

**Problem:** Query doesn't match indexed content

**Solutions:**
1. Rephrase your question
2. Check if codebase was indexed correctly
3. Try broader queries

### "Azure OpenAI error"

**Problem:** API key or endpoint incorrect

**Solutions:**
1. Check environment variables are set correctly
2. Verify API key is valid
3. Check endpoint URL format
4. Run in search-only mode (no API key needed)

---

## Comparison: Search-Only vs RAG Mode

### Search-Only Mode (FREE)

**Pros:**
- ✅ Completely FREE
- ✅ Fast (local search)
- ✅ Works offline
- ✅ No API keys needed

**Cons:**
- ❌ Returns raw search results
- ❌ No natural language answers
- ❌ Requires manual interpretation

**When to use:**
- Testing/development
- No OpenAI budget
- Simple lookups

### RAG Mode (with OpenAI)

**Pros:**
- ✅ Natural language answers
- ✅ Contextual understanding
- ✅ Explains complex logic
- ✅ Provides recommendations

**Cons:**
- ❌ Requires Azure OpenAI API key
- ❌ Costs ~$0.01 per query
- ❌ Slightly slower (API calls)

**When to use:**
- Production use
- Business users
- Complex queries
- Need explanations

---

## Integration with Main System

The RAG chatbot is fully integrated with `run_analysis.py`:

```bash
# Run full analysis with indexing
python3 run_analysis.py \
    --abinitio-path "/path/to/Ab-Initio" \
    --hadoop-path "/path/to/hadoop" \
    --mode full

# This will:
# 1. Parse codebases
# 2. Generate STTM
# 3. Run gap analysis
# 4. Index everything into vector database
# 5. Ready for chatbot queries!
```

---

## Next Steps

### After Setup

1. **Index your codebase**
   ```bash
   python3 index_codebase.py --abinitio-path "/path/to/Ab-Initio"
   ```

2. **Try the chatbot**
   ```bash
   python3 chatbot_cli.py
   ```

3. **Ask some questions**
   - Start with simple queries
   - Try specialized commands
   - Explore your codebase!

### Migration to Azure AI Search (Later)

When you have Azure AI Search access:

1. Update `.env` or environment variables:
   ```bash
   export AZURE_SEARCH_ENDPOINT="your-endpoint"
   export AZURE_SEARCH_KEY="your-key"
   ```

2. The system will automatically use Azure AI Search
3. ChromaDB becomes backup/fallback

---

## Summary

✅ **FREE Local Search** - ChromaDB + sentence-transformers
✅ **Optional OpenAI** - Add API key for AI answers
✅ **Easy Indexing** - One command to index codebase
✅ **Interactive CLI** - User-friendly command-line interface
✅ **Production Ready** - Tested and documented

**Get started now:**
```bash
# 1. Index
python3 index_codebase.py --abinitio-path "/path/to/Ab-Initio"

# 2. Chat
python3 chatbot_cli.py

# 3. Ask away!
```

**No API keys required! Works 100% locally!**
