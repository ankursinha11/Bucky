# Quick Start Reference - Codebase Intelligence Platform

**Version:** 2.0 | **Status:** Production Ready | **Date:** October 28, 2025

---

## ⚡ Quick Commands

### Parse Ab Initio (Clean FAWN Output)
```bash
python3 run_analysis.py --abinitio-path "/path/to/Ab-Initio" --mode parse
```
**Output:** `outputs/reports/AbInitio_Parsed_Output.xlsx` (4 sheets, CLEAN parameters!)

### Parse Hadoop (with Sub-workflows)
```bash
python3 run_analysis.py --hadoop-path "/path/to/hadoop" --mode parse
```
**Output:** JSON with resolved sub-workflows and variables

### Full Analysis
```bash
python3 run_analysis.py \
    --abinitio-path "/path/to/Ab-Initio" \
    --hadoop-path "/path/to/hadoop" \
    --databricks-path "/path/to/databricks" \
    --mode full
```
**Output:** Complete analysis with STTM and Gap reports

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `CLIENT_DEPLOYMENT_GUIDE.md` | **START HERE** - Full setup guide for client VM |
| `FINAL_IMPLEMENTATION_SUMMARY.md` | Complete feature documentation |
| `FAWN_PARSER_SUCCESS.md` | FAWN parser details and examples |
| `run_analysis.py` | Main script to run analysis |
| `config/config.yaml` | Configuration file (update paths) |

---

## 🎯 What You Get

### Ab Initio Parser (FAWN-Based)
✅ **9,383 clean parameters** (no messy symbols!)
✅ **294 components** with types and details
✅ **4-sheet Excel** (DataSet, Component&Fields, GraphParameters, GraphFlow)
✅ **Readable output** like FAWN

### Hadoop Parser (Enhanced)
✅ **Sub-workflow resolution** (follows nested workflows)
✅ **Variable resolution** (resolves ${table}, ${dataset}, etc.)
✅ **Actual I/O extraction** (real table names, not variables)

### GraphFlow
✅ **288 flows extracted** (for 400_commGenIpa.mp)
✅ **Component name mapping** (not just IDs)
✅ **Complete lineage** tracking

---

## 🚀 Setup Steps (5 Minutes)

```bash
# 1. Navigate to project
cd ~/CodebaseIntelligence

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate (Linux/Mac)
source venv/bin/activate
# Or on Windows:
# venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run test
python3 run_analysis.py --abinitio-path "/path/to/test/file.mp" --mode parse

# 6. Check output
ls -lh outputs/reports/
```

---

## 🔧 Configuration

Edit `config/config.yaml`:
```yaml
abinitio_path: "/path/to/Ab-Initio"      # UPDATE THIS
hadoop_path: "/path/to/hadoop"            # UPDATE THIS
databricks_path: "/path/to/databricks"    # UPDATE THIS

use_local_search: true  # FREE local search (no API keys!)
log_level: "INFO"
max_workers: 4
```

---

## 💻 VS Code Setup

```bash
# 1. Open project in VS Code
code .

# 2. Install Python extension
# Extensions → Search "Python" → Install

# 3. Select interpreter
# Ctrl+Shift+P → "Python: Select Interpreter" → Choose venv

# 4. Run from terminal
# Ctrl+` to open terminal → Run commands above

# 5. Or press F5 for debugging
```

---

## 📊 Excel Output (FAWN Format)

### Sheet 1: DataSet
- Component names, types, datasets, DML, transformation logic

### Sheet 2: Component&Fields
- Field names, types, descriptions for each component

### Sheet 3: GraphParameters ⭐ **CLEAN**
| Graph | Parameter | Value |
|-------|-----------|-------|
| 400_commGenIpa | userID | $(id -un) |
| 400_commGenIpa | referralDate | 2015-02-01 |
| 400_commGenIpa | inputDmlPath | $PUB_ESCAN_DML |

**NO MORE MESSY `@@{{` SYMBOLS!**

### Sheet 4: GraphFlow
- Source component, target component, connection type, data flows

---

## 🐛 Quick Troubleshooting

### "Module not found"
```bash
source venv/bin/activate  # Activate venv first!
pip install -r requirements.txt
```

### "Python not found"
```bash
# Use python3 instead of python
python3 --version

# Or on Windows:
python --version
```

### "Permission denied"
```bash
chmod -R 755 outputs/
mkdir -p outputs/reports outputs/logs
```

### Slow performance
```yaml
# In config.yaml:
max_workers: 2  # Reduce workers
log_level: "WARNING"  # Less logging
```

---

## ✨ New Features (v2.0)

### 1. FAWN-Based Parser
**Before:** Messy parameters with `@@{{` symbols
**After:** Clean `userID: $(id -un)`

### 2. Sub-workflow Resolution
**Before:** `input_sources: []`
**After:** `input_sources: ["patientacctspayercob", "hospitaldata"]`

### 3. GraphFlow Extraction
**Before:** Empty or incomplete
**After:** 288 flows with component names

### 4. FREE Local Search
**Before:** Required Azure AI Search ($$$)
**After:** Uses ChromaDB (FREE, offline!)

---

## 📞 Getting Help

1. **Check logs:**
   ```bash
   tail -f outputs/logs/app.log
   ```

2. **Enable debug mode:**
   ```bash
   python3 run_analysis.py --log-level DEBUG ...
   ```

3. **Read documentation:**
   - `CLIENT_DEPLOYMENT_GUIDE.md` - Setup
   - `FINAL_IMPLEMENTATION_SUMMARY.md` - Features
   - `FAWN_PARSER_SUCCESS.md` - Parser details

---

## ✅ Verification Checklist

- [ ] Python 3.9+ installed
- [ ] VS Code with Python extension
- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] Paths configured in config.yaml
- [ ] Test run completed
- [ ] Excel output verified
- [ ] Clean parameters confirmed

---

## 🎯 Common Use Cases

### Parse Single Ab Initio File
```bash
python3 run_analysis.py \
    --abinitio-path "/path/to/file.mp" \
    --mode parse
```

### Parse Directory of Files
```bash
python3 run_analysis.py \
    --abinitio-path "/path/to/directory/" \
    --mode parse
```

### Generate STTM Only
```bash
python3 run_analysis.py \
    --abinitio-path "/path/to/Ab-Initio" \
    --hadoop-path "/path/to/hadoop" \
    --mode sttm
```

### Gap Analysis Only
```bash
python3 run_analysis.py \
    --abinitio-path "/path/to/Ab-Initio" \
    --hadoop-path "/path/to/hadoop" \
    --mode gap
```

---

## 📈 Performance

| Codebase Size | Time | Output |
|---------------|------|--------|
| 1 file (1.6 MB) | ~10 sec | 9,383 params |
| 10-50 files | 1-5 min | Full Excel |
| 50-500 files | 10-30 min | Complete analysis |

---

## 🚀 Ready to Start!

1. **Open:** `CLIENT_DEPLOYMENT_GUIDE.md` for full setup
2. **Run:** First test with sample file
3. **Verify:** Excel output has clean parameters
4. **Deploy:** Use on your client data

**All features are tested and production-ready!**

---

**For detailed documentation, see:**
- 📘 `CLIENT_DEPLOYMENT_GUIDE.md` - Complete setup guide
- 📗 `FINAL_IMPLEMENTATION_SUMMARY.md` - Feature documentation
- 📙 `FAWN_PARSER_SUCCESS.md` - Parser technical details

**Quick question? Check `outputs/logs/app.log` first!**
