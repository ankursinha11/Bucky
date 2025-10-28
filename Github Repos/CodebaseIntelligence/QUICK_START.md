# CodebaseIntelligence - Quick Start

**One project that works on Windows, Mac, and Linux**

---

## ⚡ 3-Step Installation

### Windows (including Azure Virtual Desktop)
```cmd
1. Copy CodebaseIntelligence folder to: C:\Projects\CodebaseIntelligence
2. cd C:\Projects\CodebaseIntelligence
3. setup_windows.bat
```

### Mac/Linux
```bash
1. Copy CodebaseIntelligence folder to your location
2. cd /path/to/CodebaseIntelligence
3. python3 -m venv venv && source venv/bin/activate && pip install -r requirements-minimal.txt
```

---

## ✅ Verify Installation

```bash
python verify_parser_version.py
```

Should show all ✓ checkmarks

---

## 🚀 Index Your First Repository

**Hadoop:**
```bash
python index_codebase.py --parser hadoop --source "/path/to/hadoop"
```

**Ab Initio:**
```bash
python index_codebase.py --parser abinitio --source "/path/to/abinitio"
```

**Databricks:**
```bash
python index_codebase.py --parser databricks --source "/path/to/databricks"
```

---

## 💬 Query with Chatbot

```bash
python chatbot_cli.py
```

Try asking:
- "What workflows use the patient_data table?"
- "Show me all Sqoop imports"
- "List all coordinators and their schedules"

---

## 📚 Documentation

- **Quick Start**: This file
- **Full Deployment Guide**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Chatbot Guide**: [RAG_CHATBOT_GUIDE.md](RAG_CHATBOT_GUIDE.md)
- **Main README**: [README.md](README.md)

---

## 🎯 What Makes This Cross-Platform?

✅ **Path handling**: Works with `/` (Unix) and `\` (Windows)
✅ **Batch + Shell scripts**: `.bat` for Windows, `.sh` for Mac/Linux
✅ **Path normalization**: Consistent hashing across platforms
✅ **Tested on**: Windows 10/11, Mac, Linux, Azure Virtual Desktop

---

## 🆘 Need Help?

1. Run: `python verify_parser_version.py`
2. Check: `outputs/` directory for logs
3. Read: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

**That's it! One folder, works everywhere.**
