# CodebaseIntelligence - Deployment Guide

**Complete guide for deploying on Windows, Mac, Linux, and Azure Virtual Desktop (AVD)**

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Windows Deployment](#windows-deployment)
3. [Mac/Linux Deployment](#maclinux-deployment)
4. [Azure Virtual Desktop (AVD)](#azure-virtual-desktop-avd)
5. [Verification](#verification)
6. [First Index](#first-index)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Step 1: Copy Project

Copy the entire `CodebaseIntelligence` folder to your target machine:

**Windows:**
```
Source: /path/to/CodebaseIntelligence
Target: C:\Projects\CodebaseIntelligence
```

**Mac/Linux:**
```bash
cp -r /path/to/CodebaseIntelligence /your/destination/
```

### Step 2: Run Setup

**Windows:**
```cmd
cd C:\Projects\CodebaseIntelligence
setup_windows.bat
```

**Mac/Linux:**
```bash
cd /path/to/CodebaseIntelligence
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-minimal.txt
```

### Step 3: Verify
```bash
python verify_parser_version.py
```

Should show all ✓ checkmarks.

### Step 4: Start Using
```bash
# Index a repository
python index_codebase.py --parser hadoop --source "/path/to/hadoop"

# Query with chatbot
python chatbot_cli.py
```

---

## 🪟 Windows Deployment

### Prerequisites

1. **Python 3.8+**
   - Download from [python.org](https://www.python.org/downloads/)
   - **Important**: Check "Add Python to PATH" during installation

2. **Check Python Installation:**
   ```cmd
   python --version
   ```
   Should show Python 3.8 or higher

### Installation Steps

1. **Copy Project to Windows**

   Copy entire `CodebaseIntelligence` folder to:
   ```
   C:\Projects\CodebaseIntelligence
   ```
   Or any preferred location (use D: drive on AVD for persistence)

2. **Open Command Prompt**

   Navigate to the project:
   ```cmd
   cd C:\Projects\CodebaseIntelligence
   ```

3. **Run Setup Script**
   ```cmd
   setup_windows.bat
   ```

   This will:
   - Create Python virtual environment
   - Install all dependencies
   - Verify installation automatically

4. **Verify Installation**
   ```cmd
   python verify_parser_version.py
   ```

   Expected output:
   ```
   ✓ parsers/hadoop/parser.py has 'import hashlib'
   ✓ parsers/hadoop/parser.py has file hash generation code
   ✓ parsers/hadoop/parser.py uses file_hash in process ID
   ✓ parsers/hadoop/parser.py uses index in component ID
   ✓ parsers/hadoop/oozie_parser.py has coordinator support
   ✓ core/models/component.py has OOZIE_COORDINATOR type
   ✓ services/codebase_indexer.py has correct ID structure
   ```

### Windows-Specific Features

✅ **Path Separators**: Works with both `/` and `\`
✅ **UNC Paths**: Supports `\\server\share\path`
✅ **Network Drives**: Map with `net use Z: \\server\share`
✅ **Quotes**: Auto-handles paths with spaces
✅ **Long Paths**: Supports 260+ character paths

### Windows Examples

```cmd
REM Local drive
python index_codebase.py --parser hadoop --source "C:\Data\Hadoop"

REM Network UNC path
python index_codebase.py --parser hadoop --source "\\corpserver\hadoop_data"

REM Mapped network drive
net use Z: \\corpserver\hadoop_data
python index_codebase.py --parser hadoop --source "Z:\"

REM Path with spaces
python index_codebase.py --parser hadoop --source "C:\Program Files\Hadoop Data"
```

---

## 🍎 Mac/Linux Deployment

### Prerequisites

1. **Python 3.8+**

   Check installation:
   ```bash
   python3 --version
   ```

2. **pip (usually included)**
   ```bash
   python3 -m pip --version
   ```

### Installation Steps

1. **Navigate to Project**
   ```bash
   cd /path/to/CodebaseIntelligence
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   ```

3. **Activate Virtual Environment**
   ```bash
   source venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements-minimal.txt
   ```

5. **Verify Installation**
   ```bash
   python verify_parser_version.py
   ```

### Mac/Linux Examples

```bash
# Local directory
python index_codebase.py --parser hadoop --source "/Users/yourname/Data/Hadoop"

# Network mount
python index_codebase.py --parser hadoop --source "/mnt/network/hadoop"

# With verbose output
python index_codebase.py --parser hadoop --source "/data/hadoop" --verbose
```

---

## ☁️ Azure Virtual Desktop (AVD)

### AVD-Specific Considerations

1. **Persistent Storage**

   Use D: drive for persistence (C: may be reset):
   ```cmd
   D:\CodebaseIntelligence
   ```

2. **Network Access**

   Ensure network access for initial `pip install`:
   - Check firewall rules
   - Configure proxy if needed:
     ```cmd
     set HTTP_PROXY=http://proxy.company.com:8080
     set HTTPS_PROXY=http://proxy.company.com:8080
     ```

3. **Resources**

   Recommended VM size: D4s_v3 or higher (4 vCPU, 16 GB RAM)

### AVD Installation

Same as Windows installation, just use D: drive:

```cmd
cd D:\CodebaseIntelligence
setup_windows.bat
```

### AVD Performance Tips

1. **Copy data locally** before indexing (if from network drive)
2. **Use SSD storage** for better performance
3. **Close other applications** during indexing
4. **Monitor RAM usage** for large repositories

---

## ✅ Verification

### Run Verification Tool

```bash
python verify_parser_version.py
```

### What It Checks

- ✓ Python version compatible
- ✓ Parser files present
- ✓ Unique ID generation code
- ✓ Path handling fixes
- ✓ Coordinator support
- ✓ Component types
- ✓ Document ID structure

### Manual Verification

```bash
# Check parsers exist
ls parsers/hadoop/parser.py
ls parsers/abinitio/parser.py
ls parsers/databricks/parser.py

# Test import
python -c "from parsers.hadoop import HadoopParser; print('OK')"
```

---

## 🎯 First Index

### Activate Environment

**Windows:**
```cmd
call venv\Scripts\activate.bat
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### Index Different Systems

**Hadoop/Oozie:**
```bash
python index_codebase.py --parser hadoop --source "/path/to/hadoop"
```

Features:
- Finds all `workflow.xml` and `*_workflow.xml` files
- Finds all `coordinator.xml` and `*_coordinator.xml` files
- Resolves sub-workflows recursively
- Extracts variables and properties
- Captures schedules from coordinators

**Ab Initio:**
```bash
python index_codebase.py --parser abinitio --source "/path/to/abinitio"
```

Features:
- Parses `.mp` files
- Clean parameter extraction (FAWN approach)
- GraphFlow data lineage
- Excel export with 4 sheets

**Databricks:**
```bash
python index_codebase.py --parser databricks --source "/path/to/databricks"
```

Features:
- Parses `.py`, `.sql`, `.scala`, `.ipynb` files
- Extracts SQL queries
- Identifies table references

### Monitor Progress

```bash
# Check logs
tail -f outputs/indexing.log

# Check database size
du -sh chroma_db/  # Mac/Linux
dir chroma_db      # Windows
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. "python is not recognized" (Windows)

**Problem**: Python not in PATH

**Solution**:
```cmd
REM Find Python
where python

REM If not found, reinstall Python with "Add to PATH" checked
REM Or add manually:
set PATH=%PATH%;C:\Python39
```

#### 2. "No module named 'parsers'"

**Problem**: Not running from correct directory or parsers not installed

**Solution**:
```bash
# Ensure you're in CodebaseIntelligence directory
cd /path/to/CodebaseIntelligence

# Verify parsers exist
ls parsers/

# Reinstall if needed
pip install -e .
```

#### 3. Duplicate ID Errors

**Error**: `Expected IDs to be unique, found X duplicated IDs`

**Solution**:
```bash
# Verify you have the latest parser code
python verify_parser_version.py

# Should see all ✓ checkmarks
# If not, you need updated parser files
```

#### 4. Slow Performance on Network Drive

**Problem**: Indexing very slow when source is on network drive

**Solution**:
```bash
# Windows: Copy to local drive first
robocopy \\server\share\Hadoop C:\Temp\Hadoop /E

# Mac/Linux: Use rsync
rsync -av /network/hadoop /local/hadoop

# Then index from local copy
python index_codebase.py --source "C:\Temp\Hadoop"
```

#### 5. Permission Denied

**Problem**: Cannot write to directory

**Windows Solution**:
```cmd
REM Run Command Prompt as Administrator
REM Or grant permissions:
icacls C:\Projects\CodebaseIntelligence /grant Users:F /T
```

**Mac/Linux Solution**:
```bash
# Fix permissions
chmod -R 755 /path/to/CodebaseIntelligence

# Or use sudo (not recommended)
sudo python index_codebase.py ...
```

#### 6. Virtual Environment Activation Fails (Windows PowerShell)

**Problem**: Execution policy restriction

**Solution**:
```powershell
# In PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
.\venv\Scripts\Activate.ps1
```

#### 7. Long Path Errors (Windows)

**Error**: `FileNotFoundError: [WinError 206] The filename or extension is too long`

**Solution**: Enable long paths in Windows:
```
1. Run regedit as Administrator
2. Navigate to: HKLM\SYSTEM\CurrentControlSet\Control\FileSystem
3. Set LongPathsEnabled = 1
4. Reboot
```

#### 8. Firewall/Proxy Issues

**Problem**: Cannot download packages

**Solution**:
```bash
# Windows
set HTTP_PROXY=http://proxy:8080
set HTTPS_PROXY=http://proxy:8080

# Mac/Linux
export HTTP_PROXY=http://proxy:8080
export HTTPS_PROXY=http://proxy:8080

# Then retry installation
pip install -r requirements-minimal.txt
```

---

## 📊 Performance Expectations

### Indexing Time

| Repo Size | Files | Components | Time (AVD D4s_v3) | Memory |
|-----------|-------|------------|-------------------|--------|
| Small | 50 | 200 | ~45 seconds | 2 GB |
| Medium | 200 | 1,000 | ~3 minutes | 4 GB |
| Large | 500 | 2,500 | ~8 minutes | 6 GB |
| Very Large | 1,000 | 5,000 | ~18 minutes | 10 GB |

### Disk Space Usage

- **CodebaseIntelligence**: ~50 MB
- **venv (installed)**: ~450 MB
- **Per indexed repo**: 100-500 MB (depends on size)
- **Total recommended**: 2-5 GB

---

## 🔄 Updates & Maintenance

### Updating Code

If you receive updated parser files:

1. **Backup current version** (optional but recommended):
   ```bash
   # Windows
   xcopy CodebaseIntelligence CodebaseIntelligence_backup /E /I

   # Mac/Linux
   cp -r CodebaseIntelligence CodebaseIntelligence_backup
   ```

2. **Copy new files** over existing

3. **Verify update**:
   ```bash
   python verify_parser_version.py
   ```

### Rebuilding Index

If you need to rebuild the index from scratch:

```bash
# Delete existing database
rm -rf chroma_db/  # Mac/Linux
rmdir /s chroma_db  # Windows

# Reindex
python index_codebase.py --parser hadoop --source "/path/to/hadoop"
```

---

## 🎓 Best Practices

### 1. Virtual Environment

**Always activate before use**:
```bash
# Windows
call venv\Scripts\activate.bat

# Mac/Linux
source venv/bin/activate
```

### 2. Path Handling

**Use quotes for paths with spaces**:
```bash
python index_codebase.py --source "C:\Program Files\Hadoop"
```

### 3. Network Drives

**Copy large repos locally first** for better performance

### 4. Antivirus

**Add exclusions** for:
- `venv/` directory
- Python executable
- `chroma_db/` directory

### 5. Resources

**Close other applications** during indexing of large repositories

---

## 📞 Getting Help

### Before Asking for Help

1. Run verification:
   ```bash
   python verify_parser_version.py
   ```

2. Check logs:
   ```bash
   ls outputs/
   cat outputs/indexing.log
   ```

3. Verify environment:
   ```bash
   python --version
   pip list
   ```

### What to Include in Bug Reports

- Python version (`python --version`)
- Operating system and version
- Output of `verify_parser_version.py`
- Error messages (full traceback)
- Log files from `outputs/` directory

---

## ✨ Quick Reference

### Common Commands

```bash
# Setup
setup_windows.bat                    # Windows
python3 -m venv venv && ...          # Mac/Linux

# Activate
call venv\Scripts\activate.bat       # Windows
source venv/bin/activate              # Mac/Linux

# Verify
python verify_parser_version.py

# Index
python index_codebase.py --parser hadoop --source "/path"

# Query
python chatbot_cli.py

# Check status
python -c "import sys; print(sys.executable)"  # Which Python?
pip list                                       # What's installed?
```

---

**Ready to deploy? Copy `CodebaseIntelligence` folder and run the setup script for your platform!**
