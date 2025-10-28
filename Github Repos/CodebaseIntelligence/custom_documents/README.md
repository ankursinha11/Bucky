# Custom Documents Folder

This folder is where you can add **your own documents** for the chatbot to index and answer questions about.

## Supported File Formats

The chatbot can read and index the following formats:

### 📊 Spreadsheets
- **Excel**: `.xlsx`, `.xls`
- **CSV**: `.csv`

### 📄 Documents
- **Word**: `.docx`
- **PDF**: `.pdf`
- **Text**: `.txt`, `.md`, `.rst`
- **JSON**: `.json`

## How to Use

### 1. Add Your Files

Simply copy your files into this folder (or create subfolders):

```
custom_documents/
├── README.md (this file)
├── my_documentation.xlsx
├── project_specs.docx
├── technical_notes.pdf
├── data_dictionary.csv
└── requirements.txt
```

### 2. Index the Documents

Run the indexer with the `custom` parser:

```bash
python index_codebase.py --parser custom --source ./custom_documents
```

Or on Windows:

```cmd
python index_codebase.py --parser custom --source .\custom_documents
```

### 3. Ask Questions

Start the chatbot and ask questions about your documents:

```bash
python chatbot_cli.py
```

Example questions:
- "What's in the data dictionary?"
- "Summarize the project specifications"
- "What are the requirements from requirements.txt?"
- "Show me the columns in the Excel file"

## Tips

### Excel Files
- Each sheet will be indexed separately
- Column headers are preserved
- Data is converted to searchable text

### Word Documents
- Paragraphs and tables are extracted
- Formatting is preserved where possible

### PDF Files
- Text is extracted from all pages
- Requires PyPDF2: `pip install PyPDF2`

### CSV Files
- Entire CSV is converted to searchable text
- Column headers are included

### Text Files
- `.txt`, `.md`, `.rst` files are indexed directly
- Great for documentation, notes, requirements

### JSON Files
- Entire JSON structure is indexed
- Useful for configuration, API responses

## Best Practices

1. **Organize by topic**: Create subfolders for different topics
   ```
   custom_documents/
   ├── data_dictionaries/
   ├── requirements/
   ├── meeting_notes/
   └── specifications/
   ```

2. **Use descriptive filenames**: `patient_data_dictionary.xlsx` is better than `data.xlsx`

3. **Keep files updated**: Re-run the indexer when you update files

4. **File size**: Large PDFs may take longer to process

## Re-indexing

If you add or update files, re-run the indexer:

```bash
python index_codebase.py --parser custom --source ./custom_documents
```

The indexer will:
- Generate unique IDs based on file paths
- Extract all text content
- Index for fast semantic search
- Make content available to the chatbot

## Installation Requirements

Some file formats require additional libraries:

```bash
# For Excel files
pip install pandas openpyxl

# For Word documents
pip install python-docx

# For PDF files
pip install PyPDF2

# All at once
pip install pandas openpyxl python-docx PyPDF2
```

If a library is missing, the parser will try to read the file as plain text.

## Examples

### Example 1: Data Dictionary

Add `data_dictionary.xlsx` to this folder, then ask:
- "What columns are in the patient table?"
- "What's the data type for claim_id?"

### Example 2: Requirements Document

Add `requirements.docx` to this folder, then ask:
- "What are the functional requirements?"
- "List the security requirements"

### Example 3: Meeting Notes

Add `meeting_notes_2024.txt` to this folder, then ask:
- "What was discussed in the last meeting?"
- "What are the action items?"

## Troubleshooting

**Issue**: File not being indexed
- Check file extension is supported
- Verify file is not corrupted
- Check console output for errors

**Issue**: Content not searchable
- Make sure file contains readable text
- PDFs with images may not work (OCR not supported)
- Protected/encrypted files may fail

**Issue**: Slow indexing
- Large files take longer
- PDFs with many pages are slower
- Consider splitting very large files

## Need Help?

1. Check the console output when indexing
2. Look for error messages
3. Try reading the file as plain text first
4. Check if required libraries are installed

---

**Ready to start?** Just drop your files here and run:
```bash
python index_codebase.py --parser custom --source ./custom_documents
```
