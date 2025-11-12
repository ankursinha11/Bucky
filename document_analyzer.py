"""
Document Analysis System
Analyzes uploaded documents (Excel, PDF, images, text files)
Understands comparison sheets, flow diagrams, STTM sheets, and custom formats
Uses AI to understand document structure and content
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False


class DocumentAnalyzer:
    """
    Analyze uploaded documents and extract structured information

    Supports:
    - Excel files (.xlsx, .xls) - Comparison sheets, STTM sheets, mapping tables
    - JSON files (.json) - Structured data
    - Text files (.txt, .md) - Unstructured text
    - CSV files (.csv) - Tabular data
    - PDF files (.pdf) - Reports and documentation (future)
    - Images (.png, .jpg) - Flow diagrams, screenshots (future with AI)
    """

    def __init__(self, ai_client=None):
        """Initialize document analyzer"""
        self.ai_client = ai_client
        self.analysis_cache: Dict[str, Dict] = {}

    def analyze_document(self, file_path: str, document_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze any uploaded document

        Args:
            file_path: Path to uploaded document
            document_type: Optional hint about document type
                         ('comparison_sheet', 'sttm', 'flow_diagram', 'mapping_table', etc.)

        Returns:
            Dict with analysis results
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }

        logger.info(f"Analyzing document: {file_path.name}")

        # Detect file type
        suffix = file_path.suffix.lower()

        try:
            if suffix in ['.xlsx', '.xls']:
                return self._analyze_excel(file_path, document_type)
            elif suffix == '.json':
                return self._analyze_json(file_path, document_type)
            elif suffix in ['.csv']:
                return self._analyze_csv(file_path, document_type)
            elif suffix in ['.txt', '.md']:
                return self._analyze_text(file_path, document_type)
            elif suffix in ['.pdf']:
                return self._analyze_pdf(file_path, document_type)
            elif suffix in ['.png', '.jpg', '.jpeg']:
                return self._analyze_image(file_path, document_type)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {suffix}"
                }

        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _analyze_excel(self, file_path: Path, document_type: Optional[str]) -> Dict[str, Any]:
        """Analyze Excel file"""
        if not EXCEL_AVAILABLE:
            return {
                "success": False,
                "error": "Excel support not available (install openpyxl)"
            }

        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheets = {}

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheets[sheet_name] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "preview": df.head(5).to_dict('records')
                }

            # Detect document structure using AI if available
            structure_analysis = self._detect_excel_structure(sheets, document_type)

            return {
                "success": True,
                "file_type": "excel",
                "file_name": file_path.name,
                "sheets": sheets,
                "structure": structure_analysis,
                "summary": self._generate_excel_summary(sheets, structure_analysis)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to analyze Excel: {e}"
            }

    def _detect_excel_structure(self, sheets: Dict, document_type: Optional[str]) -> Dict[str, Any]:
        """Detect Excel document structure"""
        # Heuristic-based detection

        structure = {
            "detected_type": document_type or "unknown",
            "is_comparison_sheet": False,
            "is_mapping_table": False,
            "is_sttm_sheet": False,
            "confidence": "low"
        }

        # Check for comparison sheet patterns
        for sheet_name, sheet_data in sheets.items():
            columns = [str(col).lower() for col in sheet_data["column_names"]]

            # Comparison sheet indicators
            if any(keyword in sheet_name.lower() for keyword in ['comparison', 'compare', 'vs']):
                structure["is_comparison_sheet"] = True
                structure["detected_type"] = "comparison_sheet"
                structure["confidence"] = "high"

            if any(keyword in ' '.join(columns) for keyword in ['hadoop', 'databricks', 'abinitio', 'source', 'target']):
                structure["is_comparison_sheet"] = True
                if structure["detected_type"] == "unknown":
                    structure["detected_type"] = "comparison_sheet"

            # Mapping table indicators
            if any(keyword in sheet_name.lower() for keyword in ['mapping', 'map', 'correlation']):
                structure["is_mapping_table"] = True
                if structure["detected_type"] == "unknown":
                    structure["detected_type"] = "mapping_table"
                    structure["confidence"] = "medium"

            # STTM sheet indicators
            if any(keyword in sheet_name.lower() for keyword in ['sttm', 'source to target', 's2t']):
                structure["is_sttm_sheet"] = True
                structure["detected_type"] = "sttm_sheet"
                structure["confidence"] = "high"

        return structure

    def _generate_excel_summary(self, sheets: Dict, structure: Dict) -> str:
        """Generate human-readable summary of Excel file"""
        summary_parts = []

        summary_parts.append(f"Excel file with {len(sheets)} sheet(s)")

        if structure["detected_type"] != "unknown":
            summary_parts.append(f"Detected as: {structure['detected_type']}")

        total_rows = sum(sheet["rows"] for sheet in sheets.values())
        summary_parts.append(f"Total rows: {total_rows}")

        # List sheets
        summary_parts.append("\nSheets:")
        for sheet_name, sheet_data in sheets.items():
            summary_parts.append(f"  - {sheet_name}: {sheet_data['rows']} rows, {sheet_data['columns']} columns")

        return "\n".join(summary_parts)

    def _analyze_json(self, file_path: Path, document_type: Optional[str]) -> Dict[str, Any]:
        """Analyze JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Detect structure
            structure = self._detect_json_structure(data)

            return {
                "success": True,
                "file_type": "json",
                "file_name": file_path.name,
                "structure": structure,
                "data_preview": self._preview_json(data),
                "summary": self._generate_json_summary(data, structure)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to analyze JSON: {e}"
            }

    def _detect_json_structure(self, data: Any) -> Dict[str, Any]:
        """Detect JSON structure"""
        structure = {
            "type": type(data).__name__,
            "is_workflow_mapping": False,
            "is_pipeline_definition": False,
            "is_validation_report": False
        }

        if isinstance(data, dict):
            keys = set(data.keys())

            # Check for workflow mapping
            if "mappings" in keys or "workflow_mappings" in keys:
                structure["is_workflow_mapping"] = True

            # Check for pipeline definition
            if "pipelines" in keys or "notebooks" in keys:
                structure["is_pipeline_definition"] = True

            # Check for validation report
            if "gaps" in keys or "validation" in keys or "report" in keys:
                structure["is_validation_report"] = True

        return structure

    def _preview_json(self, data: Any, max_depth: int = 3) -> Any:
        """Create preview of JSON data"""
        if isinstance(data, dict):
            if max_depth == 0:
                return f"<dict with {len(data)} keys>"
            return {k: self._preview_json(v, max_depth - 1) for k, v in list(data.items())[:10]}
        elif isinstance(data, list):
            if max_depth == 0:
                return f"<list with {len(data)} items>"
            return [self._preview_json(item, max_depth - 1) for item in data[:5]]
        else:
            return data

    def _generate_json_summary(self, data: Any, structure: Dict) -> str:
        """Generate summary of JSON file"""
        summary_parts = [f"JSON file ({structure['type']})"]

        if isinstance(data, dict):
            summary_parts.append(f"{len(data)} top-level keys")

            if structure["is_workflow_mapping"]:
                summary_parts.append("Contains workflow mappings")
            if structure["is_pipeline_definition"]:
                summary_parts.append("Contains pipeline definitions")
            if structure["is_validation_report"]:
                summary_parts.append("Contains validation report")

        elif isinstance(data, list):
            summary_parts.append(f"{len(data)} items")

        return "\n".join(summary_parts)

    def _analyze_csv(self, file_path: Path, document_type: Optional[str]) -> Dict[str, Any]:
        """Analyze CSV file"""
        try:
            df = pd.read_csv(file_path)

            return {
                "success": True,
                "file_type": "csv",
                "file_name": file_path.name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "preview": df.head(10).to_dict('records'),
                "summary": f"CSV file with {len(df)} rows and {len(df.columns)} columns"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to analyze CSV: {e}"
            }

    def _analyze_text(self, file_path: Path, document_type: Optional[str]) -> Dict[str, Any]:
        """Analyze text file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            lines = content.split('\n')

            return {
                "success": True,
                "file_type": "text",
                "file_name": file_path.name,
                "lines": len(lines),
                "characters": len(content),
                "preview": '\n'.join(lines[:20]),
                "summary": f"Text file with {len(lines)} lines and {len(content)} characters"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to analyze text file: {e}"
            }

    def _analyze_pdf(self, file_path: Path, document_type: Optional[str]) -> Dict[str, Any]:
        """Analyze PDF file"""
        # TODO: Implement PDF analysis (requires PyPDF2 or similar)
        return {
            "success": False,
            "error": "PDF analysis not yet implemented"
        }

    def _analyze_image(self, file_path: Path, document_type: Optional[str]) -> Dict[str, Any]:
        """Analyze image file (flow diagrams, screenshots)"""
        # TODO: Implement image analysis with AI (requires vision model)
        return {
            "success": False,
            "error": "Image analysis not yet implemented (requires AI vision model)"
        }

    def compare_with_indexed_data(
        self,
        document_analysis: Dict,
        workflow_intelligence
    ) -> Dict[str, Any]:
        """
        Compare uploaded document with indexed workflow data

        This allows answering questions like:
        - "How does this comparison sheet match our actual mappings?"
        - "What's in this STTM that's not in our system?"
        - "Validate this flow diagram against indexed workflows"
        """
        if not document_analysis.get("success"):
            return {
                "success": False,
                "error": "Document analysis failed"
            }

        # Extract workflows/mappings from document
        document_workflows = self._extract_workflows_from_document(document_analysis)

        if not document_workflows:
            return {
                "success": True,
                "message": "No workflows found in document to compare"
            }

        # Compare with indexed data
        comparison_results = {
            "document_workflows": len(document_workflows),
            "matches": [],
            "missing_in_system": [],
            "missing_in_document": []
        }

        # This would use workflow_intelligence to find matches
        # TODO: Implement actual comparison logic

        return {
            "success": True,
            "comparison": comparison_results
        }

    def _extract_workflows_from_document(self, document_analysis: Dict) -> List[Dict]:
        """Extract workflow information from analyzed document"""
        workflows = []

        # Extract from Excel comparison sheet
        if document_analysis.get("file_type") == "excel":
            if document_analysis.get("structure", {}).get("is_comparison_sheet"):
                # Extract workflows from comparison sheet
                for sheet_name, sheet_data in document_analysis.get("sheets", {}).items():
                    # Look for source/target columns
                    # TODO: Implement extraction logic
                    pass

        # Extract from JSON
        elif document_analysis.get("file_type") == "json":
            # Look for workflow/mapping structures
            # TODO: Implement extraction logic
            pass

        return workflows


# Testing function
def test_document_analyzer():
    """Test document analyzer"""
    print("=" * 60)
    print("DOCUMENT ANALYZER TEST")
    print("=" * 60)

    analyzer = DocumentAnalyzer()

    # Test with actual files if they exist
    test_files = [
        "databricks_pipeline_analysis.json",
        "abinitio_graph_mappings.json",
        "outputs/hadoop_to_databricks_validation.json"
    ]

    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\n📄 Analyzing: {test_file}")
            result = analyzer.analyze_document(test_file)

            if result.get("success"):
                print(f"✅ Success!")
                print(f"   Type: {result.get('file_type')}")
                print(f"   Summary: {result.get('summary', 'N/A')}")
            else:
                print(f"❌ Failed: {result.get('error')}")

    print("\n✅ Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_document_analyzer()
