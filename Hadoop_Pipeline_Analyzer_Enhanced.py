#!/usr/bin/env python3
"""
Enhanced Hadoop Pipeline Analyzer - Databricks-style Output
- Analyzes Hadoop repositories and generates Excel similar to Databricks format
- Handles missing scripts (referenced but not in repo)
- Extracts script references from workflow XML
- Provides comprehensive summary statistics
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
from collections import defaultdict


class HadoopPipelineAnalyzer:
    """Analyzes Hadoop pipelines similar to Databricks format"""

    def __init__(self):
        self.results = []
        self.missing_scripts = []  # Track referenced but missing scripts

    def analyze_folder(self, folder_path: str):
        """Analyze all Hadoop repos in a folder"""
        folder = Path(folder_path)
        
        print(f"📂 Scanning: {folder_path}\n")
        
        # Find all repos
        repos = []
        for item in folder.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it has workflows
                if (item / "workflows").exists() or list(item.rglob("*.xml")):
                    repos.append(item)

        print(f"✅ Found {len(repos)} repositories\n")

        for repo in repos:
            self.analyze_repo(str(repo))

    def analyze_repo(self, repo_path: str):
        """Analyze a single Hadoop repository"""
        repo = Path(repo_path)
        repo_name = repo.name

        print(f"\n{'='*80}")
        print(f"🔍 Repository: {repo_name}")
        print(f"{'='*80}")

        # Find all workflow.xml files (excluding coordinators)
        workflow_files = self._find_workflow_files(repo)
        
        print(f"📋 Found {len(workflow_files)} workflow files\n")

        for workflow_file in workflow_files:
            self._analyze_workflow(repo, repo_name, workflow_file)

    def _find_workflow_files(self, repo: Path) -> List[Path]:
        """Find all workflow.xml files (not coordinators)"""
        workflow_files = []
        
        # Search for all XML files
        for xml_file in repo.rglob("*.xml"):
            # Skip if coordinator
            if "coordinator" in xml_file.name.lower():
                continue
            
            # Check if it's a workflow file
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                # Check if it has workflow-app root element
                if 'workflow-app' in root.tag:
                    workflow_files.append(xml_file)
            except:
                continue
        
        return workflow_files

    def _analyze_workflow(self, repo: Path, repo_name: str, workflow_file: Path):
        """Analyze a single workflow file"""
        
        # Extract module/pipeline info
        module = self._extract_module_name(workflow_file)
        pipeline_name = self._extract_pipeline_name(workflow_file)
        
        print(f"  📄 {pipeline_name}")
        print(f"     Module: {module}")
        
        # Parse workflow XML
        workflow_data = self._parse_workflow_xml(workflow_file)
        
        # Find actual scripts in repo
        actual_scripts = self._find_actual_scripts(workflow_file)
        
        # Extract referenced scripts from XML
        referenced_scripts = workflow_data['referenced_scripts']
        
        # Identify missing scripts
        missing = self._identify_missing_scripts(referenced_scripts, actual_scripts)
        if missing:
            print(f"     ⚠️  {len(missing)} script(s) referenced but NOT in repo")
            for script in missing:
                self.missing_scripts.append({
                    'repo': repo_name,
                    'pipeline': pipeline_name,
                    'script': script
                })
        
        # Count total scripts (actual + referenced)
        all_scripts = list(actual_scripts.values())
        total_script_count = len(referenced_scripts)  # Use referenced count (what workflow expects)
        
        # Calculate metrics
        total_loc = sum(s.get('lines', 0) for s in all_scripts)
        actions = workflow_data['actions']
        tables = workflow_data['table_count']
        
        # Classify size
        pipeline_length = self._classify_pipeline_length(total_script_count, total_loc, actions, tables)
        
        print(f"     📊 Actions: {actions}, Tables: {tables}")
        print(f"     📝 Scripts: {total_script_count} ({len(actual_scripts)} found, {len(missing)} missing)")
        print(f"     📏 LOC: {total_loc:,}")
        print(f"     {pipeline_length}\n")
        
        # Store result
        self.results.append({
            'repo': repo_name,
            'pipeline_name': pipeline_name,
            'module': module,
            'no_of_scripts': total_script_count,
            'scripts_found': len(actual_scripts),
            'scripts_missing': len(missing),
            'pipeline_length': pipeline_length,
            'actions': actions,
            'tables': tables,
            'total_loc': total_loc,
            'actual_scripts': actual_scripts,
            'referenced_scripts': list(referenced_scripts),
            'missing_scripts': missing,
            'workflow_path': str(workflow_file)
        })

    def _extract_module_name(self, workflow_file: Path) -> str:
        """Extract module name from workflow path"""
        parts = workflow_file.parts
        
        # Try to find module from path structure
        if "workflows" in parts:
            idx = parts.index("workflows")
            if idx + 1 < len(parts):
                return parts[idx + 1]  # e.g., ingest_chc, ingest_table
        
        # Fallback: use parent directory name
        return workflow_file.parent.name

    def _extract_pipeline_name(self, workflow_file: Path) -> str:
        """Extract pipeline name from workflow XML name attribute"""
        try:
            # First, try to get name from XML
            tree = ET.parse(workflow_file)
            root = tree.getroot()
            
            # Check if workflow-app has a name attribute
            if 'name' in root.attrib:
                xml_name = root.attrib['name']
                # Clean up the name
                if xml_name and len(xml_name) > 0:
                    # Remove common prefixes/suffixes
                    cleaned = xml_name.replace('escan_data_ingestion', '').replace(':', '').strip()
                    if cleaned:
                        return cleaned
            
            # Fallback to filename-based name
            name = workflow_file.stem
            
            # Remove common suffixes
            name = name.replace('_workflow', '').replace('workflow_', '')
            
            # If it's just "workflow", try using parent directory name
            if name == 'workflow':
                parent_name = workflow_file.parent.name
                if parent_name != 'oozie':
                    return parent_name
            
            # If still empty or generic, use file stem
            if not name or name == '_':
                name = workflow_file.stem
            
            return name
            
        except Exception as e:
            # If XML parsing fails, use filename
            name = workflow_file.stem
            if name == 'workflow':
                return workflow_file.parent.name
            return name

    def _parse_workflow_xml(self, workflow_file: Path) -> Dict:
        """Parse workflow XML to extract actions, tables, and script references"""
        try:
            tree = ET.parse(workflow_file)
            root = tree.getroot()
            
            # Get all text content for parsing
            all_text = ET.tostring(root, encoding='unicode')
            
            # Count actions - using simple iteration without XPath
            action_count = 0
            for elem in root.iter():
                if elem.tag.endswith('action') or 'action' in elem.tag:
                    action_count += 1
            
            # Extract referenced scripts from actions
            referenced_scripts = set()
            
            # Iterate through all elements to find script references
            for elem in root.iter():
                tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                
                # Look for script-related tags
                if tag_name in ['script', 'file', 'exec', 'jar', 'app-path']:
                    if elem.text:
                        script_path = elem.text.strip()
                        # Skip if it's a variable or path-only
                        if '$' in script_path or not script_path:
                            continue
                        
                        # Extract filename
                        if '/' in script_path:
                            filename = script_path.split('/')[-1]
                        else:
                            filename = script_path
                        
                        # Only include actual script files
                        if any(filename.endswith(ext) for ext in ['.py', '.pig', '.sh', '.sql', '.hql', '.jar']):
                            referenced_scripts.add(filename)
            
            # Extract tables
            tables = self._extract_tables_from_xml(all_text)
            
            return {
                'actions': action_count,
                'table_count': len(tables),
                'tables': tables,
                'referenced_scripts': referenced_scripts
            }
            
        except Exception as e:
            print(f"     ⚠️  XML parsing error: {e}")
            return {
                'actions': 0,
                'table_count': 0,
                'tables': [],
                'referenced_scripts': set()
            }

    def _extract_tables_from_xml(self, xml_text: str) -> List[str]:
        """Extract table names from XML text"""
        tables = set()
        
        # HDFS paths
        hdfs_patterns = re.findall(r'hdfs://[^<>"\s]+', xml_text)
        for path in hdfs_patterns:
            if '/' in path:
                segments = path.split('/')
                # Skip common paths
                skip_words = ['warehouse', 'user', 'tmp', 'stage', 'staging', 'temp', 'hdfs', 'nameservice']
                if not any(word in segments for word in skip_words):
                    # Get last meaningful segment
                    for seg in reversed(segments):
                        if seg and len(seg) > 1 and not seg.startswith('$'):
                            tables.add(seg)
                            break
        
        # Look for table properties
        table_patterns = re.findall(r'<name>table</name>\s*<value>([^<]+)</value>', xml_text)
        tables.update(table_patterns)
        
        # Database.table patterns
        db_table_patterns = re.findall(r'(\w+\.\w+)(?:\s|<|;|\'|")', xml_text)
        for pattern in db_table_patterns:
            if '.' in pattern and pattern.count('.') == 1:
                tables.add(pattern.split('.')[1])
        
        return list(tables)

    def _find_actual_scripts(self, workflow_file: Path) -> Dict:
        """Find actual script files in the repo - RECURSIVELY"""
        scripts = {}
        
        # Get base directory (parent or grandparent of workflow file)
        base_dir = workflow_file.parent
        
        # If workflow is in oozie/ subdirectory, go up one more level
        if base_dir.name == 'oozie':
            base_dir = base_dir.parent
        
        # Search for script directories RECURSIVELY (using rglob instead of glob)
        script_dirs = {
            'spark': ['*.py'],
            'pig': ['*.pig'],
            'shell': ['*.sh'],
            'hive': ['*.sql', '*.hql'],
            'scripts': ['*.py', '*.sh', '*.pig', '*.sql', '*.hql']
        }
        
        for dir_name, patterns in script_dirs.items():
            script_dir = base_dir / dir_name
            if script_dir.exists():
                for pattern in patterns:
                    # Use rglob for recursive search
                    for script_file in script_dir.rglob(pattern):
                        script_type = self._get_script_type(script_file)
                        lines = self._count_lines(script_file)
                        
                        # Store by filename only (not full path) to match references
                        scripts[script_file.name] = {
                            'type': script_type,
                            'path': str(script_file),
                            'lines': lines,
                            'found': True
                        }
        
        return scripts

    def _identify_missing_scripts(self, referenced: Set[str], actual: Dict) -> List[str]:
        """Identify scripts referenced in XML but not found in repo"""
        actual_names = set(actual.keys())
        missing = referenced - actual_names
        return sorted(list(missing))

    def _get_script_type(self, file_path: Path) -> str:
        """Determine script type from extension"""
        ext = file_path.suffix.lower()
        type_map = {
            '.py': 'PySpark',
            '.pig': 'Pig',
            '.sh': 'Shell',
            '.sql': 'Hive',
            '.hql': 'Hive',
            '.jar': 'Java'
        }
        return type_map.get(ext, 'Unknown')

    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return len(f.readlines())
        except:
            return 0

    def _classify_pipeline_length(self, scripts: int, loc: int, actions: int, tables: int) -> str:
        """Classify pipeline length (Small/Medium/Big Pipeline)"""
        # Calculate complexity score
        script_score = min(scripts / 5, 1.0)     # 5+ scripts = max score
        loc_score = min(loc / 1500, 1.0)         # 1500+ LOC = max score
        action_score = min(actions / 8, 1.0)     # 8+ actions = max score
        table_score = min(tables / 5, 1.0)       # 5+ tables = max score
        
        # Weighted average
        complexity = (script_score * 0.3 + loc_score * 0.3 + action_score * 0.2 + table_score * 0.2)
        
        if complexity <= 0.35:
            return "Small"
        elif complexity <= 0.65:
            return "Medium"
        else:
            return "Big Pipeline"

    def export_to_excel(self, output_file: str = None):
        """Export to Excel in Databricks-style format"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"Hadoop_Pipeline_Analysis_{timestamp}.xlsx"

        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils import get_column_letter

            wb = Workbook()

            # ==================== SHEET 1: Pipeline Summary (Databricks-style) ====================
            ws1 = wb.active
            ws1.title = 'Pipeline Summary'

            # Headers matching Databricks format
            headers1 = ['Pipeline Name', 'Module', 'No. of Scripts', 'Pipeline Length', 'Actions', 'Tables', 'LOC', 'Scripts Missing']
            ws1.append(headers1)

            # Style headers
            header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
            header_font = Font(bold=True, color='FFFFFF', size=11)
            thin_border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            
            for cell in ws1[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.border = thin_border

            # Add data
            for r in self.results:
                ws1.append([
                    r['pipeline_name'],
                    r['module'],
                    r['no_of_scripts'],
                    r['pipeline_length'],
                    r['actions'],
                    r['tables'],
                    r['total_loc'],
                    r['scripts_missing']
                ])

            # Apply borders and alignment to data
            for row in ws1.iter_rows(min_row=2, max_row=ws1.max_row, min_col=1, max_col=len(headers1)):
                for cell in row:
                    cell.border = thin_border
                    if cell.column in [3, 5, 6, 7, 8]:  # Numeric columns
                        cell.alignment = Alignment(horizontal='center')

            # Set column widths
            ws1.column_dimensions['A'].width = 35  # Pipeline Name
            ws1.column_dimensions['B'].width = 25  # Module
            ws1.column_dimensions['C'].width = 15  # No. of Scripts
            ws1.column_dimensions['D'].width = 18  # Pipeline Length
            ws1.column_dimensions['E'].width = 10  # Actions
            ws1.column_dimensions['F'].width = 10  # Tables
            ws1.column_dimensions['G'].width = 12  # LOC
            ws1.column_dimensions['H'].width = 15  # Scripts Missing

            # Add summary statistics (similar to Databricks sidebar)
            summary_start_row = ws1.max_row + 3
            
            # Calculate statistics
            total_pipelines = len(self.results)
            big_count = sum(1 for r in self.results if r['pipeline_length'] == 'Big Pipeline')
            medium_count = sum(1 for r in self.results if r['pipeline_length'] == 'Medium')
            small_count = sum(1 for r in self.results if r['pipeline_length'] == 'Small')
            total_scripts = sum(r['no_of_scripts'] for r in self.results)
            total_missing = sum(r['scripts_missing'] for r in self.results)
            master_pipelines = sum(1 for r in self.results if 'master' in r['pipeline_name'].lower())
            
            # Summary section
            summary_fill = PatternFill(start_color='E7E6E6', end_color='E7E6E6', fill_type='solid')
            summary_font = Font(bold=True, size=10)
            
            ws1[f'G{summary_start_row}'] = 'Pipeline Summary/Details'
            ws1[f'G{summary_start_row}'].font = Font(bold=True, size=12)
            ws1.merge_cells(f'G{summary_start_row}:H{summary_start_row}')
            
            summary_data = [
                ['Big Pipelines', big_count],
                ['Medium', medium_count],
                ['Small', small_count],
                ['Total Pipelines', total_pipelines],
                ['Master Pipeline', master_pipelines],
                ['Total Scripts', total_scripts],
                ['Scripts Missing', total_missing]
            ]
            
            for i, (label, value) in enumerate(summary_data, start=1):
                row_num = summary_start_row + i
                ws1[f'G{row_num}'] = label
                ws1[f'H{row_num}'] = value
                ws1[f'G{row_num}'].fill = summary_fill
                ws1[f'H{row_num}'].fill = summary_fill
                ws1[f'G{row_num}'].font = summary_font
                ws1[f'H{row_num}'].font = Font(bold=True, size=10, color='0000FF')
                ws1[f'G{row_num}'].border = thin_border
                ws1[f'H{row_num}'].border = thin_border

            # ==================== SHEET 2: Script Details ====================
            ws2 = wb.create_sheet('Script Details')
            
            headers2 = ['Pipeline Name', 'Module', 'Script Name', 'Type', 'Lines', 'Status']
            ws2.append(headers2)
            
            for cell in ws2[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.border = thin_border
            
            # Add script details
            for r in self.results:
                # Add found scripts
                for script_name, script_data in r['actual_scripts'].items():
                    ws2.append([
                        r['pipeline_name'],
                        r['module'],
                        script_name,
                        script_data['type'],
                        script_data['lines'],
                        '✓ Found'
                    ])
                
                # Add missing scripts
                for script_name in r['missing_scripts']:
                    ws2.append([
                        r['pipeline_name'],
                        r['module'],
                        script_name,
                        'Unknown',
                        0,
                        '✗ Missing'
                    ])
            
            # Apply formatting
            for row in ws2.iter_rows(min_row=2, max_row=ws2.max_row, min_col=1, max_col=len(headers2)):
                for cell in row:
                    cell.border = thin_border
                    if cell.column == 5:  # Lines column
                        cell.alignment = Alignment(horizontal='center')
                    if cell.column == 6:  # Status column
                        if '✗' in str(cell.value):
                            cell.font = Font(color='FF0000', bold=True)
                        else:
                            cell.font = Font(color='008000')
            
            # Set column widths
            ws2.column_dimensions['A'].width = 35
            ws2.column_dimensions['B'].width = 25
            ws2.column_dimensions['C'].width = 40
            ws2.column_dimensions['D'].width = 15
            ws2.column_dimensions['E'].width = 10
            ws2.column_dimensions['F'].width = 15

            # ==================== SHEET 3: Missing Scripts Report ====================
            ws3 = wb.create_sheet('Missing Scripts')
            
            headers3 = ['Pipeline Name', 'Module', 'Missing Script', 'Note']
            ws3.append(headers3)
            
            for cell in ws3[1]:
                cell.fill = PatternFill(start_color='FF0000', end_color='FF0000', fill_type='solid')
                cell.font = Font(bold=True, color='FFFFFF', size=11)
                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.border = thin_border
            
            for r in self.results:
                if r['missing_scripts']:
                    for script in r['missing_scripts']:
                        ws3.append([
                            r['pipeline_name'],
                            r['module'],
                            script,
                            'Referenced in workflow XML but not found in repo'
                        ])
            
            for row in ws3.iter_rows(min_row=2, max_row=ws3.max_row, min_col=1, max_col=len(headers3)):
                for cell in row:
                    cell.border = thin_border
            
            ws3.column_dimensions['A'].width = 35
            ws3.column_dimensions['B'].width = 25
            ws3.column_dimensions['C'].width = 40
            ws3.column_dimensions['D'].width = 50

            # Save workbook
            wb.save(output_file)
            print(f"\n✅ Excel report generated: {output_file}")
            print(f"   📊 {total_pipelines} pipelines analyzed")
            print(f"   📝 {total_scripts} total scripts ({total_scripts - total_missing} found, {total_missing} missing)")

        except Exception as e:
            print(f"⚠️  Excel generation error: {e}")
            # Fallback to JSON
            import json
            json_file = output_file.replace('.xlsx', '.json') if output_file else 'hadoop_analysis.json'
            with open(json_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"✅ JSON backup saved: {json_file}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Hadoop Pipeline Analyzer - Databricks-style output'
    )

    parser.add_argument(
        'path',
        help='Path to Hadoop repository or folder containing multiple repos'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output Excel file name'
    )

    args = parser.parse_args()

    analyzer = HadoopPipelineAnalyzer()

    # Detect if single repo or folder
    path = Path(args.path)
    if path.is_dir():
        if (path / "workflows").exists() or list(path.rglob("*.xml")):
            # Single repo
            print("📦 Single repository mode\n")
            analyzer.analyze_repo(args.path)
        else:
            # Folder with multiple repos
            print("📦 Multi-repository mode\n")
            analyzer.analyze_folder(args.path)
    else:
        print(f"⚠️  Invalid path: {args.path}")
        return

    # Export results
    analyzer.export_to_excel(args.output)

    print("\n" + "="*80)
    print("✅ Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

