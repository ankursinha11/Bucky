#!/usr/bin/env python3
"""
Main Runner Script
Runs complete analysis: parsing, STTM generation, gap analysis, and report generation
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from parsers.abinitio import AbInitioParser
from parsers.hadoop import HadoopParser
from parsers.databricks import DatabricksParser
from core.sttm_generator import STTMGenerator
from core.gap_analyzer import GapAnalyzer
from core.matchers import ProcessMatcher
from utils import ExcelExporter


def setup_logging(log_level="INFO"):
    """Setup logging"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )
    logger.add(
        "outputs/logs/app.log",
        rotation="100 MB",
        level=log_level,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Codebase Intelligence - Gap Analysis & STTM Generator"
    )

    # Input paths
    parser.add_argument(
        "--abinitio-path",
        type=str,
        help="Path to Ab Initio files directory",
    )
    parser.add_argument(
        "--hadoop-path",
        type=str,
        help="Path to Hadoop repository",
    )
    parser.add_argument(
        "--databricks-path",
        type=str,
        help="Path to Databricks notebooks and ADF pipelines",
    )

    # Analysis options
    parser.add_argument(
        "--mode",
        type=str,
        choices=["parse", "sttm", "gap", "full"],
        default="full",
        help="Analysis mode: parse only, STTM only, gap only, or full analysis",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/reports",
        help="Output directory for reports",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    logger.info("=" * 80)
    logger.info("Codebase Intelligence Platform - Starting Analysis")
    logger.info("=" * 80)

    # Storage
    abinitio_processes = []
    abinitio_components = []
    hadoop_processes = []
    hadoop_components = []
    databricks_processes = []
    databricks_components = []

    # Step 1: Parse Ab Initio
    if args.abinitio_path:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Parsing Ab Initio Files")
        logger.info("=" * 80)

        abi_parser = AbInitioParser()
        abi_result = abi_parser.parse_directory(args.abinitio_path)

        abinitio_processes = abi_result["processes"]
        abinitio_components = abi_result["components"]

        logger.info(f"✓ Parsed {len(abinitio_processes)} processes")
        logger.info(f"✓ Extracted {len(abinitio_components)} components")

        # Export to Excel (FAWN-like format)
        abi_excel_path = Path(args.output_dir) / "AbInitio_Parsed_Output.xlsx"
        abi_parser.export_to_excel(str(abi_excel_path))
        logger.info(f"✓ Exported to: {abi_excel_path}")

    # Step 2: Parse Hadoop
    if args.hadoop_path:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Parsing Hadoop Repository")
        logger.info("=" * 80)

        hadoop_parser = HadoopParser()
        hadoop_result = hadoop_parser.parse_directory(args.hadoop_path)

        hadoop_processes = hadoop_result["processes"]
        hadoop_components = hadoop_result["components"]

        logger.info(f"✓ Parsed {len(hadoop_processes)} processes")
        logger.info(f"✓ Extracted {len(hadoop_components)} components")

    # Step 3: Parse Databricks
    if args.databricks_path:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Parsing Databricks Repository")
        logger.info("=" * 80)

        databricks_parser = DatabricksParser()
        databricks_result = databricks_parser.parse_directory(args.databricks_path)

        databricks_processes = databricks_result["processes"]
        databricks_components = databricks_result["components"]

        logger.info(f"✓ Parsed {len(databricks_processes)} processes")
        logger.info(f"✓ Extracted {len(databricks_components)} components")

    if args.mode == "parse":
        logger.info("\n✓ Parsing completed. Exiting.")
        return

    # Step 4: Generate STTM
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Generating Source-to-Target Mappings (STTM)")
    logger.info("=" * 80)

    sttm_generator = STTMGenerator()
    exporter = ExcelExporter(args.output_dir)

    # Generate STTM for each Ab Initio process
    if abinitio_processes:
        for process in abinitio_processes[:3]:  # Limit to first 3 for demo
            sttm_report = sttm_generator.generate_from_process(process, abinitio_components)
            logger.info(f"✓ Generated STTM for {process.name}: {sttm_report.total_mappings} mappings")

            # Export
            filename = f"STTM_{process.name}.xlsx"
            exporter.export_sttm_report(sttm_report, filename)

    # Generate STTM for each Hadoop process
    if hadoop_processes:
        for process in hadoop_processes[:3]:  # Limit to first 3 for demo
            sttm_report = sttm_generator.generate_from_process(process, hadoop_components)
            logger.info(f"✓ Generated STTM for {process.name}: {sttm_report.total_mappings} mappings")

            # Export
            filename = f"STTM_{process.name}.xlsx"
            exporter.export_sttm_report(sttm_report, filename)

    # Generate STTM for each Databricks process
    if databricks_processes:
        for process in databricks_processes[:3]:  # Limit to first 3 for demo
            sttm_report = sttm_generator.generate_from_process(process, databricks_components)
            logger.info(f"✓ Generated STTM for {process.name}: {sttm_report.total_mappings} mappings")

            # Export
            filename = f"STTM_{process.name}.xlsx"
            exporter.export_sttm_report(sttm_report, filename)

    if args.mode == "sttm":
        logger.info("\n✓ STTM generation completed. Exiting.")
        return

    # Step 5: Process Matching & Gap Analysis
    # Combine target systems (Hadoop and Databricks)
    target_processes = hadoop_processes + databricks_processes
    target_components = hadoop_components + databricks_components

    if abinitio_processes and target_processes:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Process Matching")
        logger.info("=" * 80)

        matcher = ProcessMatcher()
        process_mappings = matcher.match_processes(
            abinitio_processes,
            target_processes,
            abinitio_components,
            target_components,
        )

        logger.info(f"✓ Matched {len(process_mappings)} process pairs")

        # Convert mappings format
        mappings_dict = {src_id: tgt_id for src_id, (tgt_id, score) in process_mappings.items()}

        # Gap Analysis
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Gap Analysis")
        logger.info("=" * 80)

        gap_analyzer = GapAnalyzer()
        gaps = gap_analyzer.analyze(
            abinitio_processes,
            target_processes,
            abinitio_components,
            target_components,
            mappings_dict,
        )

        logger.info(f"✓ Identified {len(gaps)} gaps")

        # Print summary
        summary = gap_analyzer.get_summary()
        logger.info("\nGap Summary:")
        logger.info(f"  Total Gaps: {summary['total_gaps']}")
        logger.info(f"  By Severity: {summary['by_severity']}")
        logger.info(f"  By Type: {summary['by_type']}")

        # Export gap analysis
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: Exporting Reports")
        logger.info("=" * 80)

        gap_report_path = exporter.export_gap_analysis(gaps)
        logger.info(f"✓ Gap Analysis Report: {gap_report_path}")

        # Generate cross-system STTM
        if mappings_dict:
            cross_sttm = sttm_generator.generate_cross_system(
                abinitio_processes,
                target_processes,
                abinitio_components + target_components,
                mappings_dict,
            )
            cross_sttm_path = exporter.export_sttm_report(cross_sttm, "STTM_Cross_System.xlsx")
            logger.info(f"✓ Cross-System STTM: {cross_sttm_path}")

            # Combined report
            combined_path = exporter.export_combined_report(cross_sttm, gaps)
            logger.info(f"✓ Combined Report: {combined_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Analysis Complete!")
    logger.info(f"✓ Reports saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
