"""
Logic Comparator - AI-Powered Cross-System Logic Comparison
============================================================
Compares transformation logic across different systems using Azure OpenAI

Capabilities:
- Detect equivalent logic patterns
- Identify implementation differences
- Calculate logic similarity scores
- Generate migration recommendations
"""

import os
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LogicComparator:
    """
    Compare logic across systems using AI

    Examples:
    - Ab Initio Transform vs Spark DataFrame operation
    - Hive SQL vs Ab Initio DML
    - Pig Latin vs PySpark
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize logic comparator"""
        self.llm = None
        self.enabled = False

        if OPENAI_AVAILABLE and (api_key or os.getenv("AZURE_OPENAI_API_KEY")):
            try:
                self.llm = AzureOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                )
                self.model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
                self.enabled = True

                logger.info("✓ Logic Comparator initialized with Azure OpenAI")

            except Exception as e:
                logger.warning(f"Could not initialize Logic Comparator: {e}")
                self.enabled = False
        else:
            logger.info("Logic Comparator disabled (no Azure OpenAI)")

        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for deep logic comparison"""
        return """You are an expert ETL architect specializing in DEEP technical comparison of data pipelines across different technologies.

Your expertise spans:
- Ab Initio (Co>Operating System, GDE, transforms, DML)
- Hadoop (Hive, Pig, Spark - SQL, Python, Scala)
- Databricks (notebooks, Delta Lake, SQL, PySpark)
- SQL variants (PostgreSQL, Oracle, Teradata, etc.)

CRITICAL INSTRUCTION: Focus your comparison on PROCESS FLOW, LOGIC, and DATA LEVEL analysis.
Technology differences should ONLY be mentioned at the END and kept minimal.

Users already know these are different technologies - they want to understand:
1. HOW the pipelines work (process flow)
2. WHAT logic is implemented (business rules, transformations)
3. WHAT data is processed (sources, targets, transformations)

NOT just "System 1 uses Ab Initio, System 2 uses Spark" - that's obvious.

Analyze these aspects in DEPTH (in priority order):

1. **Business Logic Equivalence**
   - Are ALL business rules implemented the same way?
   - Any missing transformations or logic?
   - Field-by-field calculation comparison
   - Date/time logic differences
   - Null handling differences
   - Default value differences

2. **Data Quality Rules**
   - Validation rules (NOT NULL, CHECK constraints)
   - Data cleansing logic (TRIM, UPPER, date parsing)
   - Deduplication logic (exact algorithm used)
   - Error handling (what happens on bad data)
   - Reject record handling

3. **Transformation Logic**
   - Join strategies (INNER vs LEFT vs BROADCAST)
   - Aggregation logic (GROUP BY, window functions)
   - Filter conditions (exact WHERE clauses)
   - Sort order (ORDER BY, partitioning)
   - Data type conversions (CAST logic)

4. **Field-Level Analysis**
   - Input → Output field mapping
   - Calculation formulas (show exact SQL/code)
   - Conditional logic (CASE WHEN statements)
   - String manipulations (CONCAT, SUBSTRING)
   - Math operations (ROUND, FLOOR, precision)

5. **Edge Cases & Error Handling**
   - Division by zero handling
   - NULL propagation in calculations
   - Date boundary cases (leap years, timezone)
   - Large number handling (overflow)
   - Empty string vs NULL handling

6. **Performance & Optimization**
   - Caching strategies
   - Partition/bucketing differences
   - Index usage
   - Broadcast vs shuffle joins
   - File format differences (Parquet, ORC, text)

7. **Data Discrepancies Risk**
   - Identify ANY logic that could produce different results
   - Flag precision loss (FLOAT vs DECIMAL)
   - Timezone conversion issues
   - Rounding differences
   - Character encoding differences

Provide analysis in JSON format:
{
  "are_equivalent": true/false,
  "similarity_score": 0.0-1.0,

  "pipeline_process_flow_comparison": {
    "system1_stages": [
      {"stage": 1, "name": "Read Source", "description": "Reads from table X"},
      {"stage": 2, "name": "Filter Records", "description": "Filters by condition Y"},
      {"stage": 3, "name": "Transform", "description": "Applies transformations Z"},
      {"stage": 4, "name": "Write Output", "description": "Writes to table W"}
    ],
    "system2_stages": [
      {"stage": 1, "name": "Read Source", "description": "Reads from table X"},
      {"stage": 2, "name": "Filter Records", "description": "Filters by condition Y"},
      {"stage": 3, "name": "Transform", "description": "Applies transformations Z"},
      {"stage": 4, "name": "Write Output", "description": "Writes to table W"}
    ],
    "stage_by_stage_comparison": [
      {
        "stage": "Data Ingestion",
        "system1": "Reads from Oracle table CUSTOMERS using JDBC",
        "system2": "Reads from Delta table customers using Spark",
        "are_similar": true,
        "differences": "Different source format but same logical data",
        "impact": "None - data content is identical"
      }
    ],
    "overall_flow_similarity": "HIGH|MEDIUM|LOW",
    "missing_stages_in_system1": ["List any stages present in system2 but not system1"],
    "missing_stages_in_system2": ["List any stages present in system1 but not system2"],
    "flow_summary": "Both pipelines follow the same 4-stage process: read, filter, transform, write"
  },

  "business_logic_summary": "Detailed comparison of business rules implemented in both systems...",

  "data_sources_and_targets": {
    "system1_data_flow": {
      "input_sources": ["table1", "table2", "file.csv"],
      "output_targets": ["output_table", "report.xlsx"],
      "intermediate_datasets": ["temp_table1", "staging_table2"]
    },
    "system2_data_flow": {
      "input_sources": ["delta_table1", "delta_table2", "s3://bucket/file.csv"],
      "output_targets": ["output_delta_table", "s3://reports/report.parquet"],
      "intermediate_datasets": ["temp_view1", "temp_view2"]
    },
    "source_comparison": [
      {
        "logical_source": "Customer Master Data",
        "system1": "Oracle table CUST_MASTER",
        "system2": "Delta table customer_master",
        "same_data": true,
        "notes": "Same logical data, different physical format"
      }
    ],
    "target_comparison": [
      {
        "logical_target": "Customer Dimension",
        "system1": "Teradata table DW_CUSTOMER_DIM",
        "system2": "Delta table customer_dimension",
        "same_schema": true,
        "column_differences": ["system2 has additional audit_timestamp column"]
      }
    ]
  },

  "field_level_analysis": [
    {
      "field_name": "customer_age",
      "system1_logic": "CAST((CURRENT_DATE - birth_date) / 365 AS INT)",
      "system2_logic": "FLOOR(DATEDIFF(CURRENT_DATE(), birth_date) / 365)",
      "are_equivalent": true,
      "potential_discrepancy": "Both use integer division, may differ by 1 day on leap years",
      "validation_needed": true
    }
  ],

  "business_rule_differences": [
    {
      "rule": "Customer status determination",
      "system1": "IF total_purchases > 1000 THEN 'PREMIUM' ELSE 'STANDARD'",
      "system2": "CASE WHEN total_purchases >= 1000 THEN 'PREMIUM' ELSE 'STANDARD' END",
      "impact": "CRITICAL: Boundary case difference - customer with exactly $1000 classified differently",
      "data_impact": "~50 customers/month potentially misclassified",
      "recommendation": "Align logic to use >= in both systems"
    }
  ],

  "data_quality_differences": [
    {
      "aspect": "Email validation",
      "system1": "REGEX_MATCH(email, '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$')",
      "system2": "email LIKE '%@%.%'",
      "impact": "System2 validation is MUCH weaker - allows invalid emails",
      "risk_level": "HIGH",
      "examples_affected": "emails like 'test@@example.com' pass system2, fail system1"
    }
  ],

  "transformation_differences": [
    {
      "operation": "Deduplication",
      "system1": "GROUP BY customer_id HAVING MAX(updated_date)",
      "system2": "ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY updated_date DESC) = 1",
      "are_equivalent": true,
      "performance_note": "System2 window function is faster for large datasets",
      "data_difference": "None - both keep latest record"
    }
  ],

  "join_logic_comparison": [
    {
      "join_name": "Customer to Orders",
      "system1": "INNER JOIN orders ON customer.id = orders.customer_id",
      "system2": "LEFT JOIN orders ON customer.id = orders.customer_id",
      "impact": "CRITICAL: System2 includes customers with no orders, System1 excludes them",
      "row_count_impact": "System2 produces ~10% more rows",
      "recommendation": "Verify business requirement - should customers without orders be included?"
    }
  ],

  "aggregation_differences": [
    {
      "metric": "total_revenue",
      "system1": "SUM(order_amount)",
      "system2": "SUM(COALESCE(order_amount, 0))",
      "are_equivalent": false,
      "impact": "System1 returns NULL if all amounts are NULL, System2 returns 0",
      "validation_query": "SELECT COUNT(*) FROM orders WHERE order_amount IS NULL"
    }
  ],

  "null_handling_differences": [
    {
      "field": "middle_name",
      "system1": "Uses NULL for missing middle names",
      "system2": "Uses empty string '' for missing middle names",
      "impact": "Downstream systems expecting NULL may break",
      "recommendation": "Standardize on NULL or add NULLIF('', '') in system2"
    }
  ],

  "critical_issues": [
    {
      "severity": "HIGH|MEDIUM|LOW",
      "issue": "Description of critical difference",
      "system1_behavior": "...",
      "system2_behavior": "...",
      "potential_data_loss": true/false,
      "affected_records": "Estimated number or percentage",
      "validation_query": "SQL to identify affected records"
    }
  ],

  "validation_queries": [
    {
      "purpose": "Compare row counts",
      "query": "SELECT COUNT(*) FROM table WHERE ..."
    },
    {
      "purpose": "Find data discrepancies",
      "query": "SELECT fields WHERE system1.value <> system2.value"
    }
  ],

  "performance_analysis": {
    "system1_strategy": "Detailed execution plan",
    "system2_strategy": "Detailed execution plan",
    "performance_difference": "System2 is 3x faster due to...",
    "optimization_opportunities": ["Add index on X", "Partition by Y"]
  },

  "overall_assessment": {
    "confidence_in_equivalence": "HIGH|MEDIUM|LOW",
    "major_concerns": ["List critical issues"],
    "minor_differences": ["List non-critical differences"],
    "recommendation": "APPROVE|REVIEW|REJECT migration with reasoning",
    "testing_priority": ["Test case 1", "Test case 2"]
  },

  "technology_comparison": {
    "note": "This section should be BRIEF and mentioned LAST - users already know systems use different technologies",
    "system1_technology_stack": "Ab Initio, DML, Co>Operating System",
    "system2_technology_stack": "Databricks, PySpark, Delta Lake",
    "technology_impact_on_logic": "Minimal - both implement same business logic despite different platforms",
    "platform_specific_features": [
      "System1 uses Ab Initio's parallel processing, System2 uses Spark's distributed processing - both achieve same result"
    ]
  }
}

CRITICAL REQUIREMENTS (in priority order):
1. **PROCESS FLOW FIRST**: Always start with pipeline_process_flow_comparison - show stages, what each stage does
2. **LOGIC SECOND**: Deep dive into business logic, transformations, rules - show ACTUAL code snippets
3. **DATA THIRD**: Compare data sources, targets, transformations at column level
4. **TECHNOLOGY LAST**: Keep technology comparison brief and at the end
5. Show ACTUAL code snippets, not just descriptions
6. Identify ANY difference that could cause data discrepancies
7. Provide SQL queries to validate differences
8. Flag precision/rounding/timezone issues
9. Compare exact formulas and calculations
10. Identify boundary condition differences (>, >=, <, <=)
11. Compare NULL handling everywhere
12. Show data type differences (INT vs BIGINT, FLOAT vs DECIMAL)

IMPORTANT: Users want to understand WHAT the pipelines do, not WHICH technology they use.
Focus on business logic, data flow, and transformations - not platform differences.

Be EXTREMELY detailed and technical on LOGIC and DATA, brief on TECHNOLOGY."""

    def compare_logic(
        self,
        system1: Dict[str, Any],
        system2: Dict[str, Any],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare logic between two systems

        Args:
            system1: Dict with 'system_name', 'code', 'description', 'metadata'
            system2: Dict with same structure
            context: Optional context about the comparison

        Returns:
            Dict with comparison results
        """
        if not self.enabled:
            return self._fallback_comparison(system1, system2)

        user_prompt = f"""Compare the following two ETL logic implementations:

**SYSTEM 1: {system1['system_name']}**
Name: {system1.get('name', 'N/A')}
Description: {system1.get('description', 'N/A')}

Code/Logic:
```
{system1.get('code', system1.get('logic', 'N/A'))[:5000]}
```

**SYSTEM 2: {system2['system_name']}**
Name: {system2.get('name', 'N/A')}
Description: {system2.get('description', 'N/A')}

Code/Logic:
```
{system2.get('code', system2.get('logic', 'N/A'))[:5000]}
```
"""

        if context:
            user_prompt += f"\n\n**Context**: {context}"

        user_prompt += "\n\nProvide detailed comparison in JSON format."

        try:
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=4000,  # INCREASED from 2000 to 4000 for in-depth comparison
            )

            content = response.choices[0].message.content

            # Parse JSON
            import json
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()

            comparison = json.loads(json_str)

            logger.info(f"✓ Compared {system1['system_name']} vs {system2['system_name']}")
            logger.info(f"  Similarity: {comparison.get('similarity_score', 0):.2f}")

            return comparison

        except Exception as e:
            logger.error(f"Logic comparison error: {e}")
            return self._fallback_comparison(system1, system2)

    def _fallback_comparison(self, system1: Dict[str, Any], system2: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback comparison without LLM"""
        return {
            "are_equivalent": False,
            "similarity_score": 0.5,
            "semantic_summary": f"Comparing {system1['system_name']} and {system2['system_name']} (AI analysis unavailable)",
            "similarities": ["Both are ETL processes"],
            "differences": [
                {
                    "aspect": "platform",
                    "system1": system1['system_name'],
                    "system2": system2['system_name'],
                    "impact": "Different platforms, manual comparison needed"
                }
            ],
            "migration_recommendation": {
                "difficulty": "unknown",
                "effort_estimate": "Requires detailed analysis",
                "key_challenges": ["Manual logic review needed"],
                "approach": "Enable Azure OpenAI for AI-powered comparison"
            }
        }

    def compare_batch(
        self,
        comparisons: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple logic pairs

        Args:
            comparisons: List of dicts with 'system1', 'system2', 'context'

        Returns:
            List of comparison results
        """
        results = []

        for i, comp in enumerate(comparisons, 1):
            logger.info(f"Comparing {i}/{len(comparisons)}...")

            result = self.compare_logic(
                system1=comp["system1"],
                system2=comp["system2"],
                context=comp.get("context")
            )

            results.append(result)

        return results

    def detect_migration_opportunities(
        self,
        source_system: str,
        target_system: str,
        source_logic_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze migration opportunities from source to target

        Args:
            source_system: Source system name (e.g., "Ab Initio")
            target_system: Target system name (e.g., "Databricks")
            source_logic_list: List of logic components from source

        Returns:
            Dict with migration analysis
        """
        if not self.enabled:
            return {
                "feasibility": "unknown",
                "total_components": len(source_logic_list),
                "recommendation": "Enable Azure OpenAI for migration analysis"
            }

        # Sample components for analysis (avoid hitting token limits)
        sample_size = min(10, len(source_logic_list))
        samples = source_logic_list[:sample_size]

        user_prompt = f"""Analyze migration feasibility from {source_system} to {target_system}.

Source components ({sample_size} of {len(source_logic_list)}):
"""

        for i, comp in enumerate(samples, 1):
            user_prompt += f"\n{i}. {comp.get('name', 'N/A')}: {comp.get('description', 'N/A')[:200]}"

        user_prompt += f"""

Provide migration analysis in JSON:
{{
  "feasibility": "low|medium|high",
  "estimated_effort": "X weeks/months",
  "components_easy": X,
  "components_medium": X,
  "components_hard": X,
  "key_challenges": ["challenge 1", ...],
  "recommended_approach": "Step by step migration plan",
  "automation_potential": "percentage of components that can be auto-migrated",
  "manual_work_required": "What needs manual work",
  "risk_assessment": "Migration risks and mitigation"
}}"""

        try:
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500,
            )

            content = response.choices[0].message.content

            import json
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            else:
                json_str = content

            analysis = json.loads(json_str)
            analysis["total_components"] = len(source_logic_list)
            analysis["sample_size"] = sample_size

            return analysis

        except Exception as e:
            logger.error(f"Migration analysis error: {e}")
            return {
                "feasibility": "unknown",
                "error": str(e),
                "total_components": len(source_logic_list)
            }


# Convenience function
def create_logic_comparator() -> LogicComparator:
    """Create logic comparator instance"""
    return LogicComparator()


# Example usage
if __name__ == "__main__":
    comparator = LogicComparator()

    # Mock comparison
    system1 = {
        "system_name": "Ab Initio",
        "name": "customer_load.graph",
        "description": "Loads customer data from staging to dimension",
        "code": """
        TRANSFORM
          lookup_join(customer_staging, customer_master, customer_id)
          aggregate(customer_id, sum(amount))
          filter(status == 'ACTIVE')
        """
    }

    system2 = {
        "system_name": "Hadoop Spark",
        "name": "customer_load_spark.py",
        "description": "Loads customer data using PySpark",
        "code": """
        df_staging = spark.read.table("customer_staging")
        df_master = spark.read.table("customer_master")

        df_joined = df_staging.join(broadcast(df_master), "customer_id", "inner")
        df_agg = df_joined.groupBy("customer_id").agg(sum("amount"))
        df_filtered = df_agg.filter(col("status") == "ACTIVE")
        """
    }

    if comparator.enabled:
        result = comparator.compare_logic(system1, system2)

        print("\nLogic Comparison Result:")
        print(f"Equivalent: {result.get('are_equivalent')}")
        print(f"Similarity: {result.get('similarity_score', 0):.2f}")
        print(f"\nSummary: {result.get('semantic_summary')}")
    else:
        print("Logic Comparator not enabled (Azure OpenAI required)")
