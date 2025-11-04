"""
AI-Powered Script Analyzer
Uses Azure OpenAI GPT-4 to understand and explain script logic
"""

import os
from typing import Dict, List, Optional
from loguru import logger

try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available - AI analysis will be skipped")

from core.models.script_logic import ScriptLogic


class AIScriptAnalyzer:
    """
    Use GPT-4 to analyze scripts and extract business logic

    This is what makes the system INTELLIGENT!
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize AI analyzer"""
        self.enabled = False
        self.client = None

        if OPENAI_AVAILABLE and (openai_api_key or os.getenv("AZURE_OPENAI_API_KEY")):
            try:
                self.client = AzureOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=openai_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                )
                self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
                self.enabled = True
                logger.info("✓ AI Script Analyzer initialized with GPT-4")
            except Exception as e:
                logger.warning(f"Could not initialize AI analyzer: {e}")
                self.enabled = False
        else:
            logger.info("AI Script Analyzer disabled (no API key or OpenAI not available)")

    def analyze_script(self, script_logic: ScriptLogic) -> None:
        """
        Analyze script and add AI insights

        Modifies script_logic in place, adding:
        - business_purpose
        - business_logic_summary
        - ai_logic_summary
        - ai_optimization_suggestions
        - ai_potential_issues
        - key_business_rules
        """
        if not self.enabled:
            return

        script_type = script_logic.script_type
        script_content = script_logic.raw_content[:4000]  # First 4000 chars (GPT-4 context limit consideration)

        # Create analysis prompt based on script type
        if script_type == "pig":
            analysis = self._analyze_pig_script(script_content, script_logic)
        elif script_type in ["pyspark", "spark", "scala_spark"]:
            analysis = self._analyze_spark_script(script_content, script_logic)
        elif script_type in ["hive", "sql"]:
            analysis = self._analyze_sql_script(script_content, script_logic)
        else:
            analysis = self._analyze_generic_script(script_content, script_logic)

        if analysis:
            # Update script_logic with AI insights
            script_logic.business_purpose = analysis.get("business_purpose")
            script_logic.business_logic_summary = analysis.get("business_logic_summary")
            script_logic.ai_logic_summary = analysis.get("detailed_analysis")
            script_logic.ai_optimization_suggestions = analysis.get("optimization_suggestions", [])
            script_logic.ai_potential_issues = analysis.get("potential_issues", [])
            script_logic.key_business_rules = analysis.get("key_business_rules", [])

            # Update transformation business meanings
            for i, trans in enumerate(script_logic.transformations):
                if i < len(analysis.get("transformation_meanings", [])):
                    trans.business_meaning = analysis["transformation_meanings"][i]

            logger.info(f"✓ AI analyzed script: {script_logic.script_name}")

    def _analyze_pig_script(self, content: str, script_logic: ScriptLogic) -> Dict:
        """Analyze Pig Latin script"""

        system_prompt = """You are an expert data engineer specializing in Apache Pig and healthcare data pipelines.

Analyze Pig Latin scripts to understand:
1. Business purpose (what business problem does this solve?)
2. Data transformations (what operations are performed?)
3. Business rules (filtering logic, join logic, calculations)
4. Potential issues (data quality, performance, null handling)
5. Optimization opportunities

Be specific and use domain knowledge about healthcare claims, patient data, and coverage discovery.
Focus on BUSINESS MEANING, not just technical details."""

        transformations_summary = ""
        for trans in script_logic.transformations[:10]:  # First 10
            transformations_summary += f"- {trans.transformation_type.value}: {trans.code_snippet[:100]}\n"

        user_prompt = f"""Analyze this Pig Latin script:

Script Name: {script_logic.script_name}
Input Tables/Files: {', '.join(script_logic.input_tables + script_logic.input_files)}
Output Tables/Files: {', '.join(script_logic.output_tables + script_logic.output_files)}

Transformations Found:
{transformations_summary}

Script Content (first 4000 chars):
```pig
{content}
```

Provide analysis in this JSON format:
{{
  "business_purpose": "One sentence: what business problem does this script solve?",
  "business_logic_summary": "2-3 sentences: detailed business logic explanation",
  "detailed_analysis": "Comprehensive analysis of what the script does and why",
  "key_business_rules": ["rule 1", "rule 2", ...],
  "transformation_meanings": ["meaning of transformation 1", "meaning of transformation 2", ...],
  "data_quality_impact": "How does this script affect data quality?",
  "optimization_suggestions": ["suggestion 1", "suggestion 2", ...],
  "potential_issues": ["issue 1", "issue 2", ...]
}}

Focus on BUSINESS MEANING - help someone understand WHY this script exists and WHAT it accomplishes."""

        return self._call_gpt4(system_prompt, user_prompt)

    def _analyze_spark_script(self, content: str, script_logic: ScriptLogic) -> Dict:
        """Analyze PySpark/Spark script"""

        system_prompt = """You are an expert data engineer specializing in Apache Spark and big data processing.

Analyze Spark scripts (PySpark or Scala) to understand business logic, data flow, and optimization opportunities.
Focus on what the code accomplishes from a business perspective."""

        user_prompt = f"""Analyze this Spark script:

Script Name: {script_logic.script_name}
Type: {script_logic.script_type}
Inputs: {', '.join(script_logic.input_tables + script_logic.input_files)}
Outputs: {', '.join(script_logic.output_tables + script_logic.output_files)}

Code (first 4000 chars):
```python
{content}
```

Provide business-focused analysis in JSON format:
{{
  "business_purpose": "One sentence purpose",
  "business_logic_summary": "Detailed business logic",
  "detailed_analysis": "Comprehensive analysis",
  "key_business_rules": ["rule 1", ...],
  "optimization_suggestions": ["suggestion 1", ...],
  "potential_issues": ["issue 1", ...]
}}"""

        return self._call_gpt4(system_prompt, user_prompt)

    def _analyze_sql_script(self, content: str, script_logic: ScriptLogic) -> Dict:
        """Analyze SQL/Hive script"""

        system_prompt = """You are an expert SQL analyst specializing in data analytics and healthcare data."""

        user_prompt = f"""Analyze this SQL script:

Script: {script_logic.script_name}
Code:
```sql
{content}
```

Provide business analysis in JSON format with business_purpose, business_logic_summary, detailed_analysis, key_business_rules, optimization_suggestions, and potential_issues."""

        return self._call_gpt4(system_prompt, user_prompt)

    def _analyze_generic_script(self, content: str, script_logic: ScriptLogic) -> Dict:
        """Analyze generic script"""

        system_prompt = """You are an expert data engineer analyzing data processing scripts."""

        user_prompt = f"""Analyze this {script_logic.script_type} script:

{content[:2000]}

Provide brief business analysis."""

        return self._call_gpt4(system_prompt, user_prompt)

    def analyze_abinitio_component(self, component_data: Dict, script_logic: ScriptLogic = None) -> Dict:
        """
        Analyze Ab Initio component using AI (NEW - FAWN enhancement)

        Args:
            component_data: Component dictionary with type, name, parameters
            script_logic: Optional ScriptLogic object to update

        Returns:
            Dictionary with AI analysis results
        """
        if not self.enabled:
            return {}

        comp_type = component_data.get("component_type", "Unknown")
        comp_name = component_data.get("component_name", "Unknown")
        parameters = component_data.get("parameters", [])

        # Build parameter summary
        param_summary = ""
        for param in parameters[:20]:  # First 20 parameters
            param_name = param.get("parameter_name", "")
            param_value = param.get("parameter_value", "")
            if param_value:
                param_summary += f"  - {param_name}: {param_value[:100]}\n"

        system_prompt = """You are an expert Ab Initio developer specializing in graph design and ETL processes.

Analyze Ab Initio components to understand:
1. Business purpose (what business function does this component serve?)
2. Data transformation logic (what operations are performed?)
3. DML schemas and data structures
4. Business rules embedded in transform expressions
5. Potential data quality issues
6. Performance optimization opportunities

Focus on BUSINESS MEANING and explain in terms non-technical stakeholders can understand."""

        user_prompt = f"""Analyze this Ab Initio {comp_type} component:

Component Name: {comp_name}
Component Type: {comp_type}

Parameters:
{param_summary}

Provide analysis in JSON format:
{{
  "business_purpose": "One sentence: what business purpose does this component serve?",
  "business_logic_summary": "2-3 sentences: detailed business logic explanation",
  "detailed_analysis": "Comprehensive analysis of what the component does and why",
  "key_transformations": ["transformation 1", "transformation 2", ...],
  "data_quality_considerations": "How does this component impact data quality?",
  "optimization_suggestions": ["suggestion 1", "suggestion 2", ...],
  "potential_issues": ["issue 1", "issue 2", ...]
}}

Focus on making the technical details understandable from a business perspective."""

        analysis = self._call_gpt4(system_prompt, user_prompt)

        # Update script_logic if provided
        if script_logic and analysis:
            script_logic.business_purpose = analysis.get("business_purpose")
            script_logic.business_logic_summary = analysis.get("business_logic_summary")
            script_logic.ai_logic_summary = analysis.get("detailed_analysis")
            script_logic.ai_optimization_suggestions = analysis.get("optimization_suggestions", [])
            script_logic.ai_potential_issues = analysis.get("potential_issues", [])

        return analysis

    def analyze_abinitio_graph(self, graph_data: Dict) -> Dict:
        """
        Analyze complete Ab Initio graph using AI (NEW - FAWN enhancement)

        Args:
            graph_data: Graph dictionary with components, parameters, flows

        Returns:
            Dictionary with graph-level AI analysis
        """
        if not self.enabled:
            return {}

        graph_name = graph_data.get("file_name", "Unknown")
        components = graph_data.get("components", [])
        graph_params = graph_data.get("graph_parameters", [])
        flows = graph_data.get("graph_flow", [])

        # Build component summary
        comp_summary = ""
        for comp in components[:15]:  # First 15 components
            comp_summary += f"  - {comp.get('component_name', 'Unknown')} ({comp.get('component_type', 'Unknown')})\n"

        # Build parameter summary
        param_summary = ""
        for param in graph_params[:10]:  # First 10 parameters
            param_summary += f"  - {param.get('parameter_name', '')}: {param.get('parameter_value', '')[:80]}\n"

        system_prompt = """You are a senior Ab Initio architect analyzing ETL graphs.

Analyze Ab Initio graphs to understand:
1. Overall business purpose and value
2. Data flow and transformations
3. Integration patterns
4. Architecture and design patterns
5. Business rules and logic
6. Performance and scalability considerations

Provide insights that help stakeholders understand the business value and technical design."""

        user_prompt = f"""Analyze this Ab Initio graph:

Graph Name: {graph_name}
Total Components: {len(components)}
Total Flows: {len(flows)}

Components:
{comp_summary}

Graph Parameters:
{param_summary}

Provide analysis in JSON format:
{{
  "business_purpose": "What is the overall business purpose of this graph?",
  "data_flow_summary": "Describe the data flow through this graph",
  "key_transformations": "What are the main data transformations?",
  "business_value": "What business value does this graph provide?",
  "architecture_pattern": "What architecture/design pattern is used?",
  "optimization_suggestions": ["suggestion 1", ...],
  "complexity_assessment": "low/medium/high and explanation"
}}"""

        return self._call_gpt4(system_prompt, user_prompt)

    def analyze_autosys_job(self, job_data: Dict) -> Dict:
        """
        Analyze Autosys job using AI (NEW - FAWN enhancement)

        Args:
            job_data: Job dictionary with name, command, dependencies

        Returns:
            Dictionary with AI analysis results
        """
        if not self.enabled:
            return {}

        job_name = job_data.get("job_name", "Unknown")
        command = job_data.get("command", "")
        job_type = job_data.get("job_type", "CMD")
        condition = job_data.get("condition", "")
        description = job_data.get("description", "")

        system_prompt = """You are an expert in Autosys job scheduling and ETL orchestration.

Analyze Autosys jobs to understand:
1. Business purpose and scheduling intent
2. Integration with downstream systems (Ab Initio, scripts, etc.)
3. Dependencies and execution order
4. Business criticality and SLA requirements
5. Error handling and recovery strategies

Provide insights that help stakeholders understand job orchestration and business impact."""

        user_prompt = f"""Analyze this Autosys job:

Job Name: {job_name}
Job Type: {job_type}
Command: {command[:200]}
Dependencies: {condition}
Description: {description}

Provide analysis in JSON format:
{{
  "business_purpose": "What is the business purpose of this job?",
  "scheduling_intent": "Why is this job scheduled? What triggers it?",
  "downstream_systems": "What systems or processes does this job interact with?",
  "business_criticality": "high/medium/low and explanation",
  "optimization_suggestions": ["suggestion 1", ...],
  "potential_issues": ["issue 1", ...]
}}"""

        return self._call_gpt4(system_prompt, user_prompt)

    def analyze_with_context(self, query: str, context: str) -> Dict:
        """
        General-purpose method to answer questions with context

        Args:
            query: User's question
            context: Context information to help answer

        Returns:
            Dict with 'analysis' or 'response' key containing the answer
        """
        if not self.enabled:
            return {
                "analysis": "AI analysis is not available. Please configure Azure OpenAI credentials in .env file.",
                "response": "AI is not enabled. Search results show what was found in the vector database, but AI analysis requires Azure OpenAI configuration."
            }

        try:
            system_prompt = """You are an expert code analyst for data pipeline systems (Ab Initio, Hadoop, Databricks, Autosys).
Your job is to answer user questions based on the provided context from the codebase.
Provide clear, concise, and accurate answers. If the context doesn't contain enough information, say so."""

            # Combine context and query
            if context:
                user_prompt = f"""Context from codebase:
{context[:3000]}

User Question: {query}

Please provide a clear answer based on the context above."""
            else:
                user_prompt = query

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            content = response.choices[0].message.content

            return {
                "analysis": content,
                "response": content
            }

        except Exception as e:
            logger.error(f"Error in analyze_with_context: {e}")
            return {
                "analysis": f"I found some relevant information but couldn't generate a detailed analysis. Error: {str(e)}",
                "response": "AI analysis encountered an error. Please check your Azure OpenAI configuration."
            }

    def _call_gpt4(self, system_prompt: str, user_prompt: str) -> Optional[Dict]:
        """Call GPT-4 and parse response"""
        if not self.enabled:
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for factual analysis
                max_tokens=2000,
            )

            content = response.choices[0].message.content

            # Try to parse JSON response
            import json
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()

            return json.loads(json_str)

        except Exception as e:
            logger.error(f"Error calling GPT-4: {e}")
            # Fallback: return content as detailed_analysis
            return {
                "detailed_analysis": content if 'content' in locals() else "AI analysis failed"
            }

    def analyze_workflow_flow(self, workflow_flow) -> str:
        """Analyze entire workflow flow and generate high-level summary"""
        if not self.enabled:
            return None

        system_prompt = """You are an expert in data pipeline architecture.
Analyze workflow flows and explain the high-level business purpose and data flow."""

        actions_summary = "\n".join([
            f"- {action.action_name} ({action.action_type.value}): {action.script_path or 'N/A'}"
            for action in workflow_flow.actions[:20]
        ])

        user_prompt = f"""Analyze this workflow:

Workflow: {workflow_flow.workflow_name}
Type: {workflow_flow.workflow_type}

Actions:
{actions_summary}

Overall Inputs: {', '.join(workflow_flow.overall_inputs[:10])}
Overall Outputs: {', '.join(workflow_flow.overall_outputs[:10])}

Provide a 2-3 sentence business summary of what this workflow accomplishes and how data flows through it."""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error analyzing workflow: {e}")
            return None

    def analyze_repository(self, repository) -> Dict:
        """Analyze entire repository and generate insights"""
        if not self.enabled:
            return {}

        system_prompt = """You are a principal data architect analyzing entire code repositories."""

        user_prompt = f"""Analyze this repository:

Repository: {repository.name}
Type: {repository.repo_type.value}
Workflows: {repository.total_workflows}
Scripts: {repository.total_scripts} ({repository.pig_scripts} Pig, {repository.spark_scripts} Spark, {repository.hive_scripts} Hive)

Business Domains: {', '.join(repository.business_domains)}
Technologies: {', '.join(repository.technologies)}

Provide JSON with:
- architecture_summary: High-level architecture description
- business_value: What business value does this repository provide?
- key_capabilities: List of main capabilities
- recommendations: Architecture improvement suggestions"""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            content = response.choices[0].message.content

            import json
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            else:
                json_str = content

            return json.loads(json_str)

        except Exception as e:
            logger.error(f"Error analyzing repository: {e}")
            return {}
