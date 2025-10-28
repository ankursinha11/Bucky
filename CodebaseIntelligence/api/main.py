"""
FastAPI REST API for Codebase Intelligence Platform
Provides endpoints for parsing, STTM generation, gap analysis, and chatbot queries
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid
from loguru import logger

# Import our modules
from parsers.abinitio.enhanced_parser import EnhancedAbInitioParser
from parsers.hadoop import HadoopParser
from parsers.databricks import DatabricksParser
from core.sttm_generator import STTMGenerator
from core.gap_analyzer import GapAnalyzer
from core.matchers import ProcessMatcher
from utils import ExcelExporter
from services.azure_search.search_client import CodebaseSearchClient
from services.openai.rag_chatbot import CodebaseRAGChatbot

# Initialize FastAPI
app = FastAPI(
    title="Codebase Intelligence API",
    description="API for parsing codebases, generating STTM, analyzing gaps, and querying with natural language",
    version="1.0.0",
)

# Initialize services
search_client = None  # Will be initialized when needed
chatbot = None  # Will be initialized when needed

# Storage for jobs and results
jobs = {}  # job_id -> status
results_store = {}  # job_id -> results


# ============================================================================
# Request/Response Models
# ============================================================================


class ParseRequest(BaseModel):
    path: str
    system: str  # "abinitio", "hadoop", "databricks"

    class Config:
        schema_extra = {
            "example": {
                "path": "/path/to/repository",
                "system": "databricks"
            }
        }


class STTMRequest(BaseModel):
    process_ids: List[str]
    output_format: str = "excel"  # "excel", "json"


class GapAnalysisRequest(BaseModel):
    source_system: str
    target_system: str
    source_process_ids: List[str]
    target_process_ids: List[str]


class ChatQuery(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: Optional[int] = None
    result: Optional[Any] = None
    error: Optional[str] = None


# ============================================================================
# Health Check
# ============================================================================


@app.get("/")
async def root():
    """Health check"""
    return {
        "name": "Codebase Intelligence API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "azure_search": "configured" if search_client else "not configured",
            "chatbot": "configured" if chatbot else "not configured",
        },
    }


# ============================================================================
# Parsing Endpoints
# ============================================================================


@app.post("/parse/abinitio")
async def parse_abinitio(request: ParseRequest, background_tasks: BackgroundTasks):
    """Parse Ab Initio files"""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0}

    background_tasks.add_task(_parse_abinitio_task, job_id, request.path)

    return {"job_id": job_id, "message": "Parsing started"}


def _parse_abinitio_task(job_id: str, path: str):
    """Background task for parsing Ab Initio"""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 10

        parser = EnhancedAbInitioParser()

        jobs[job_id]["progress"] = 30

        # Parse files
        result = parser.parse_file(path)

        jobs[job_id]["progress"] = 80

        # Store results
        results_store[job_id] = {
            "process": result["process"].to_dict(),
            "components": [c.to_dict() for c in result["components"]],
            "flows": [{"source": f.source_component_id, "target": f.target_component_id} for f in result["flows"]],
        }

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100

    except Exception as e:
        logger.error(f"Error in parse task {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.post("/parse/hadoop")
async def parse_hadoop(request: ParseRequest, background_tasks: BackgroundTasks):
    """Parse Hadoop repository"""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0}

    background_tasks.add_task(_parse_hadoop_task, job_id, request.path)

    return {"job_id": job_id, "message": "Parsing started"}


def _parse_hadoop_task(job_id: str, path: str):
    """Background task for parsing Hadoop"""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 10

        parser = HadoopParser()

        jobs[job_id]["progress"] = 30

        result = parser.parse_directory(path)

        jobs[job_id]["progress"] = 80

        results_store[job_id] = {
            "processes": [p.to_dict() for p in result["processes"]],
            "components": [c.to_dict() for c in result["components"]],
        }

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100

    except Exception as e:
        logger.error(f"Error in parse task {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.post("/parse/databricks")
async def parse_databricks(request: ParseRequest, background_tasks: BackgroundTasks):
    """Parse Databricks notebooks and ADF pipelines"""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0}

    background_tasks.add_task(_parse_databricks_task, job_id, request.path)

    return {"job_id": job_id, "message": "Parsing started"}


def _parse_databricks_task(job_id: str, path: str):
    """Background task for parsing Databricks"""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 10

        parser = DatabricksParser()

        jobs[job_id]["progress"] = 30

        result = parser.parse_directory(path)

        jobs[job_id]["progress"] = 80

        results_store[job_id] = {
            "processes": [p.to_dict() for p in result["processes"]],
            "components": [c.to_dict() for c in result["components"]],
        }

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100

    except Exception as e:
        logger.error(f"Error in parse task {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# ============================================================================
# STTM Generation Endpoints
# ============================================================================


@app.post("/sttm/generate")
async def generate_sttm(request: STTMRequest, background_tasks: BackgroundTasks):
    """Generate STTM report"""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0}

    background_tasks.add_task(_generate_sttm_task, job_id, request.process_ids, request.output_format)

    return {"job_id": job_id, "message": "STTM generation started"}


def _generate_sttm_task(job_id: str, process_ids: List[str], output_format: str):
    """Background task for STTM generation"""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 20

        generator = STTMGenerator()

        # Generate STTM (simplified for now)
        # In production, would retrieve processes from database

        jobs[job_id]["progress"] = 60

        # Generate report
        # report = generator.generate_from_process(process, components)

        jobs[job_id]["progress"] = 90

        # Export
        if output_format == "excel":
            exporter = ExcelExporter()
            # file_path = exporter.export_sttm_report(report)
            # results_store[job_id] = {"file_path": file_path}

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100

    except Exception as e:
        logger.error(f"Error in STTM task {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# ============================================================================
# Gap Analysis Endpoints
# ============================================================================


@app.post("/gaps/analyze")
async def analyze_gaps(request: GapAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze gaps between systems"""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0}

    background_tasks.add_task(
        _analyze_gaps_task,
        job_id,
        request.source_system,
        request.target_system,
        request.source_process_ids,
        request.target_process_ids,
    )

    return {"job_id": job_id, "message": "Gap analysis started"}


def _analyze_gaps_task(
    job_id: str,
    source_system: str,
    target_system: str,
    source_process_ids: List[str],
    target_process_ids: List[str],
):
    """Background task for gap analysis"""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 20

        # Match processes
        matcher = ProcessMatcher()

        jobs[job_id]["progress"] = 40

        # Analyze gaps
        analyzer = GapAnalyzer()

        jobs[job_id]["progress"] = 70

        # gaps = analyzer.analyze(...)

        jobs[job_id]["progress"] = 90

        # results_store[job_id] = {"gaps": [g.to_dict() for g in gaps]}

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100

    except Exception as e:
        logger.error(f"Error in gap analysis task {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# ============================================================================
# Chatbot Endpoints
# ============================================================================


@app.post("/chat/query")
async def chat_query(query: ChatQuery):
    """Query the codebase with natural language"""
    global chatbot

    if not chatbot:
        # Initialize on first use
        try:
            chatbot = CodebaseRAGChatbot()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Chatbot not available: {str(e)}. Please configure Azure OpenAI and Azure Search.",
            )

    try:
        result = chatbot.query(query.question, context_filters=query.filters)
        return result
    except Exception as e:
        logger.error(f"Error in chat query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/ask-about-process/{process_name}")
async def ask_about_process(process_name: str, system: Optional[str] = None):
    """Ask about a specific process"""
    global chatbot

    if not chatbot:
        chatbot = CodebaseRAGChatbot()

    try:
        result = chatbot.ask_about_process(process_name, system)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/find-sttm")
async def find_sttm(
    source_table: Optional[str] = None,
    target_table: Optional[str] = None,
    column_name: Optional[str] = None,
):
    """Find STTM mappings"""
    global chatbot

    if not chatbot:
        chatbot = CodebaseRAGChatbot()

    try:
        result = chatbot.find_sttm(source_table, target_table, column_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/find-gaps")
async def find_gaps(
    source_system: Optional[str] = None,
    target_system: Optional[str] = None,
    severity: Optional[str] = None,
):
    """Find gaps"""
    global chatbot

    if not chatbot:
        chatbot = CodebaseRAGChatbot()

    try:
        result = chatbot.find_gaps(source_system, target_system, severity)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Job Status Endpoints
# ============================================================================


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job_info = jobs[job_id]

    response = {
        "job_id": job_id,
        "status": job_info["status"],
        "progress": job_info.get("progress"),
    }

    if job_info["status"] == "completed" and job_id in results_store:
        response["result"] = results_store[job_id]

    if job_info["status"] == "failed":
        response["error"] = job_info.get("error")

    return response


# ============================================================================
# Report Download Endpoints
# ============================================================================


@app.get("/reports/list")
async def list_reports():
    """List available reports"""
    output_dir = Path("./outputs/reports")

    if not output_dir.exists():
        return {"reports": []}

    reports = []
    for file in output_dir.glob("*.xlsx"):
        reports.append(
            {
                "filename": file.name,
                "size": file.stat().st_size,
                "modified": file.stat().st_mtime,
            }
        )

    return {"reports": reports}


@app.get("/reports/download/{filename}")
async def download_report(filename: str):
    """Download a report"""
    file_path = Path(f"./outputs/reports/{filename}")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        path=file_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )


# ============================================================================
# Search Endpoints
# ============================================================================


@app.get("/search")
async def search(
    query: str,
    doc_type: Optional[str] = None,
    system: Optional[str] = None,
    top: int = 10,
):
    """Search the index"""
    global search_client

    if not search_client:
        try:
            search_client = CodebaseSearchClient()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Search not available: {str(e)}",
            )

    try:
        filters = []
        if doc_type:
            filters.append(f"doc_type eq '{doc_type}'")
        if system:
            filters.append(f"system eq '{system}'")

        filter_str = " and ".join(filters) if filters else None

        results = search_client.search(query, filters=filter_str, top=top)
        return {"results": results, "count": len(results)}

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
