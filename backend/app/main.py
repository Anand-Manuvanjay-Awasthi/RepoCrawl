"""
RepoCrawl FastAPI application entrypoint.

Phase 1 wires up: app creation, CORS, structured error handling, logging,
and the health endpoint. Repository indexing and chat endpoints are added
in later phases (see app/api/routes/).
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import health
from app.core.config import get_settings
from app.core.exceptions import RepoCrawlError
from app.core.logging import configure_logging, get_logger
from app.models.responses import ErrorDetail, ErrorResponse

configure_logging()
logger = get_logger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Ask natural-language questions about a public GitHub repository, "
    "answered strictly from retrieved repository evidence (RAG).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    settings.ensure_runtime_dirs()
    logger.info("%s starting up (environment=%s)", settings.app_name, settings.environment)


@app.exception_handler(RepoCrawlError)
def handle_repocrawl_error(request: Request, exc: RepoCrawlError) -> JSONResponse:
    """Translate known application errors into a structured JSON error envelope."""
    logger.warning("RepoCrawlError [%s] on %s: %s", exc.code, request.url.path, exc.message)
    payload = ErrorResponse(
        error=ErrorDetail(code=exc.code, message=exc.message, context=exc.context)
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(Exception)
def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so unhandled errors still return the structured error shape."""
    logger.exception("Unhandled error on %s", request.url.path)
    payload = ErrorResponse(
        error=ErrorDetail(code="internal_error", message="An unexpected error occurred.")
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


app.include_router(health.router, prefix="/api")


@app.get("/")
def root() -> dict:
    return {"service": settings.app_name, "status": "running"}
