"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import auth, documents, evaluation, health, query, review
from app.config import cors_origins, settings
from app.database import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized (%s).", settings.database_url)
    yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "Hybrid RAG + citation verification, multimodal document intelligence, "
        "and agentic retrieval - sharing one document database and retrieval stack."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})


app.include_router(health.router)
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(review.router)
app.include_router(evaluation.router)


@app.get("/")
def root():
    return {
        "app": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "modules": [
            "Module 1: Hybrid RAG + Citation Verification",
            "Module 2: Multimodal Document Intelligence",
            "Module 3: Agentic RAG",
        ],
    }
