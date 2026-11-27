"""FastAPI application. User answers are never stored or logged."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router
from app.core.config import get_settings
from app.ml.model import get_model

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Fail at startup if the artifact is missing or broken.
    logging.getLogger(__name__).info("model %s loaded", get_model().version)
    yield


app = FastAPI(
    title=settings.app_name, version="2.0.0", lifespan=lifespan,
    docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json",
)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)
app.include_router(router, prefix="/api/v1")
