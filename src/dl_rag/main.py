"""FastAPI application entrypoint.

Run: ``uvicorn dl_rag.main:app --host 0.0.0.0 --port 8000``
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from dl_rag import __version__
from dl_rag.api.deps import build_container
from dl_rag.api.errors import register_exception_handlers
from dl_rag.api.middleware import RateLimitMiddleware, RequestContextMiddleware
from dl_rag.api.routers import admin, chat, documents, feedback, index, system, ui
from dl_rag.config import get_settings
from dl_rag.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    container = build_container(settings)
    app.state.container = container
    app.state.cache = container.cache

    # Best-effort schema creation — guarded so the API still boots (degraded)
    # when a datastore is briefly unavailable. Use Alembic for real migrations.
    try:
        await container.db.create_all()
        logger.info("startup.schema_ready")
    except Exception as exc:  # noqa: BLE001
        logger.error("startup.create_all_failed", error=str(exc))

    logger.info(
        "startup.ready",
        version=__version__,
        environment=settings.environment,
        require_auth=settings.require_auth,
    )
    try:
        yield
    finally:
        try:
            await container.db.dispose()
        except Exception:  # noqa: BLE001
            pass
        try:
            await container.cache.close()
        except Exception:  # noqa: BLE001
            pass
        logger.info("shutdown.complete")


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Retrieval-Augmented Generation API over the digitalLEARNING archive "
            "(2005–present). Ask education-policy questions in natural language and "
            "get grounded, cited, analyst-style answers."
        ),
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Middleware — added last is outermost, so RequestContext wraps RateLimit and
    # request_id is bound before the limiter can reject a request.
    app.add_middleware(
        RateLimitMiddleware,
        limit=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    app.include_router(system.router)
    app.include_router(ui.router)
    prefix = settings.api_prefix
    app.include_router(chat.router, prefix=prefix)
    app.include_router(index.router, prefix=prefix)
    app.include_router(documents.router, prefix=prefix)
    app.include_router(feedback.router, prefix=prefix)
    app.include_router(admin.router, prefix=prefix)

    return app


app = create_app()
