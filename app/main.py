from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pymongo import MongoClient
from app.core.config import settings
from app.api.routes import router
from app.core.handlers import register_exception_handlers
from app.core.middleware import TokenAuthMiddleware
from app.core.admin_middleware import AdminAuthMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo_client = MongoClient(settings.MONGO_URI)
    app.state.mongo_client = mongo_client
    app.state.db = mongo_client[settings.MONGO_DB]
    try:
        yield
    finally:
        mongo_client.close()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="FastAPI backend for Finlens AI Hackathon",
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)


# Custom OpenAPI schema with Bearer token security
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="FastAPI backend for Finlens AI Hackathon",
        routes=app.routes,
    )

    openapi_schema["components"]["securitySchemes"] = {
        "Bearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter a valid JWT token",
        }
    }

    # Apply security to all routes except exempt ones
    for path, path_item in openapi_schema["paths"].items():
        # Exempt public endpoints
        if path in ["/auth/login", "/auth/signup", "/health"]:
            continue

        for operation in path_item.values():
            if isinstance(operation, dict) and "operationId" in operation:
                operation.setdefault("security", [{"Bearer": []}])

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Register exception handlers
register_exception_handlers(app)

# Register user token auth middleware. Exempt public endpoints (login, signup, health, and docs).
app.add_middleware(
    TokenAuthMiddleware,
)

# Register admin auth middleware. Only applies to /admin/* routes.
app.add_middleware(
    AdminAuthMiddleware,
)

app.include_router(router)


@app.get(
    "/health",
    tags=["health"],
    summary="Health Check",
    description="Check if the API is running",
)
def health():
    """
    Health check endpoint to verify the API is running.

    Returns:
        dict: Status of the API
    """
    return {"status": "ok"}
