from contextlib import asynccontextmanager
from fastapi import FastAPI
from pymongo import MongoClient
from app.core.config import settings
from app.api.routes import router
from app.core.handlers import register_exception_handlers

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
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    tags_metadata=[
        {
            "name": "health",
            "description": "Health check endpoints",
        },
        {
            "name": "auth",
            "description": "User authentication endpoints",
        },
        {
            "name": "tests",
            "description": "Test CRUD operations",
        },
    ],
)

# Register exception handlers
register_exception_handlers(app)

app.include_router(router)

@app.get("/health", tags=["health"], summary="Health Check", description="Check if the API is running")
def health():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        dict: Status of the API
    """
    return {"status": "ok"}
