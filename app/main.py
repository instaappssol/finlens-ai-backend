from contextlib import asynccontextmanager
from fastapi import FastAPI
from pymongo import MongoClient
from app.api.v1.auth_controller import router as auth_router
from app.core.config import settings

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

app.include_router(auth_router)

@app.get("/health", tags=["health"], summary="Health Check", description="Check if the API is running")
def health():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        dict: Status of the API
    """
    return {"status": "ok"}
