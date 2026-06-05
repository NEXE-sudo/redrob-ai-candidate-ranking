"""Main FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api import jobs, candidates, rankings

# Models import for table creation
from app.models import database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    
    # Startup
    print("Starting Redrob AI Recruiter API...")
    
    yield
    
    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(jobs.router)
app.include_router(candidates.router)
app.include_router(rankings.router)


@app.get("/")
async def root():
    """Root endpoint."""
    
    return {
        "message": "Redrob AI Recruiter API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "endpoints": {
            "jobs": "/api/jobs",
            "candidates": "/api/candidates",
            "rankings": "/api/rankings",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    
    print(f"Error: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
