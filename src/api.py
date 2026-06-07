from fastapi import FastAPI, Request
from contextlib import asynccontextmanager

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.core import Screening

from src.config import config
from src.schema import FinalOutput


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.screening = await Screening.create()
    yield

app = FastAPI(
    title="Adverse Media Screening API",
    description="API for classifying news snippets and retrieving" \
    "relevant entities from a watchlist.",
    version="1.0.0",
    lifespan=lifespan
)
screening = None


app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True
    )


@app.get("/health")
async def health_check():
    """Health check endpoint with dependency checks."""
    health_status = {"status": "ok", "service": "Screening API",
                     "checks": {}}
    return JSONResponse(content=health_status, status_code=200)


@app.post("/screen", response_model=FinalOutput)
async def screen_snippets(query: str, request: Request):
    """
    Endpoint to process a list of snippets and classify them.
    """
    try:
        screening = request.app.state.screening
        return await screening.process_snippets(query)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
