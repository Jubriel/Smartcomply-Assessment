from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.core import Screening

from src.config import config
from src.schema import ClassificationOutput, FinalOutput, RetrievalOutput


app = FastAPI(
    title="Adverse Media Screening API",
    description="API for classifying news snippets and retrieving" \
    "relevant entities from a watchlist.",
    version="1.0.0"
)
screening = Screening()

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
    health_status = {"status": "ok", "service": "Screening API", "checks": {}}
    return JSONResponse(content=health_status, status_code=200)

@app.post("/screen", response_model=FinalOutput)
async def screen_snippets(query: str):
    """
    Endpoint to process a list of snippets and classify them.
    """
    try:
        result = await screening.process_snippets(query)
        return JSONResponse(content=result.dict(), status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
