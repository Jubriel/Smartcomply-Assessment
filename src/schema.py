from typing import List
from pydantic import BaseModel, Field


class ClassificationOutput(BaseModel):
    snippet_id: str = Field(..., description="Unique identifier for the text snippet")
    adverse: bool = Field(..., description="True if the snippet indicates an adverse event, false otherwise")
    rationale: str = Field(..., description="Short explanation for the classification decision")


class RetrievalOutput(BaseModel):
    matched_name: str = Field(..., description="Name of the matched entity")
    score: float = Field(..., description="Similarity score between the query and the matched entity")
    rerank_score: float = Field(..., description="Reranking score from the cross-encoder model")


class FinalOutput(BaseModel):
    watchlist: list[RetrievalOutput] = Field(..., description="List of matched entities with scores")
    classification: List[ClassificationOutput] = Field(..., description="List of classification results for each snippet")
