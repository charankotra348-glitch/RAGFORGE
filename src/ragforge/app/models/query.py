from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class QuerySource(BaseModel):
    citation: str
    page: str
    section: str
    text: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[QuerySource]
    abstained: bool = Field(
        default=False,
        description="True when the answer is a refusal rather than a claim.",
    )
    cached: bool = Field(
        default=False,
        description="True when this answer was served from the answer cache.",
    )
    abstention_reason: Literal["no_results", "model_abstained", "failed_verification"] | None = Field(
        default=None,
        description=(
            "Why the system abstained: retrieval found nothing, the model said the "
            "sources do not contain the answer, or the grounding check rejected the answer."
        ),
    )
