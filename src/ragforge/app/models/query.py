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
