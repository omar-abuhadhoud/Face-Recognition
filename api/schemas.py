from pydantic import BaseModel


class CompareResponse(BaseModel):
    """Result of comparing an uploaded face to the stored faces."""
    person: str | None   # None if no stored person is similar enough
    similarity: float    # cosine similarity to the closest stored person
