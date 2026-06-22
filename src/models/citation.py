from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Citation(BaseModel):
    source_id: str = ""
    filename: str = ""
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    text_snippet: str = ""
