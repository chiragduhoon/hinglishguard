from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator
import hashlib

Action = Literal['NO_ISSUE_DETECTED', 'FLAG', 'HUMAN_REVIEW']

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Comment(Strict):
    id: str = ''
    text: str = Field(min_length=1, max_length=2000)
    parent_id: str | None = None
    parent_text: str | None = Field(default=None, max_length=2000)

    @field_validator('text')
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError('Comment cannot be blank')
        return value

    def identity(self):
        return self.id or hashlib.sha256((self.text+'\0'+(self.parent_id or '')+'\0'+(self.parent_text or '')).encode()).hexdigest()[:24]

class Finding(Strict):
    task: str
    label: str
    scores: dict[str, float]
    positive_score: float
    branch: Literal['positive','negative','review']
    version: str
    truncated: bool = False

class Evidence(Strict):
    source: Literal['original','context','tool']
    quote: str = Field(min_length=1, max_length=500)

class Verdict(Strict):
    recommendation: Action
    contextual_ambiguity: bool
    reason_codes: list[str] = Field(min_length=1, max_length=8)
    evidence: list[Evidence] = Field(min_length=1, max_length=8)
    brief_explanation: str = Field(min_length=1, max_length=700)

class ClassifyArgs(Strict):
    variant: Literal['original','normalized'] = 'original'

class EmptyArgs(Strict):
    pass
