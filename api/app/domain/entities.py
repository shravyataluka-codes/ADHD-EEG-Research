"""Minimal domain records used by the Stage 4 persistence boundary."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class User:
    id: str
    email: str
    created_at: datetime
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Recording:
    id: str
    owner_id: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Analysis:
    id: str
    owner_id: str
    recording_id: str
    created_at: datetime
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class Explanation:
    id: str
    owner_id: str
    analysis_id: str
    created_at: datetime
    content: dict[str, Any] = field(default_factory=dict)
