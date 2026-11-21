"""Embedded SQLite tiered memory store and deterministic fact extractor for EdgeMem."""

from edgemem.memory.consolidator import (
    ConsolidationReport,
    ConsolidationWorker,
    MemoryConsolidator,
)
from edgemem.memory.extractor import ExtractedFact, extract_facts_from_messages
from edgemem.memory.store import Episode, Fact, MemoryStore, compute_effective_salience

__all__ = [
    "MemoryStore",
    "Fact",
    "Episode",
    "ExtractedFact",
    "extract_facts_from_messages",
    "ConsolidationReport",
    "ConsolidationWorker",
    "MemoryConsolidator",
    "compute_effective_salience",
]
