"""
moses.memory — Persistent Knowledge Accumulation for Moses v4.0

Modules:
    experience_store   : SQLite-backed experiment replay with vector similarity
    causal_reasoning   : Did this change cause improvement or coincidence?
    knowledge_graph    : Structured concepts and relationships (improves, compatible_with, ...)
    transfer_learning  : Cross-task principle extraction and recommendation

Quickstart:
    >>> from moses.memory import ExperienceStore, KnowledgeGraph, CausalEngine, TransferEngine
    >>> store = ExperienceStore("/data/moses/exp.db")
    >>> kg = KnowledgeGraph("/data/moses/kg.db")
    >>> causal = CausalEngine(store)
    >>> transfer = TransferEngine(store, kg)
"""

from moses.memory.experience_store import ExperienceStore, ExperimentRecord
from moses.memory.knowledge_graph import KnowledgeGraph, Node, Edge
from moses.memory.causal_reasoning import CausalEngine, CausalEstimate
from moses.memory.transfer_learning import TransferEngine, Principle, TransferRecommendation

__all__ = [
    "ExperienceStore",
    "ExperimentRecord",
    "KnowledgeGraph",
    "Node",
    "Edge",
    "CausalEngine",
    "CausalEstimate",
    "TransferEngine",
    "Principle",
    "TransferRecommendation",
]
