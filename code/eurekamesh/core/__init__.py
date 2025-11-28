"""
CCAD Core Framework - Domain-agnostic hypothesis exploration engine.

This package contains the core CCAD (Cumulative Context Anti-Duplication) 
framework components that are independent of any specific domain 
(drug discovery, materials science, synthetic biology, etc.).
"""

from .interfaces import (
    Canonicalizer,
    DistanceMetric,
    Scorer,
    RetrievalSystem,
    ExperimentalDesignSystem
)
from .ccad_engine import CCAdEngine
from .metrics import (
    MetricsTracker,
    calculate_upt,
    calculate_duplicate_rate,
    calculate_coverage,
    calculate_saturation_score,
    calculate_peak_efficiency,
    generate_summary_report,
    compare_baselines
)
from .rag_system import InMemoryRAGSystem, SimpleRAGSystem

__all__ = [
    # Interfaces
    'Canonicalizer',
    'DistanceMetric',
    'Scorer',
    'RetrievalSystem',
    'ExperimentalDesignSystem',
    # Engine
    'CCAdEngine',
    # Metrics
    'MetricsTracker',
    'calculate_upt',
    'calculate_duplicate_rate',
    'calculate_coverage',
    'calculate_saturation_score',
    'calculate_peak_efficiency',
    'generate_summary_report',
    'compare_baselines',
    # RAG
    'InMemoryRAGSystem',
    'SimpleRAGSystem'
]

__version__ = '2.0.0'  # Major refactor to framework architecture

