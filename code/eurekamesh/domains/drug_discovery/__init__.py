"""
Drug Discovery Domain Adapter for CCAD.

Implements molecular generation, canonicalization, and scoring
for small molecule drug discovery applications.
"""

from .canonicalizer import SMILESCanonicalizer, SMILESCanonicalizerBasic
from .scorer import DrugLikenessScorer, MinimalDrugLikenessScorer
from .constraints import (
    DrugDiscoveryConstraints,
    KINASE_INHIBITOR_CONSTRAINTS,
    GPCR_LIGAND_CONSTRAINTS,
    CNS_DRUG_CONSTRAINTS,
    get_constraint_template
)

__all__ = [
    # Canonicalizers
    'SMILESCanonicalizer',
    'SMILESCanonicalizerBasic',
    # Scorers
    'DrugLikenessScorer',
    'MinimalDrugLikenessScorer',
    # Constraints
    'DrugDiscoveryConstraints',
    'KINASE_INHIBITOR_CONSTRAINTS',
    'GPCR_LIGAND_CONSTRAINTS',
    'CNS_DRUG_CONSTRAINTS',
    'get_constraint_template'
]

