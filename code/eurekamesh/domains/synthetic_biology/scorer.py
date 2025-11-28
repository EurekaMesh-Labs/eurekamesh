"""
Protein Stability Scorer for Synthetic Biology.

Implements scoring for protein sequences based on bioinformatics metrics.
Uses simple heuristics and public predictors where possible.
"""

from typing import Dict, Tuple, Optional
from collections import Counter

from eurekamesh.core.interfaces import Scorer


class ProteinStabilityScorer(Scorer):
    """
    Scores protein sequences based on stability and drug-likeness metrics.
    
    Metrics (heuristic-based for MVP):
        - Sequence composition (hydrophobic/hydrophilic balance)
        - Length appropriateness
        - Charge distribution
        - Secondary structure propensity
        - Disorder prediction (simple)
    
    Examples:
        >>> scorer = ProteinStabilityScorer()
        >>> scores = scorer.score("MKTIIALSYIFCLVFA")
        {
            'hydrophobic_fraction': 0.625,
            'charge_balance': 0.0,
            'length_score': 1.0,
            'composition_score': 0.75,
            'composite_score': 0.78
        }
        
        >>> passes, reason = scorer.passes_filters("MKTIIALSYIFCLVFA")
        (True, None)
    """
    
    # Amino acid properties
    HYDROPHOBIC = set('AILMFWV')
    HYDROPHILIC = set('NQST')
    CHARGED_POS = set('KRH')
    CHARGED_NEG = set('DE')
    AROMATIC = set('FYW')
    SMALL = set('AGST')
    
    # Secondary structure propensities (simplified)
    HELIX_FORMERS = set('AELMK')
    SHEET_FORMERS = set('VIY')
    DISORDER_PRONE = set('PQSGETK')
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 1000,
        target_hydrophobic: float = 0.4,  # ~40% hydrophobic
        target_charge_balance: float = 0.1  # Low net charge
    ):
        """
        Initialize protein stability scorer.
        
        Args:
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            target_hydrophobic: Target hydrophobic fraction
            target_charge_balance: Target charge balance (|pos - neg|/length)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.target_hydrophobic = target_hydrophobic
        self.target_charge_balance = target_charge_balance
    
    def passes_filters(self, sequence: str) -> Tuple[bool, Optional[str]]:
        """
        Quick pass/fail check.
        
        Args:
            sequence: Protein sequence
        
        Returns:
            Tuple of (passes, failure_reason)
        """
        sequence = sequence.upper().replace(' ', '').replace('-', '')
        
        # Length check
        if len(sequence) < self.min_length:
            return (False, f"Too short: {len(sequence)} < {self.min_length}")
        if len(sequence) > self.max_length:
            return (False, f"Too long: {len(sequence)} > {self.max_length}")
        
        # Basic composition check (no homopolymers)
        counts = Counter(sequence)
        max_single = max(counts.values())
        if max_single / len(sequence) > 0.5:
            most_common = max(counts, key=counts.get)
            return (False, f"Homopolymer detected: {most_common} = {max_single/len(sequence):.1%}")
        
        return (True, None)
    
    def score(self, sequence: str) -> Dict[str, float]:
        """
        Compute protein stability and quality scores.
        
        Args:
            sequence: Protein sequence
        
        Returns:
            Dictionary with scores:
                - hydrophobic_fraction: Fraction of hydrophobic residues
                - charge_balance: |positive - negative| / length
                - length_score: Appropriateness of length
                - composition_score: Diversity of amino acids
                - helix_propensity: Helix-forming tendency
                - disorder_score: Predicted disorder (0=ordered, 1=disordered)
                - composite_score: Weighted combination
        """
        sequence = sequence.upper().replace(' ', '').replace('-', '')
        
        if not sequence:
            return {'valid': 0.0}
        
        length = len(sequence)
        counts = Counter(sequence)
        
        # 1. Hydrophobic fraction
        hydrophobic_count = sum(counts.get(aa, 0) for aa in self.HYDROPHOBIC)
        hydrophobic_frac = hydrophobic_count / length
        
        # 2. Charge balance
        pos_count = sum(counts.get(aa, 0) for aa in self.CHARGED_POS)
        neg_count = sum(counts.get(aa, 0) for aa in self.CHARGED_NEG)
        charge_balance = abs(pos_count - neg_count) / length
        
        # 3. Length score (sigmoid around optimal range)
        optimal_min, optimal_max = 50, 500
        if optimal_min <= length <= optimal_max:
            length_score = 1.0
        elif length < optimal_min:
            length_score = length / optimal_min
        else:  # length > optimal_max
            length_score = max(0.0, 1.0 - (length - optimal_max) / optimal_max)
        
        # 4. Composition diversity (entropy-based)
        total = sum(counts.values())
        entropy = -sum((c/total) * (c/total) for c in counts.values() if c > 0)
        max_entropy = 2.996  # log2(20) for 20 amino acids
        composition_score = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # 5. Helix propensity
        helix_count = sum(counts.get(aa, 0) for aa in self.HELIX_FORMERS)
        helix_propensity = helix_count / length
        
        # 6. Sheet propensity
        sheet_count = sum(counts.get(aa, 0) for aa in self.SHEET_FORMERS)
        sheet_propensity = sheet_count / length
        
        # 7. Disorder score (higher = more disordered)
        disorder_count = sum(counts.get(aa, 0) for aa in self.DISORDER_PRONE)
        disorder_score = disorder_count / length
        
        # 8. Aromatic content
        aromatic_count = sum(counts.get(aa, 0) for aa in self.AROMATIC)
        aromatic_frac = aromatic_count / length
        
        scores = {
            'valid': 1.0,
            'length': length,
            'hydrophobic_fraction': hydrophobic_frac,
            'charge_balance': charge_balance,
            'length_score': length_score,
            'composition_score': composition_score,
            'helix_propensity': helix_propensity,
            'sheet_propensity': sheet_propensity,
            'disorder_score': disorder_score,
            'aromatic_fraction': aromatic_frac
        }
        
        # Composite score
        scores['composite_score'] = self._calculate_composite_score(scores)
        
        return scores
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted composite stability score.
        
        Factors:
            - Length appropriateness: 20%
            - Composition diversity: 25%
            - Hydrophobic balance: 20%
            - Charge balance: 15%
            - Structural propensity: 10%
            - Low disorder: 10%
        
        Args:
            scores: Scores dictionary
        
        Returns:
            Composite score [0, 1]
        """
        if scores.get('valid', 0) == 0:
            return 0.0
        
        composite = 0.0
        
        # 1. Length appropriateness (20%)
        composite += 0.20 * scores.get('length_score', 0.0)
        
        # 2. Composition diversity (25%)
        composite += 0.25 * scores.get('composition_score', 0.0)
        
        # 3. Hydrophobic balance (20%)
        # Optimal around 40%
        hydro = scores.get('hydrophobic_fraction', 0.0)
        hydro_score = 1.0 - abs(hydro - self.target_hydrophobic) / self.target_hydrophobic
        hydro_score = max(0.0, hydro_score)
        composite += 0.20 * hydro_score
        
        # 4. Charge balance (15%)
        # Lower is better (neutral)
        charge = scores.get('charge_balance', 0.0)
        charge_score = max(0.0, 1.0 - charge / 0.3)  # Penalize >30% net charge
        composite += 0.15 * charge_score
        
        # 5. Structural propensity (10%)
        # Prefer some structure (helix or sheet)
        helix = scores.get('helix_propensity', 0.0)
        sheet = scores.get('sheet_propensity', 0.0)
        structure_score = min(1.0, helix + sheet)
        composite += 0.10 * structure_score
        
        # 6. Low disorder (10%)
        # Lower disorder is better for stability
        disorder = scores.get('disorder_score', 0.0)
        order_score = 1.0 - min(1.0, disorder)
        composite += 0.10 * order_score
        
        return min(1.0, composite)
    
    def calculate_sequence_identity(self, seq1: str, seq2: str) -> float:
        """
        Calculate sequence identity between two sequences.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
        
        Returns:
            Sequence identity [0, 1]
        """
        seq1 = seq1.upper().replace(' ', '').replace('-', '')
        seq2 = seq2.upper().replace(' ', '').replace('-', '')
        
        if not seq1 or not seq2:
            return 0.0
        
        # Align to shorter sequence
        min_len = min(len(seq1), len(seq2))
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        
        # Normalize by longer sequence (more conservative)
        max_len = max(len(seq1), len(seq2))
        return matches / max_len if max_len > 0 else 0.0

