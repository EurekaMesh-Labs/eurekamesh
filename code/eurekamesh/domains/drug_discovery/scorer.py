"""
Drug-likeness Scorer for Drug Discovery.

Implements Scorer interface for molecular evaluation.
Evaluates molecular candidates on multiple criteria:
    - Synthetic accessibility (SA-Score) - uses RDKit's sascorer
    - Lipinski Rule of Five compliance
    - Molecular properties (MW, LogP, HBD/HBA, TPSA)
    - Basic validity checks
"""

from typing import Dict, Tuple, Optional, Set
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
try:
    from rdkit.Chem import FilterCatalog, FilterCatalogParams
    HAS_FILTER_CATALOG = True
except Exception:
    HAS_FILTER_CATALOG = False
try:
    from rdkit.Chem import RDConfig
    import os
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    HAS_SASCORER = True
except ImportError:
    HAS_SASCORER = False

from eurekamesh.core.interfaces import Scorer


class DrugLikenessScorer(Scorer):
    """
    Scores molecules based on drug-likeness criteria.
    
    Evaluates:
        - SA-Score (synthetic accessibility): Lower is better, <3.5 is good
        - Lipinski Rule of Five compliance: Binary pass/fail
        - Molecular properties: MW, LogP, HBD, HBA, TPSA
        - Atom validity: Common drug-like atoms
    
    Examples:
        >>> scorer = DrugLikenessScorer()
        >>> scores = scorer.score("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # Ibuprofen
        {
            'sa_score': 2.1,
            'lipinski_pass': 1.0,
            'mw': 206.28,
            'logp': 3.5,
            'hbd': 1,
            'hba': 2,
            'tpsa': 37.3,
            'composite_score': 0.85
        }
        
        >>> passes, reason = scorer.passes_filters("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        (True, None)
    """
    
    def __init__(
        self,
        allowed_atoms: Optional[Set[str]] = None,
        sa_score_max: float = 4.0,
        use_composite_score: bool = True,
        enable_pains: bool = True,
        pains_max_hits: int = 0
    ):
        """
        Initialize drug-likeness scorer.
        
        Args:
            allowed_atoms: Set of allowed atom symbols
            sa_score_max: Maximum SA-Score threshold
            use_composite_score: Calculate weighted composite score
        """
        self.allowed_atoms = allowed_atoms or {
            'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'H'
        }
        self.sa_score_max = sa_score_max
        self.use_composite_score = use_composite_score
        self.enable_pains = enable_pains and HAS_FILTER_CATALOG
        self.pains_max_hits = pains_max_hits
        self._pains_catalog = None
        if self.enable_pains:
            try:
                params = FilterCatalogParams()
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
                self._pains_catalog = FilterCatalog.FilterCatalogParamsToFilterCatalog(params)
            except Exception:
                self._pains_catalog = None
                self.enable_pains = False
    
    def passes_filters(self, smiles: str) -> Tuple[bool, Optional[str]]:
        """
        Quick pass/fail check without full scoring.
        
        Args:
            smiles: SMILES string to check
        
        Returns:
            Tuple of (passes, failure_reason)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return (False, "Invalid SMILES")
            
            # Check Lipinski
            if not self._check_lipinski(mol):
                return (False, "Lipinski Rule of Five violation")
            
            # Check atoms
            atoms = set(atom.GetSymbol() for atom in mol.GetAtoms())
            invalid_atoms = atoms - self.allowed_atoms
            if invalid_atoms:
                return (False, f"Invalid atoms: {invalid_atoms}")
            
            # Check SA-Score if available
            if HAS_SASCORER:
                sa_score = sascorer.calculateScore(mol)
                if sa_score > self.sa_score_max:
                    return (False, f"SA-Score too high: {sa_score:.1f} > {self.sa_score_max}")
            
            # PAINS filters (optional)
            if self.enable_pains and self._pains_catalog is not None:
                try:
                    entries = self._pains_catalog.GetMatches(mol)
                    pains_hits = len(entries) if entries is not None else 0
                    if pains_hits > self.pains_max_hits:
                        return (False, f"PAINS hits: {pains_hits} > {self.pains_max_hits}")
                except Exception:
                    # If PAINS fails, do not block
                    pass
            
            return (True, None)
        except Exception as e:
            return (False, str(e))
    
    def score(self, smiles: str) -> Dict[str, float]:
        """
        Compute comprehensive drug-likeness scores.
        
        Args:
            smiles: SMILES string to score
        
        Returns:
            Dictionary with multiple scores:
                - sa_score: Synthetic accessibility [1-10], lower is better
                - lipinski_pass: Binary 0/1
                - mw: Molecular weight (Da)
                - logp: Partition coefficient
                - hbd: H-bond donors
                - hba: H-bond acceptors
                - tpsa: Topological polar surface area (Ų)
                - composite_score: Weighted combination [0-1], higher is better
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {'valid': 0.0}
            
            # Calculate SA-Score if available
            sa_score = 5.0  # Default fallback
            if HAS_SASCORER:
                sa_score = sascorer.calculateScore(mol)
            
            # Base scores
            scores = {
                'valid': 1.0,
                'sa_score': sa_score,
                'lipinski_pass': 1.0 if self._check_lipinski(mol) else 0.0,
                'mw': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hbd': float(Lipinski.NumHDonors(mol)),
                'hba': float(Lipinski.NumHAcceptors(mol)),
                'tpsa': rdMolDescriptors.CalcTPSA(mol)
            }
            # Add PAINS info if available
            if self.enable_pains and self._pains_catalog is not None:
                try:
                    entries = self._pains_catalog.GetMatches(mol)
                    pains_hits = len(entries) if entries is not None else 0
                    scores['pains_hits'] = float(pains_hits)
                    scores['pains_pass'] = 1.0 if pains_hits <= self.pains_max_hits else 0.0
                except Exception:
                    scores['pains_hits'] = 0.0
                    scores['pains_pass'] = 1.0
            
            # Composite score (weighted combination)
            if self.use_composite_score:
                scores['composite_score'] = self._calculate_composite_score(scores)
            
            return scores
        except Exception as e:
            return {'valid': 0.0, 'error': str(e)}
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted composite drug-likeness score.
        
        Combines multiple factors:
            - SA-Score: 30% weight (normalized, inverted)
            - Lipinski: 30% weight (binary)
            - MW in range: 20% weight
            - LogP in range: 20% weight
        
        Args:
            scores: Scores dictionary
        
        Returns:
            Composite score [0, 1] where 1 is best
        """
        if scores.get('valid', 0) == 0:
            return 0.0
        
        composite = 0.0
        
        # 1. SA-Score contribution (30%)
        # Normalize SA-Score: 1-10 scale, invert so lower is better
        sa_score = scores.get('sa_score', 5.0)
        sa_normalized = max(0.0, min(1.0, (6.0 - sa_score) / 5.0))
        composite += 0.30 * sa_normalized
        
        # 2. Lipinski compliance (30%)
        composite += 0.30 * scores.get('lipinski_pass', 0.0)
        
        # 3. MW in optimal range (20%)
        # Optimal: 200-500 Da
        mw = scores.get('mw', 0)
        if 200 <= mw <= 500:
            composite += 0.20
        elif 150 <= mw < 200:
            composite += 0.20 * ((mw - 150) / 50)
        elif 500 < mw <= 600:
            composite += 0.20 * ((600 - mw) / 100)
        
        # 4. LogP in optimal range (20%)
        # Optimal: 1-3
        logp = scores.get('logp', 0)
        if 1 <= logp <= 3:
            composite += 0.20
        elif 0 <= logp < 1:
            composite += 0.20 * logp
        elif 3 < logp <= 5:
            composite += 0.20 * ((5 - logp) / 2)
        
        return min(1.0, composite)
    
    def _check_lipinski(self, mol) -> bool:
        """Quick Lipinski Rule of Five check."""
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        return (
            mw <= 500 and
            logp <= 5 and
            hbd <= 5 and
            hba <= 10
        )
    
    def batch_score(self, smiles_list: list) -> list:
        """
        Score multiple molecules (batch processing).
        
        Args:
            smiles_list: List of SMILES strings
        
        Returns:
            List of score dictionaries
        """
        return [self.score(smiles) for smiles in smiles_list]


class MinimalDrugLikenessScorer(Scorer):
    """
    Lightweight scorer without full industrial pipeline (faster).
    
    Use when you only need basic property calculation without
    comprehensive filtering.
    """
    
    def passes_filters(self, smiles: str) -> Tuple[bool, Optional[str]]:
        """Basic validity check only."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return (False, "Invalid SMILES")
            return (True, None)
        except Exception as e:
            return (False, str(e))
    
    def score(self, smiles: str) -> Dict[str, float]:
        """Calculate basic molecular properties."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {'valid': 0.0}
            
            return {
                'valid': 1.0,
                'mw': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hbd': Lipinski.NumHDonors(mol),
                'hba': Lipinski.NumHAcceptors(mol),
                'tpsa': rdMolDescriptors.CalcTPSA(mol),
                'lipinski_pass': 1.0 if self._check_lipinski(mol) else 0.0
            }
        except Exception:
            return {'valid': 0.0}
    
    def _check_lipinski(self, mol) -> bool:
        """Quick Lipinski Rule of Five check."""
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        return (
            mw <= 500 and
            logp <= 5 and
            hbd <= 5 and
            hba <= 10
        )

