"""
SMILES Canonicalizer for Drug Discovery.

Implements Canonicalizer interface using RDKit to ensure that
chemically equivalent molecules (regardless of SMILES notation)
are treated as duplicates.
"""

from typing import Any
from rdkit import Chem
from rdkit.Chem import MolStandardize

from eurekamesh.core.interfaces import Canonicalizer


class SMILESCanonicalizer(Canonicalizer):
    """
    Canonicalizes SMILES strings using RDKit.
    
    Features:
        - Converts SMILES to canonical form (unique representation)
        - Validates SMILES syntax
        - (Optional) Molecular standardization (desalting, neutralization)
    
    Examples:
        >>> canon = SMILESCanonicalizer()
        >>> canon.canonicalize("CC1=CC=C(C=C1)C(=O)O")
        "Cc1ccc(C(=O)O)cc1"
        
        >>> canon.canonicalize("O=C(O)c1ccc(C)cc1")
        "Cc1ccc(C(=O)O)cc1"  # Same as above!
        
        >>> canon.is_valid("CCO")
        True
        
        >>> canon.is_valid("C1CC")  # Unclosed ring
        False
    """
    
    def __init__(
        self,
        standardize: bool = False,  # Changed to False by default (compatibility)
        remove_stereo: bool = False,
        canonical_tautomer: bool = False
    ):
        """
        Initialize SMILES canonicalizer.
        
        Args:
            standardize: Apply MolStandardize (desalt, neutralize) - requires newer RDKit
            remove_stereo: Remove stereochemistry information
            canonical_tautomer: Use canonical tautomer (slower, more accurate)
        """
        self.standardize = standardize
        self.remove_stereo = remove_stereo
        self.canonical_tautomer = canonical_tautomer
        
        if standardize:
            try:
                self.standardizer = MolStandardize.Standardizer()
            except AttributeError:
                # Fallback for older RDKit versions
                self.standardize = False
        if canonical_tautomer:
            try:
                self.tautomer_enumerator = MolStandardize.TautomerEnumerator()
            except AttributeError:
                self.canonical_tautomer = False
    
    def preprocess(self, smiles: str) -> str:
        """
        Preprocess SMILES (strip whitespace).
        
        Args:
            smiles: Raw SMILES string
        
        Returns:
            Cleaned SMILES string
        """
        if not isinstance(smiles, str):
            return str(smiles)
        return smiles.strip()
    
    def is_valid(self, smiles: str) -> bool:
        s = self.preprocess(smiles)
        if not s:
            return False
        try:
            mol = Chem.MolFromSmiles(s)
            return (mol is not None) and (mol.GetNumAtoms() > 0)
        except Exception:
            return False
    
    def canonicalize(self, smiles: str) -> str:
        """
        Convert SMILES to canonical form.
        
        Args:
            smiles: Input SMILES string
        
        Returns:
            Canonical SMILES string
        
        Raises:
            ValueError: If SMILES is invalid
        """
        smiles = self.preprocess(smiles)
        
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Standardize molecule
            if self.standardize:
                try:
                    mol = self.standardizer.standardize(mol)
                except Exception as e:
                    # Fallback if standardization fails
                    pass
            
            # Remove stereochemistry if requested
            if self.remove_stereo:
                Chem.RemoveStereochemistry(mol)
            
            # Use canonical tautomer if requested
            if self.canonical_tautomer:
                try:
                    mol = self.tautomer_enumerator.Canonicalize(mol)
                except Exception:
                    # Fallback if tautomer enumeration fails
                    pass
            
            # Generate canonical SMILES
            canonical = Chem.MolToSmiles(mol, canonical=True)
            return canonical
        
        except Exception as e:
            raise ValueError(f"Failed to canonicalize SMILES '{smiles}': {e}")
    
    def get_mol(self, smiles: str) -> Any:
        """
        Get RDKit Mol object (utility method).
        
        Args:
            smiles: SMILES string
        
        Returns:
            RDKit Mol object or None if invalid
        """
        smiles = self.preprocess(smiles)
        return Chem.MolFromSmiles(smiles)


class SMILESCanonicalizerBasic(Canonicalizer):
    """
    Basic SMILES canonicalizer without standardization (faster).
    
    Use this if you don't need desalting/neutralization/tautomer handling.
    """
    
    def preprocess(self, smiles: str) -> str:
        if not isinstance(smiles, str):
            return str(smiles)
        return smiles.strip()
    
    def is_valid(self, smiles: str) -> bool:
        try:
            smiles = self.preprocess(smiles)
            if not smiles:
                return False
            mol = Chem.MolFromSmiles(smiles)
            return (mol is not None) and (mol.GetNumAtoms() > 0)
        except Exception:
            return False
    
    def canonicalize(self, smiles: str) -> str:
        smiles = self.preprocess(smiles)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            raise ValueError(f"Failed to canonicalize SMILES '{smiles}': {e}")

