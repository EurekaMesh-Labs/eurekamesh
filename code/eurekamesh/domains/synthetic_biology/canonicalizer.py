"""
Sequence Canonicalizers for Synthetic Biology.

Implements canonicalization for protein and DNA sequences.
"""

from typing import Any
from eurekamesh.core.interfaces import Canonicalizer


class ProteinSequenceCanonicalizer(Canonicalizer):
    """
    Canonicalizes protein amino acid sequences.
    
    Canonicalization strategy:
        1. Convert to uppercase
        2. Remove gaps/spaces
        3. Validate amino acid codes
        4. Return normalized sequence
    
    Examples:
        >>> canon = ProteinSequenceCanonicalizer()
        >>> canon.canonicalize("mkt iialsyifclvfa")
        "MKTIIALSYIFCLVFA"
        
        >>> canon.is_valid("MKTIIALSYIFCLVFA")
        True
        
        >>> canon.is_valid("MKTXYZ")  # Invalid amino acids
        False
    """
    
    # Standard 20 amino acids
    VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Extended amino acids (include ambiguous/modified)
    EXTENDED_AA = VALID_AA | set('XBZJUO*')
    
    def __init__(
        self,
        allow_extended: bool = False,
        remove_terminal_stop: bool = True
    ):
        """
        Initialize protein sequence canonicalizer.
        
        Args:
            allow_extended: Allow extended amino acid codes (X, B, Z, etc.)
            remove_terminal_stop: Remove terminal stop codon (*)
        """
        self.allow_extended = allow_extended
        self.remove_terminal_stop = remove_terminal_stop
        self.valid_codes = self.EXTENDED_AA if allow_extended else self.VALID_AA
    
    def preprocess(self, sequence: str) -> str:
        """
        Preprocess sequence (uppercase, remove whitespace/gaps).
        
        Args:
            sequence: Raw protein sequence
        
        Returns:
            Preprocessed sequence
        """
        if not isinstance(sequence, str):
            sequence = str(sequence)
        
        # Uppercase
        sequence = sequence.upper()
        
        # Remove common gap characters and whitespace
        for char in ['-', '_', ' ', '\n', '\t', '.']:
            sequence = sequence.replace(char, '')
        
        # Remove terminal stop if requested
        if self.remove_terminal_stop and sequence.endswith('*'):
            sequence = sequence[:-1]
        
        return sequence
    
    def is_valid(self, sequence: str) -> bool:
        """
        Check if sequence contains only valid amino acids.
        
        Args:
            sequence: Protein sequence to validate
        
        Returns:
            True if valid, False otherwise
        """
        sequence = self.preprocess(sequence)
        
        if not sequence:
            return False
        
        # Check all characters are valid amino acids
        return all(aa in self.valid_codes for aa in sequence)
    
    def canonicalize(self, sequence: str) -> str:
        """
        Convert sequence to canonical form.
        
        Args:
            sequence: Input protein sequence
        
        Returns:
            Canonical sequence
        
        Raises:
            ValueError: If sequence is invalid
        """
        sequence = self.preprocess(sequence)
        
        if not self.is_valid(sequence):
            invalid_chars = set(sequence) - self.valid_codes
            raise ValueError(
                f"Invalid amino acid codes in sequence: {invalid_chars}"
            )
        
        return sequence
    
    def get_length(self, sequence: str) -> int:
        """Get sequence length (utility method)."""
        return len(self.preprocess(sequence))


class DNASequenceCanonicalizer(Canonicalizer):
    """
    Canonicalizes DNA sequences.
    
    Canonicalization strategy:
        1. Convert to uppercase
        2. Remove gaps/spaces
        3. Validate nucleotide codes
        4. Optionally: reverse complement for standardization
    
    Examples:
        >>> canon = DNASequenceCanonicalizer()
        >>> canon.canonicalize("atgc gatc")
        "ATGCGATC"
        
        >>> canon.is_valid("ATGCGATC")
        True
    """
    
    # Standard nucleotides
    VALID_BASES = set('ATGC')
    
    # Extended nucleotide codes (ambiguous bases)
    EXTENDED_BASES = VALID_BASES | set('NRYKMSWBDHV')
    
    def __init__(
        self,
        allow_extended: bool = True,
        standardize_direction: bool = False
    ):
        """
        Initialize DNA sequence canonicalizer.
        
        Args:
            allow_extended: Allow ambiguous nucleotide codes
            standardize_direction: Use lexicographically smaller of forward/reverse
        """
        self.allow_extended = allow_extended
        self.standardize_direction = standardize_direction
        self.valid_codes = self.EXTENDED_BASES if allow_extended else self.VALID_BASES
    
    def preprocess(self, sequence: str) -> str:
        """Preprocess DNA sequence."""
        if not isinstance(sequence, str):
            sequence = str(sequence)
        
        # Uppercase
        sequence = sequence.upper()
        
        # Remove gaps and whitespace
        for char in ['-', '_', ' ', '\n', '\t', '.']:
            sequence = sequence.replace(char, '')
        
        return sequence
    
    def is_valid(self, sequence: str) -> bool:
        """Check if sequence contains only valid nucleotides."""
        sequence = self.preprocess(sequence)
        
        if not sequence:
            return False
        
        return all(base in self.valid_codes for base in sequence)
    
    def canonicalize(self, sequence: str) -> str:
        """
        Convert DNA sequence to canonical form.
        
        Args:
            sequence: Input DNA sequence
        
        Returns:
            Canonical sequence
        
        Raises:
            ValueError: If sequence is invalid
        """
        sequence = self.preprocess(sequence)
        
        if not self.is_valid(sequence):
            invalid_chars = set(sequence) - self.valid_codes
            raise ValueError(
                f"Invalid nucleotide codes in sequence: {invalid_chars}"
            )
        
        # Optionally standardize direction
        if self.standardize_direction:
            reverse_comp = self._reverse_complement(sequence)
            # Use lexicographically smaller
            sequence = min(sequence, reverse_comp)
        
        return sequence
    
    def _reverse_complement(self, sequence: str) -> str:
        """Calculate reverse complement."""
        complement = {
            'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
            'N': 'N', 'R': 'Y', 'Y': 'R', 'K': 'M',
            'M': 'K', 'S': 'S', 'W': 'W', 'B': 'V',
            'V': 'B', 'D': 'H', 'H': 'D'
        }
        return ''.join(complement.get(base, base) for base in reversed(sequence))
    
    def get_length(self, sequence: str) -> int:
        """Get sequence length."""
        return len(self.preprocess(sequence))

