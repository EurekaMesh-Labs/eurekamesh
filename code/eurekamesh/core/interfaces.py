"""
Core interfaces for CCAD Framework.

These abstract base classes define the contract that domain-specific 
adapters must implement to work with the CCAD engine.

The framework is designed to be domain-agnostic: by implementing these
interfaces, you can use CCAD for any combinatorial space exploration
(molecules, polymers, protein sequences, experimental conditions, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Tuple


class Canonicalizer(ABC):
    """
    Converts objects to a canonical (standardized, unique) representation.
    
    This is critical for accurate duplicate detection: the same object 
    represented in different ways (e.g., different SMILES notations for 
    the same molecule) must produce the same canonical form.
    
    Examples:
        - Molecules: SMILES canonicalization via RDKit
        - Protein sequences: Uppercase normalization, gap removal
        - Polymers: Ordered, canonical SMILES of monomers
        - Experimental conditions: Normalized tuple (temp, pressure, etc.)
    """
    
    @abstractmethod
    def canonicalize(self, obj: Any) -> str:
        """
        Convert object to canonical string representation.
        
        Args:
            obj: The object to canonicalize (type depends on domain)
        
        Returns:
            Canonical string representation (unique for equivalent objects)
        
        Examples:
            >>> canonicalizer.canonicalize("CC1=CC=C(C=C1)C(=O)O")
            "Cc1ccc(C(=O)O)cc1"
            
            >>> canonicalizer.canonicalize("O=C(O)c1ccc(C)cc1")
            "Cc1ccc(C(=O)O)cc1"  # Same as above!
        """
        pass
    
    @abstractmethod
    def is_valid(self, obj: Any) -> bool:
        """
        Check if object is syntactically/structurally valid.
        
        Args:
            obj: The object to validate
        
        Returns:
            True if valid, False otherwise
        
        Examples:
            >>> canonicalizer.is_valid("CCO")  # Valid SMILES
            True
            
            >>> canonicalizer.is_valid("C1CC")  # Invalid (unclosed ring)
            False
        """
        pass
    
    def preprocess(self, obj: Any) -> Any:
        """
        Optional preprocessing before canonicalization.
        
        Override this if you need to normalize input before canonicalization
        (e.g., desalting molecules, removing whitespace, etc.).
        
        Args:
            obj: The object to preprocess
        
        Returns:
            Preprocessed object (same type as input)
        """
        return obj


class DistanceMetric(ABC):
    """
    Calculates distance/similarity between objects.
    
    Used for:
        - Fuzzy deduplication (reject "too similar" objects)
        - Diversity-driven selection (maximize distances)
        - Clustering and coverage analysis
    
    Examples:
        - Molecules: Tanimoto similarity on ECFP4 fingerprints
        - Sequences: Edit distance, sequence identity
        - Vectors: Cosine similarity, Euclidean distance
    """
    
    @abstractmethod
    def distance(self, obj1: Any, obj2: Any) -> float:
        """
        Compute distance between two objects.
        
        Args:
            obj1: First object
            obj2: Second object
        
        Returns:
            Distance in [0, 1] where:
                - 0 = identical/maximally similar
                - 1 = maximally different
        
        Note:
            If your metric returns similarity (high = similar), 
            convert to distance: distance = 1 - similarity
        
        Examples:
            >>> metric.distance("CCO", "CCCO")
            0.15  # Similar molecules
            
            >>> metric.distance("CCO", "c1ccccc1")
            0.85  # Very different molecules
        """
        pass
    
    def batch_distance(self, obj: Any, obj_list: List[Any]) -> List[float]:
        """
        Compute distances from obj to multiple objects (optional optimization).
        
        Override this if you can compute batch distances more efficiently
        than calling distance() repeatedly.
        
        Args:
            obj: Query object
            obj_list: List of objects to compare against
        
        Returns:
            List of distances (same length as obj_list)
        """
        return [self.distance(obj, other) for other in obj_list]


class Scorer(ABC):
    """
    Evaluates quality/value of generated objects.
    
    Scores are used for:
        - Filtering (reject low-scoring candidates)
        - Ranking (prioritize high-scoring candidates)
        - Multi-objective optimization
        - Adaptive exploration (focus on promising regions)
    
    Examples:
        - Molecules: SA-Score, Lipinski compliance, predicted activity
        - Polymers: Thermal conductivity, flexibility, cost
        - Proteins: Stability, activity, solubility
    """
    
    @abstractmethod
    def score(self, obj: Any) -> Dict[str, float]:
        """
        Compute multiple scores for an object.
        
        Args:
            obj: The object to score
        
        Returns:
            Dictionary mapping score names to values
            
            Conventions:
                - Higher scores = better (normalize to [0, 1] if possible)
                - Include both properties and composite scores
                - Use consistent naming across domains where applicable
        
        Examples:
            >>> scorer.score("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
            {
                "sa_score": 2.1,           # Synthetic accessibility
                "lipinski_pass": 1.0,      # Binary (0 or 1)
                "novelty": 0.85,           # vs reference database
                "predicted_activity": 0.72, # Model prediction
                "composite_score": 0.79    # Weighted combination
            }
        """
        pass
    
    def passes_filters(self, obj: Any) -> Tuple[bool, Optional[str]]:
        """
        Quick pass/fail check (optional, for early filtering).
        
        Override this if you have hard constraints that can be checked
        before computing full scores (e.g., molecular weight limits).
        
        Args:
            obj: The object to check
        
        Returns:
            Tuple of (passes: bool, reason: Optional[str])
            If passes=False, reason should explain why
        
        Examples:
            >>> scorer.passes_filters("C" * 100)  # Too large
            (False, "Molecular weight > 500 Da")
            
            >>> scorer.passes_filters("CCO")
            (True, None)
        """
        return (True, None)


class RetrievalSystem(ABC):
    """
    RAG (Retrieval-Augmented Generation) system for context enrichment.
    
    Optional but powerful: enables:
        - Constraint retrieval from knowledge bases
        - Example-based prompting
        - Fuzzy deduplication via vector search
        - Compressed anti-duplication context (prototypes vs full list)
    
    Examples:
        - Vector DB (Qdrant, Pinecone) for semantic search
        - Document store (papers, SOPs, guidelines)
        - Internal databases (ELN, LIMS)
    """
    
    @abstractmethod
    def retrieve_constraints(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve relevant constraints/rules from knowledge base.
        
        Used to enrich prompts with domain knowledge before generation.
        
        Args:
            query: Natural language query (e.g., "EGFR kinase inhibitors")
            k: Number of constraints to retrieve
        
        Returns:
            List of constraint strings
        
        Examples:
            >>> rag.retrieve_constraints("EGFR kinase", k=3)
            [
                "Maintain MW < 500 Da for oral bioavailability",
                "Include H-bond acceptor for ATP pocket binding",
                "Avoid reactive Michael acceptors (toxicity risk)"
            ]
        """
        pass
    
    @abstractmethod
    def retrieve_examples(
        self, 
        query: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Retrieve similar examples from memory/corpus.
        
        Used for:
            - Example-based prompting (few-shot learning)
            - Fuzzy deduplication (find near-duplicates)
            - Prototype selection for anti-dup context
        
        Args:
            query: Query object or description
            k: Number of examples to retrieve
            filters: Optional metadata filters (e.g., {"domain": "kinase"})
        
        Returns:
            List of example objects (type depends on domain)
        
        Examples:
            >>> rag.retrieve_examples("Cc1ccc(C(=O)O)cc1", k=3)
            [
                "Cc1ccc(C(=O)N)cc1",      # Similar structure
                "Cc1ccc(C(=O)OC)cc1",     # Similar structure
                "c1ccc(C(=O)O)cc1"        # Related scaffold
            ]
        """
        pass
    
    def retrieve_prototypes(
        self, 
        n_clusters: int = 10, 
        items_per_cluster: int = 2
    ) -> List[Any]:
        """
        Retrieve representative prototypes from current memory.
        
        Used to compress anti-duplication context: instead of listing
        all N seen items, list K << N prototypes that span the space.
        
        Args:
            n_clusters: Number of clusters/regions to represent
            items_per_cluster: Examples per cluster
        
        Returns:
            List of prototype objects
        
        Default implementation: return empty list (not implemented).
        Override for compressed context benefits.
        """
        return []
    
    def add_to_memory(
        self, 
        obj: Any, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add object to retrieval memory (optional).
        
        Override if your RAG system supports incremental updates
        during generation (e.g., adding each accepted candidate to
        vector DB for future fuzzy dedup).
        
        Args:
            obj: Object to add
            metadata: Optional metadata (scores, timestamps, etc.)
        """
        pass


class ExperimentalDesignSystem(ABC):
    """
    Optional: Adaptive experimental design for active learning.
    
    Advanced feature for iterative optimization:
        - Suggests which regions to explore next
        - Balances exploration vs exploitation
        - Integrates feedback from wet-lab results
    
    Not needed for basic CCAD, but enables closed-loop optimization.
    """
    
    @abstractmethod
    def suggest_next_batch(
        self, 
        seen_objects: List[Any],
        scores: List[Dict[str, float]],
        n_suggestions: int = 10
    ) -> List[str]:
        """
        Suggest next batch of objects to generate.
        
        Args:
            seen_objects: All objects generated so far
            scores: Scores for each object
            n_suggestions: How many suggestions to make
        
        Returns:
            List of suggestions (prompts, constraints, or seed objects)
        
        Examples:
            >>> design.suggest_next_batch(molecules, scores, n_suggestions=5)
            [
                "Explore polar modifications of scaffold X",
                "Generate analogs with LogP < 3",
                "Focus on quinoline-based structures",
                ...
            ]
        """
        pass


# Type aliases for convenience
CanonicalForm = str
Distance = float
Scores = Dict[str, float]

