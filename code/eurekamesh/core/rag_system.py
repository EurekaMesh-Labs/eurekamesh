"""
RAG (Retrieval-Augmented Generation) System for CCAD.

Provides vector-based similarity search and memory management for:
    1. Fuzzy deduplication (detect near-duplicates)
    2. Compressed anti-dup context (prototypes vs full list)
    3. Constraint retrieval from knowledge bases
    4. Example-based prompting

Uses in-memory vector store for MVP (Qdrant/Pinecone for production).
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import hashlib

from .interfaces import RetrievalSystem

logger = logging.getLogger(__name__)

# Optional dependencies (fallback to in-memory if not available)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not available, using basic RAG")

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available, clustering disabled")


class InMemoryRAGSystem(RetrievalSystem):
    """
    In-memory RAG system using embeddings and cosine similarity.
    
    Features:
        - Vector embeddings for items
        - Cosine similarity search
        - Clustering for prototype selection
        - Incremental updates
    
    Usage:
        rag = InMemoryRAGSystem()
        
        # Add items as they're generated
        rag.add_to_memory("CCO", metadata={'score': 0.8})
        rag.add_to_memory("c1ccccc1", metadata={'score': 0.9})
        
        # Find similar items (fuzzy dedup)
        similar = rag.retrieve_examples("CC(C)O", k=5)
        
        # Get prototypes for compressed context
        prototypes = rag.retrieve_prototypes(n_clusters=10)
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        max_memory_size: int = 10000
    ):
        """
        Initialize RAG system.
        
        Args:
            embedding_model: Sentence transformer model name
            similarity_threshold: Threshold for similarity matching
            max_memory_size: Maximum items to keep in memory
        """
        self.similarity_threshold = similarity_threshold
        self.max_memory_size = max_memory_size
        
        # Initialize embedding model if available
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(embedding_model)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Loaded embedding model: {embedding_model} (dim={self.embedding_dim})")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}, using fallback")
                self.model = None
                self.embedding_dim = 384  # Default dimension
        else:
            self.model = None
            self.embedding_dim = 384
        
        # Memory storage
        self.items: List[Any] = []
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self.item_to_idx: Dict[str, int] = {}
    
    def add_to_memory(
        self,
        obj: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add object to RAG memory.
        
        Args:
            obj: Object to add (will be converted to string for embedding)
            metadata: Optional metadata (scores, timestamps, etc.)
        """
        # Convert to string for embedding
        obj_str = str(obj)
        
        # Skip if already in memory
        if obj_str in self.item_to_idx:
            return
        
        # Generate embedding
        if self.model:
            try:
                embedding = self.model.encode(obj_str, convert_to_numpy=True)
            except Exception as e:
                logger.debug(f"Embedding failed: {e}, using hash-based fallback")
                embedding = self._hash_based_embedding(obj_str)
        else:
            embedding = self._hash_based_embedding(obj_str)
        
        # Add to memory
        idx = len(self.items)
        self.items.append(obj)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})
        self.item_to_idx[obj_str] = idx
        
        # Trim memory if too large (FIFO)
        if len(self.items) > self.max_memory_size:
            self._trim_memory()
    
    def retrieve_examples(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Retrieve similar examples from memory.
        
        Args:
            query: Query object or string
            k: Number of examples to retrieve
            filters: Optional metadata filters
        
        Returns:
            List of similar objects
        """
        if not self.items:
            return []
        
        # Generate query embedding
        query_str = str(query)
        if self.model:
            try:
                query_embedding = self.model.encode(query_str, convert_to_numpy=True)
            except Exception:
                query_embedding = self._hash_based_embedding(query_str)
        else:
            query_embedding = self._hash_based_embedding(query_str)
        
        # Calculate similarities
        similarities = self._calculate_similarities(query_embedding)
        
        # Apply filters if provided
        if filters:
            valid_indices = [
                i for i, meta in enumerate(self.metadata)
                if all(meta.get(key) == value for key, value in filters.items())
            ]
            similarities = [(idx, sim) for idx, sim in similarities if idx in valid_indices]
        
        # Get top-k
        top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        
        return [self.items[idx] for idx, _ in top_k]
    
    def retrieve_prototypes(
        self,
        n_clusters: int = 10,
        items_per_cluster: int = 1
    ) -> List[Any]:
        """
        Retrieve representative prototypes via clustering.
        
        Args:
            n_clusters: Number of clusters to create
            items_per_cluster: How many items per cluster
        
        Returns:
            List of prototype objects
        """
        if not self.items or not HAS_SKLEARN:
            # Fallback: return evenly spaced items
            if not self.items:
                return []
            step = max(1, len(self.items) // n_clusters)
            return self.items[::step][:n_clusters]
        
        try:
            # Ensure we don't have more clusters than items
            n_clusters = min(n_clusters, len(self.items))
            
            # Cluster embeddings
            embeddings_array = np.array(self.embeddings)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            
            # Get items closest to cluster centers
            prototypes = []
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_indices) == 0:
                    continue
                
                # Find item closest to cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]
                cluster_embeddings = embeddings_array[cluster_indices]
                
                # Calculate distances to center
                distances = np.linalg.norm(
                    cluster_embeddings - cluster_center,
                    axis=1
                )
                
                # Get closest items
                closest_indices = cluster_indices[np.argsort(distances)[:items_per_cluster]]
                prototypes.extend([self.items[idx] for idx in closest_indices])
            
            return prototypes
        
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, using fallback")
            # Fallback: evenly spaced items
            step = max(1, len(self.items) // n_clusters)
            return self.items[::step][:n_clusters]
    
    def retrieve_prototypes_with_counts(
        self,
        n_clusters: int = 10
    ) -> List[Tuple[Any, int]]:
        """
        Retrieve prototypes with cluster size information.
        
        Args:
            n_clusters: Number of clusters to create
        
        Returns:
            List of (prototype, cluster_size) tuples
        """
        if not self.items or not HAS_SKLEARN:
            if not self.items:
                return []
            step = max(1, len(self.items) // n_clusters)
            prototypes = self.items[::step][:n_clusters]
            avg_size = len(self.items) // len(prototypes) if prototypes else 0
            return [(p, avg_size) for p in prototypes]
        
        try:
            n_clusters = min(n_clusters, len(self.items))
            
            embeddings_array = np.array(self.embeddings)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            
            prototypes_with_counts = []
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_indices) == 0:
                    continue
                
                cluster_center = kmeans.cluster_centers_[cluster_id]
                cluster_embeddings = embeddings_array[cluster_indices]
                
                distances = np.linalg.norm(
                    cluster_embeddings - cluster_center,
                    axis=1
                )
                
                closest_idx = cluster_indices[np.argmin(distances)]
                prototype = self.items[closest_idx]
                cluster_size = len(cluster_indices)
                
                prototypes_with_counts.append((prototype, cluster_size))
            
            return prototypes_with_counts
        
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            step = max(1, len(self.items) // n_clusters)
            prototypes = self.items[::step][:n_clusters]
            avg_size = len(self.items) // len(prototypes) if prototypes else 0
            return [(p, avg_size) for p in prototypes]
    
    def retrieve_constraints(
        self,
        query: str,
        k: int = 5
    ) -> List[str]:
        """
        Retrieve relevant constraints (placeholder for MVP).
        
        In production, this would query a knowledge base of rules/constraints.
        For now, returns empty list.
        
        Args:
            query: Query string
            k: Number of constraints to retrieve
        
        Returns:
            List of constraint strings
        """
        # Placeholder for constraint retrieval
        # In production: query vector DB of constraints/rules
        return []
    
    def get_similarity(self, obj1: Any, obj2: Any) -> float:
        """
        Calculate similarity between two objects.
        
        Args:
            obj1: First object
            obj2: Second object
        
        Returns:
            Similarity score [0, 1]
        """
        str1, str2 = str(obj1), str(obj2)
        
        if self.model:
            try:
                emb1 = self.model.encode(str1, convert_to_numpy=True)
                emb2 = self.model.encode(str2, convert_to_numpy=True)
                
                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                return float(similarity)
            except Exception:
                pass
        
        # Fallback: simple string comparison
        return 1.0 if str1 == str2 else 0.0
    
    def clear_memory(self) -> None:
        """Clear all memory."""
        self.items.clear()
        self.embeddings.clear()
        self.metadata.clear()
        self.item_to_idx.clear()
    
    def get_memory_size(self) -> int:
        """Get current memory size."""
        return len(self.items)
    
    def _calculate_similarities(
        self,
        query_embedding: np.ndarray
    ) -> List[Tuple[int, float]]:
        """Calculate cosine similarities between query and all items."""
        if not self.embeddings:
            return []
        
        embeddings_array = np.array(self.embeddings)
        
        # Cosine similarity: dot product of normalized vectors
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        embeddings_norm = embeddings_array / (
            np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-8
        )
        
        similarities = np.dot(embeddings_norm, query_norm)
        
        return [(idx, float(sim)) for idx, sim in enumerate(similarities)]
    
    def _hash_based_embedding(self, text: str) -> np.ndarray:
        """
        Fallback embedding using hash-based method.
        
        Simple but deterministic: hash string to fixed-size vector.
        """
        # Deterministic seed from blake2b hash (independent of PYTHONHASHSEED)
        h = hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest()
        # Convert 8-byte digest to unsigned 64-bit integer
        seed_int = int.from_bytes(h, byteorder='big', signed=False)
        rng = np.random.default_rng(seed_int)
        embedding = rng.standard_normal(self.embedding_dim, dtype=np.float32)
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding
    
    def _trim_memory(self) -> None:
        """Trim memory to max size (FIFO)."""
        if len(self.items) <= self.max_memory_size:
            return
        
        # Remove oldest items
        n_to_remove = len(self.items) - self.max_memory_size
        
        self.items = self.items[n_to_remove:]
        self.embeddings = self.embeddings[n_to_remove:]
        self.metadata = self.metadata[n_to_remove:]
        
        # Rebuild index
        self.item_to_idx = {
            str(item): idx for idx, item in enumerate(self.items)
        }


class SimpleRAGSystem(RetrievalSystem):
    """
    Simplified RAG system without embeddings (faster, less accurate).
    
    Uses simple string matching for retrieval.
    Good for testing or when embedding models unavailable.
    """
    
    def __init__(self, max_memory_size: int = 10000):
        self.max_memory_size = max_memory_size
        self.items: List[Any] = []
        self.metadata: List[Dict[str, Any]] = []
    
    def add_to_memory(
        self,
        obj: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add object to memory."""
        self.items.append(obj)
        self.metadata.append(metadata or {})
        
        # Trim if needed
        if len(self.items) > self.max_memory_size:
            self.items = self.items[-self.max_memory_size:]
            self.metadata = self.metadata[-self.max_memory_size:]
    
    def retrieve_examples(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Retrieve examples (returns last k items)."""
        if filters:
            valid_items = [
                item for item, meta in zip(self.items, self.metadata)
                if all(meta.get(key) == value for key, value in filters.items())
            ]
            return valid_items[-k:]
        return self.items[-k:]
    
    def retrieve_prototypes(
        self,
        n_clusters: int = 10,
        items_per_cluster: int = 1
    ) -> List[Any]:
        """Retrieve prototypes (evenly spaced)."""
        if not self.items:
            return []
        step = max(1, len(self.items) // n_clusters)
        return self.items[::step][:n_clusters]
    
    def retrieve_constraints(
        self,
        query: str,
        k: int = 5
    ) -> List[str]:
        """Retrieve constraints (placeholder)."""
        return []
    
    def clear_memory(self) -> None:
        """Clear memory."""
        self.items.clear()
        self.metadata.clear()
    
    def get_memory_size(self) -> int:
        """Get memory size."""
        return len(self.items)

