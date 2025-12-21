from eurekamesh.core.rag_system import InMemoryRAGSystem
import numpy as np


def test_hash_based_embedding_is_deterministic():
    rag1 = InMemoryRAGSystem()
    rag2 = InMemoryRAGSystem()
    text = "CCO"
    # Access the fallback embedding directly
    emb1 = rag1._hash_based_embedding(text)
    emb2 = rag2._hash_based_embedding(text)
    assert emb1.shape == emb2.shape
    # Exact equality because rng seeded from blake2b digest
    assert np.allclose(emb1, emb2, atol=0.0)



