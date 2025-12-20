"""
CCAD Engine - Domain-agnostic hypothesis exploration with cumulative anti-duplication.

This is the core of the CCAD (Cumulative Context Anti-Duplication) framework.
It orchestrates LLM-based generation with real-time deduplication, adaptive
exploration, and saturation detection.

The engine is completely domain-agnostic and works with any combinatorial space
by relying on pluggable adapters (Canonicalizer, Scorer, etc.).
"""

import asyncio
import logging
from typing import List, Set, Any, Optional, Dict, Tuple, Callable
from datetime import datetime

from .interfaces import Canonicalizer, DistanceMetric, Scorer, RetrievalSystem
from .metrics import MetricsTracker
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


class CCAdEngine:
    """
    Cumulative Context Anti-Duplication Engine.
    
    Core component of the CCAD framework that implements:
        1. Cumulative memory of all generated items
        2. Contextual anti-duplication prompting
        3. Real-time canonicalization and deduplication
        4. Adaptive temperature control
        5. Early stopping on saturation
        6. Comprehensive metrics tracking
    
    Usage:
        engine = CCAdEngine(
            canonicalizer=MyCanonicalizer(),
            scorer=MyScorer(),
            llm_generator=my_llm_function
        )
        
        results = await engine.generate(
            prompts=["Generate items with property X"],
            target_counts=[100],
            temperature=0.8
        )
    """
    
    def __init__(
        self,
        canonicalizer: Canonicalizer,
        llm_generator: Callable,  # async function(prompt, temperature, n) -> (items, usage_stats)
        scorer: Optional[Scorer] = None,
        distance_metric: Optional[DistanceMetric] = None,
        retrieval_system: Optional[RetrievalSystem] = None,
        fuzzy_threshold: float = 0.95,  # For near-duplicate detection (Tanimoto)
        fuzzy_policy: str = "count",    # "reject" | "count" | "allow"
        enable_fuzzy_dedup: bool = False,
        enable_anti_dup_context: bool = True,  # NEW: Enable/disable anti-dup context
        enable_rag_context: bool = False,
        max_context_items: int = 50,  # How many items to show in anti-dup context
        saturation_threshold: float = 0.8,  # When to trigger early stopping
        adaptive_temperature: bool = True,
        temperature_increase: float = 0.1,
        max_temperature: float = 1.2,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        Initialize CCAD Engine.
        
        Args:
            canonicalizer: Domain-specific canonicalizer
            llm_generator: Async function to call LLM
            scorer: Optional scorer for quality evaluation
            distance_metric: Optional for fuzzy deduplication
            retrieval_system: Optional RAG system
            fuzzy_threshold: Similarity threshold for near-duplicates
            enable_fuzzy_dedup: Enable fuzzy deduplication (requires distance_metric)
            enable_anti_dup_context: Enable anti-duplication context injection (core CCAD feature)
            enable_rag_context: Use RAG for context building (requires retrieval_system)
            max_context_items: Max items to show in anti-dup prompt
            saturation_threshold: Duplicate rate threshold for early stopping
            adaptive_temperature: Enable adaptive temperature control
            temperature_increase: How much to increase temperature on saturation
            max_temperature: Maximum allowed temperature
            logger_instance: Optional custom logger
        """
        self.canonicalizer = canonicalizer
        self.llm_generator = llm_generator
        self.scorer = scorer
        self.distance_metric = distance_metric
        self.retrieval = retrieval_system
        
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_policy = fuzzy_policy
        self.enable_fuzzy_dedup = enable_fuzzy_dedup and distance_metric is not None
        self.enable_anti_dup_context = enable_anti_dup_context  # NEW
        self.enable_rag_context = enable_rag_context and retrieval_system is not None
        self.max_context_items = max_context_items
        self.saturation_threshold = saturation_threshold
        self.adaptive_temperature = adaptive_temperature
        self.temperature_increase = temperature_increase
        self.max_temperature = max_temperature
        
        self.logger = logger_instance or logger
        
        # Persistent state across generation
        self.seen_canonical: Set[str] = set()
        self.seen_objects: List[Any] = []
        self.canonical_history: List[str] = []
        self._seen_rdkit_fps: List[Any] = []  # RDKit fingerprints aligned with canonical_history
        self.metrics = MetricsTracker()
    
    def reset_state(self) -> None:
        """Reset engine state (useful for multiple independent runs)."""
        self.seen_canonical.clear()
        self.seen_objects.clear()
        self.canonical_history.clear()
        self._seen_rdkit_fps.clear()
        self.metrics = MetricsTracker()
    
    def build_anti_dup_context(self, use_rag: bool = None) -> str:
        """
        Build anti-duplication context for LLM prompt.
        
        Two modes:
            1. Basic: Show last N items explicitly
            2. RAG: Show retrieved prototypes (more efficient)
        
        Args:
            use_rag: Override engine default for RAG usage
        
        Returns:
            Anti-duplication context string to inject in prompt (empty if disabled)
        """
        # NEW: Check if anti-dup context is enabled at all
        if not self.enable_anti_dup_context:
            return ""
        
        if len(self.seen_canonical) == 0:
            return ""
        
        use_rag = use_rag if use_rag is not None else self.enable_rag_context
        
        if use_rag and self.retrieval:
            # RAG mode: Show prototypes with cluster information (optimized)
            try:
                n_clusters = min(15, max(5, len(self.seen_canonical) // 20))
                
                # Try to get prototypes with counts
                try:
                    prototypes_with_counts = self.retrieval.retrieve_prototypes_with_counts(
                        n_clusters=n_clusters
                    )
                    
                    if prototypes_with_counts:
                        # Build context with cluster information
                        proto_lines = []
                        for i, (proto, count) in enumerate(prototypes_with_counts, 1):
                            proto_lines.append(f"  Region {i}: {proto} (cluster of {count} similar items)")
                        
                        context = (
                            f"\n\n⚠️  ANTI-DUPLICATION CONTEXT (Optimized RAG):\n"
                            f"Total unique items generated: {len(self.seen_canonical)}\n\n"
                            f"CHEMICAL/SEMANTIC SPACE ALREADY EXPLORED:\n"
                            f"The following {len(prototypes_with_counts)} prototypes represent regions of the space already covered.\n"
                            f"Each prototype anchors a cluster of similar items.\n\n"
                            + "\n".join(proto_lines) + "\n\n"
                            f"REQUIREMENT:\n"
                            f"- Generate items in UNEXPLORED regions (far from all prototypes)\n"
                            f"- Maximize STRUCTURAL/SEMANTIC DIVERSITY\n"
                            f"- Each new item must be CANONICALLY UNIQUE\n"
                            f"- Avoid generating near-duplicates or close variants"
                        )
                    else:
                        raise ValueError("No prototypes returned")
                
                except Exception:
                    # Fallback to simple prototypes without counts
                    prototypes = self.retrieval.retrieve_prototypes(
                        n_clusters=n_clusters,
                        items_per_cluster=1
                    )
                    if not prototypes:
                        prototypes = self.canonical_history[-self.max_context_items:]
                    
                    context = (
                        f"\n\n⚠️  CRITICAL ANTI-DUPLICATION REQUIREMENT:\n"
                        f"You have ALREADY generated {len(self.seen_canonical)} unique items.\n\n"
                        f"FORBIDDEN PROTOTYPES (and similar variants):\n"
                        f"{', '.join(str(p) for p in prototypes)}\n\n"
                        f"STRICT REQUIREMENT:\n"
                        f"- Generate ONLY items that are STRUCTURALLY/SEMANTICALLY DIFFERENT\n"
                        f"- DO NOT generate the same item with alternative notation\n"
                        f"- Each item must be UNIQUE at the canonical level\n"
                        f"- Explore NEW regions of the space"
                    )
            except Exception as e:
                self.logger.warning(f"RAG context building failed: {e}, falling back to basic mode")
                use_rag = False
        
        if not use_rag:
            # Basic mode: Show last N items
            recent = self.canonical_history[-self.max_context_items:]
            context = (
                f"\n\n⚠️  CRITICAL ANTI-DUPLICATION REQUIREMENT:\n"
                f"You have ALREADY generated {len(self.seen_canonical)} unique items.\n\n"
                f"FORBIDDEN ITEMS (DO NOT repeat - even with different notation):\n"
                f"{', '.join(recent)}\n\n"
                f"STRICT REQUIREMENT:\n"
                f"- Generate ONLY items that are STRUCTURALLY/SEMANTICALLY DIFFERENT\n"
                f"- DO NOT generate the same item with alternative notation\n"
                f"- Each item must be UNIQUE\n"
                f"- Explore NEW regions of the space"
            )
        
        return context
    
    async def process_batch(
        self,
        candidates: List[Any],
        enable_fuzzy: bool = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Process a batch of candidates with deduplication.
        
        Steps:
            1. Validate each candidate
            2. Canonicalize to standard form
            3. Check exact duplicates
            4. (Optional) Check fuzzy duplicates
            5. (Optional) Score candidates
            6. Track metrics
        
        Args:
            candidates: List of raw candidates from LLM
            enable_fuzzy: Override engine default for fuzzy dedup
        
        Returns:
            Tuple of (accepted_unique_candidates, chunk_metrics)
        """
        enable_fuzzy = enable_fuzzy if enable_fuzzy is not None else self.enable_fuzzy_dedup
        
        raw_generated = len(candidates)
        unique_accepted = []
        exact_dups = 0
        near_dups = 0
        invalid = 0
        
        for candidate in candidates:
            # 1. Validate
            if not self.canonicalizer.is_valid(candidate):
                invalid += 1
                continue
            
            # 2. Preprocess (optional normalization)
            preprocessed = self.canonicalizer.preprocess(candidate)
            
            # 3. Canonicalize
            try:
                canonical = self.canonicalizer.canonicalize(preprocessed)
            except Exception as e:
                self.logger.warning(f"Canonicalization failed for {candidate}: {e}")
                invalid += 1
                continue
            
            # 4. Check exact duplicates
            if canonical in self.seen_canonical:
                exact_dups += 1
                continue
            
            # 5. (Optional) Fuzzy deduplication (chemical near-dup via ECFP4 Tanimoto)
            # If RDKit available, compute fingerprint of candidate and compare to recent ones
            is_near_dup = False
            if enable_fuzzy:
                try:
                    mol = Chem.MolFromSmiles(canonical)
                    if mol is not None:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                        # Compare against recent fingerprints for performance
                        recent_fps = self._seen_rdkit_fps[-min(200, len(self._seen_rdkit_fps)):]
                        if recent_fps:
                            # Compute max similarity
                            sims = [DataStructs.TanimotoSimilarity(fp, seen_fp) for seen_fp in recent_fps]
                            max_sim = max(sims) if sims else 0.0
                            if max_sim >= self.fuzzy_threshold:
                                is_near_dup = True
                                near_dups += 1
                    else:
                        # If invalid mol slipped through, treat as invalid
                        invalid += 1
                        continue
                except Exception as e:
                    self.logger.debug(f"Near-dup check failed: {e}")
            
            # 6. (Optional) Score candidate
            if self.scorer:
                try:
                    passes, reason = self.scorer.passes_filters(candidate)
                    if not passes:
                        self.logger.debug(f"Candidate rejected by filter: {reason}")
                        invalid += 1
                        continue
                except Exception as e:
                    self.logger.warning(f"Scoring failed for {candidate}: {e}")
            
            # 7. Apply fuzzy policy
            if enable_fuzzy and is_near_dup and self.fuzzy_policy == "reject":
                # Reject near-duplicates under strict policy (do not accept)
                continue
            # If "count" (default), accept but tracked as near_dups; if "allow", ignore near-dup marking
            
            # 8. Accept
            self.seen_canonical.add(canonical)
            self.seen_objects.append(candidate)
            self.canonical_history.append(canonical)
            # Keep fingerprint aligned with history
            try:
                mol_acc = Chem.MolFromSmiles(canonical)
                if mol_acc is not None:
                    fp_acc = AllChem.GetMorganFingerprintAsBitVect(mol_acc, radius=2, nBits=2048)
                    self._seen_rdkit_fps.append(fp_acc)
            except Exception:
                pass
            unique_accepted.append(candidate)
            
            # 9. (Optional) Add to RAG memory
            if self.retrieval:
                try:
                    scores = self.scorer.score(candidate) if self.scorer else {}
                    self.retrieval.add_to_memory(candidate, metadata={'scores': scores})
                except Exception as e:
                    self.logger.debug(f"Failed to add to RAG memory: {e}")
        
        # Chunk metrics
        # Track both raw and valid-generated counts
        valid_generated = len(unique_accepted) + exact_dups + max(0, near_dups)
        
        chunk_metrics = {
            'raw_generated': raw_generated,
            'valid_generated': valid_generated,
            'unique_accepted': len(unique_accepted),
            'exact_duplicates': exact_dups,
            'near_duplicates': near_dups,
            'invalid': invalid,
            # Duplicate rate among valid (legacy behavior used for adaptive control)
            'duplicate_rate': (exact_dups + max(0, near_dups)) / max(1, valid_generated),
            # Additional: duplicate rate relative to raw
            'duplicate_rate_raw': (exact_dups + max(0, near_dups)) / max(1, raw_generated)
        }
        
        # Update global metrics
        # Only exact duplicates were actually filtered; near-dups are advisory
        self.metrics.add_chunk(raw_generated, valid_generated, len(unique_accepted), exact_dups)
        
        return unique_accepted, chunk_metrics
    
    async def generate(
        self,
        prompts: List[str],
        target_counts: List[int],
        temperature: float = 0.8,
        max_per_call: int = 50,
        min_per_call: int = 5,
        max_retries: int = 3,
        max_iterations: int = 200,
        enable_adaptive_temp: bool = None,
        enable_early_stopping: bool = True
    ) -> Dict[str, Any]:
        """
        Main generation loop with CCAD.
        
        Args:
            prompts: List of prompt themes to explore
            target_counts: Target number of items per prompt
            temperature: Initial LLM temperature
            max_per_call: Maximum items to request per LLM call
            min_per_call: Minimum items to request (before giving up)
            max_retries: Max retry attempts per chunk
            max_iterations: Safety limit on iterations per prompt
            enable_adaptive_temp: Override engine default
            enable_early_stopping: Stop on saturation detection
        
        Returns:
            Dictionary with:
                - 'items': List of accepted unique items
                - 'metrics': Final metrics
                - 'summary': Human-readable summary
        """
        enable_adaptive_temp = enable_adaptive_temp if enable_adaptive_temp is not None else self.adaptive_temperature
        
        all_items = []
        
        for idx, (prompt, target_count) in enumerate(zip(prompts, target_counts)):
            if target_count <= 0:
                continue
            
            self.logger.info(f"Starting CCAD generation for prompt {idx+1}/{len(prompts)}: target={target_count}")
            
            remaining = target_count
            chunk_temperature = temperature
            iteration = 0
            
            while remaining > 0 and iteration < max_iterations:
                iteration += 1
                chunk_size = min(max_per_call, remaining)
                
                # Build anti-duplication context
                anti_dup_context = self.build_anti_dup_context()
                
                # Construct full prompt
                prompt_with_context = (
                    f"{prompt}"
                    f"{anti_dup_context}\n\n"
                    f"Generate exactly {chunk_size} NEW, UNIQUE items."
                )
                
                # Call LLM (with retries)
                raw_items = []
                for attempt in range(max_retries):
                    try:
                        raw_items, usage_stats = await self.llm_generator(
                            prompt_with_context,
                            chunk_temperature,
                            chunk_size
                        )
                        if len(raw_items) >= max(1, chunk_size // 4):
                            break  # Good enough
                        
                        # Reduce chunk size and retry
                        chunk_size = max(min_per_call, chunk_size // 2)
                        self.logger.info(f"Low yield, reducing chunk_size to {chunk_size}, retry {attempt+1}")
                    
                    except Exception as e:
                        self.logger.error(f"LLM call failed (attempt {attempt+1}): {e}")
                        if attempt == max_retries - 1:
                            raise
                
                if not raw_items:
                    self.logger.warning(f"No items generated after {max_retries} attempts, skipping this chunk")
                    break
                
                # Process batch with CCAD
                unique_items, chunk_metrics = await self.process_batch(raw_items, enable_fuzzy=self.enable_fuzzy_dedup or (self.fuzzy_policy in ("reject","count")))
                all_items.extend(unique_items)
                
                # Log chunk performance
                self.logger.info(
                    f"CCAD Chunk {iteration} (prompt {idx+1}): "
                    f"{chunk_metrics['unique_accepted']} new unique, "
                    f"{chunk_metrics['exact_duplicates']} duplicates, "
                    f"{chunk_metrics['near_duplicates']} near-dups, "
                    f"{chunk_metrics['invalid']} invalid "
                    f"(dup_rate: {chunk_metrics['duplicate_rate']:.1%}, "
                    f"total_unique: {len(self.seen_canonical)})"
                )
                
                # Adaptive temperature
                if enable_adaptive_temp and chunk_metrics['duplicate_rate'] > 0.6 and iteration > 3:
                    chunk_temperature = min(chunk_temperature + self.temperature_increase, self.max_temperature)
                    self.logger.info(f"CCAD: High dup rate, increasing temperature to {chunk_temperature:.2f}")
                
                # Simple adaptive policy (rule-based)
                # If duplicate rates remain high, tighten fuzzy and increase context window
                valid_rate = (chunk_metrics['valid_generated'] / max(1, chunk_metrics['raw_generated']))
                if chunk_metrics['duplicate_rate'] > 0.6 and iteration > 3:
                    # Increase context up to 100
                    old_k = self.max_context_items
                    self.max_context_items = min(100, self.max_context_items + 10)
                    if self.fuzzy_policy == "count" and self.fuzzy_threshold >= 0.85:
                        self.fuzzy_policy = "reject"
                        self.fuzzy_threshold = max(0.80, self.fuzzy_threshold - 0.02)
                    self.logger.info(f"CCAD policy: dup high → K {old_k}->{self.max_context_items}, fuzzy {self.fuzzy_policy}@{self.fuzzy_threshold:.2f}")
                # If valid rate drops too low, reduce temperature and shorten context a bit
                if valid_rate < 0.5 and iteration > 2:
                    old_temp = chunk_temperature
                    chunk_temperature = max(0.3, chunk_temperature - 0.1)
                    old_k = self.max_context_items
                    self.max_context_items = max(10, self.max_context_items - 5)
                    self.logger.info(f"CCAD policy: low valid_rate → temp {old_temp:.2f}->{chunk_temperature:.2f}, K {old_k}->{self.max_context_items}")
                
                # Early stopping on saturation
                if enable_early_stopping and iteration > 5:
                    recent_dup_rates = self.metrics.duplicate_rates[-3:]
                    if (len(recent_dup_rates) >= 3 and 
                        all(r > 0.7 for r in recent_dup_rates) and
                        chunk_metrics['duplicate_rate'] > self.saturation_threshold):
                        self.logger.warning(
                            f"CCAD: Saturation detected (dup_rate consistently >70%), "
                            f"stopping generation for this prompt theme"
                        )
                        break
                
                # Update remaining
                remaining -= len(unique_items)
                
                # Safeguard: if no progress, stop
                if len(unique_items) == 0 and chunk_size <= min_per_call:
                    self.logger.warning("No unique items generated at minimum chunk size, stopping")
                    break
        
        # Final results
        final_metrics = self.metrics.get_metrics()
        summary = self.metrics.get_summary()
        
        return {
            'items': all_items,
            'metrics': final_metrics,
            'summary': summary,
            'n_unique': len(self.seen_canonical),
            'n_items': len(all_items)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.get_metrics()
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        return self.metrics.get_summary()
    
    def should_stop(self) -> bool:
        """Check if generation should stop based on saturation."""
        return self.metrics.should_stop(saturation_threshold=self.saturation_threshold)

