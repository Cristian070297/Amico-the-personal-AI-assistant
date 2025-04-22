# memory.py
import math
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Import necessary classes from other modules
from database import DatabaseManager
from services import OpenAIService

class MemoryRetriever:
    """Handles scoring memories and selecting relevant context."""
    def __init__(self, db_manager: DatabaseManager, openai_service: OpenAIService):
        self.db_manager = db_manager
        self.openai_service = openai_service

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculates cosine similarity between two vectors."""
        if not isinstance(vec1, list) or not isinstance(vec2, list) or len(vec1) == 0 or len(vec1) != len(vec2):
            return 0.0
        # Avoid calculating magnitude for zero vectors explicitly
        if all(v == 0.0 for v in vec1) or all(v == 0.0 for v in vec2):
             return 0.0

        dot_product = sum(p * q for p, q in zip(vec1, vec2))
        magnitude = (sum(p ** 2 for p in vec1) ** 0.5) * (sum(q ** 2 for q in vec2) ** 0.5)
        if not magnitude:
            return 0.0
        similarity = dot_product / magnitude
        return max(-1.0, min(1.0, similarity)) # Clamp

    def _calculate_context_scores(self, query_embedding: List[float], memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculates recency, similarity, and importance scores."""
        scored_memories = []
        now = datetime.now()

        # Filter only valid memories for scoring
        valid_memories = [
            mem for mem in memories
            if mem.get("content") and mem.get("timestamp") and mem.get("embedding") and isinstance(mem.get("embedding"), list)
        ]
        if not valid_memories: return []

        max_importance = max((len(mem["content"]) for mem in valid_memories), default=1)
        if max_importance == 0: max_importance = 1

        oldest_time = min(mem["timestamp"] for mem in valid_memories)
        max_time_diff_seconds = max(1.0, (now - oldest_time).total_seconds()) # Ensure > 0

        for mem in valid_memories:
            similarity_score = self._cosine_similarity(query_embedding, mem["embedding"])
            time_diff_seconds = (now - mem["timestamp"]).total_seconds()
            # Adjusted recency: Exponential decay idea, scaled
            # Adjusted scaling factor for potentially faster decay (e.g., divide by smaller fraction of max_time_diff)
            recency_score = math.exp(-time_diff_seconds / (max_time_diff_seconds * 0.1)) # Faster decay

            importance_score = len(mem["content"]) / max_importance

            # Weights: Recency=0.2, Similarity=0.5, Importance=0.3
            weighted_score = (0.2 * recency_score) + (0.5 * similarity_score) + (0.3 * importance_score)

            scored_memories.append({
                "role": mem["role"],
                "content": mem["content"],
                "score": weighted_score,
                "timestamp": mem["timestamp"] # For tie-breaking
            })
        return scored_memories

    def _select_context_memories(self, scored_memories: List[Dict[str, Any]], max_memories: int = 10) -> str:
        """Selects top N memories based on weighted score."""
        if not scored_memories: return ""

        scored_memories.sort(key=lambda x: (x["score"], x["timestamp"]), reverse=True)
        context_list = [f"{mem['role']}: {mem['content']}" for mem in scored_memories[:max_memories]]
        return "\n".join(reversed(context_list)) # Chronological order

    def retrieve_relevant_context(self, query_text: str, fetch_limit: int = 100, context_limit: int = 5) -> Tuple[str, int, int]:
        """Fetches, scores, and selects relevant memories for a query.

        Returns:
            Tuple[str, int, int]: (selected_context_string, num_memories_fetched, num_memories_scored)
        """
        query_embedding = self.openai_service.get_embedding(query_text)
        long_term_context = "No relevant memories found."
        num_memories_fetched = 0
        num_memories_scored = 0

        # Check if embedding is valid (non-zero vector) before proceeding
        if not any(query_embedding):
            print("Warning: Could not generate embedding for query. Skipping similarity context.")
            long_term_context = "No similar memories found (embedding failed)."
            # Return context string, 0 fetched, 0 scored
            return long_term_context, num_memories_fetched, num_memories_scored

        recent_memories = self.db_manager.fetch_recent_memories(limit=fetch_limit)
        num_memories_fetched = len(recent_memories)

        if recent_memories:
            scored_memories = self._calculate_context_scores(query_embedding, recent_memories)
            num_memories_scored = len(scored_memories) # Number actually scored
            long_term_context = self._select_context_memories(scored_memories, max_memories=context_limit)
            if not long_term_context and scored_memories: # If selection resulted in empty but scoring happened
                 long_term_context = "Found memories, but none scored high enough for context."
            elif not scored_memories: # If scoring resulted in empty list
                 long_term_context = "Fetched memories, but none were valid for scoring."

        else:
            long_term_context = "No past memories found in database."

        return long_term_context, num_memories_fetched, num_memories_scored