"""
Redis cache for RAG: query embeddings, retrieval results, and LLM responses.
Reduces LLM cost, avoids recomputing embeddings, and speeds up repeated queries.
When REDIS_URL is unset or CACHE_ENABLED=false, all operations no-op (return None on get, ignore set).
"""
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from core.config import settings

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Redis connection (lazy, single instance)
# -----------------------------------------------------------------------------

_redis_client: Optional[Any] = None


def _get_redis_client() -> Optional[Any]:
    """Return connected Redis client, or None if cache is disabled or connection fails."""
    global _redis_client
    if _redis_client is False:
        return None
    if _redis_client is not None:
        return _redis_client
    if not getattr(settings, "cache_enabled", False) or not getattr(settings, "redis_url", ""):
        logger.info(
            "Redis cache not configured (set REDIS_URL and CACHE_ENABLED=true in .env to enable)"
        )
        _redis_client = False
        return None
    try:
        import redis
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        _redis_client.ping()
        url_display = settings.redis_url.split("@")[-1] if "@" in settings.redis_url else settings.redis_url
        logger.info("Redis cache connected: %s", url_display)
        return _redis_client
    except Exception as e:
        logger.warning("Redis cache disabled (connection failed): %s", e)
        _redis_client = False
        return None


def _build_cache_key(*parts: str) -> str:
    """Build a cache key from parts; long values are hashed to keep keys short."""
    key_parts = []
    for part in parts:
        if len(part) > 200:
            key_parts.append(hashlib.sha256(part.encode("utf-8")).hexdigest()[:32])
        else:
            key_parts.append(part)
    return "rag:" + ":".join(key_parts)


# -----------------------------------------------------------------------------
# Embedding cache
# -----------------------------------------------------------------------------


def cache_get_embedding(query_prefix: str) -> Optional[List[float]]:
    """Return cached query embedding if present; otherwise None."""
    client = _get_redis_client()
    if not client:
        return None
    key = _build_cache_key("embed", query_prefix)
    try:
        raw_value = client.get(key)
        if raw_value is None:
            return None
        logger.info("RAG cache hit: embedding")
        return json.loads(raw_value)
    except Exception as e:
        logger.debug("Cache get embedding failed: %s", e)
        return None


def cache_set_embedding(query_prefix: str, embedding: List[float]) -> None:
    """Store query embedding in cache. No-op if cache is disabled."""
    client = _get_redis_client()
    if not client:
        return
    key = _build_cache_key("embed", query_prefix)
    ttl_seconds = getattr(settings, "cache_ttl_embedding", 86400)
    try:
        client.setex(key, ttl_seconds, json.dumps(embedding))
    except Exception as e:
        logger.debug("Cache set embedding failed: %s", e)


# -----------------------------------------------------------------------------
# Retrieval cache
# -----------------------------------------------------------------------------


def cache_get_retrieval(query_prefix: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
    """Return cached retrieval results (list of hits with _source) if present; otherwise None."""
    client = _get_redis_client()
    if not client:
        return None
    key = _build_cache_key("retrieve", query_prefix, str(top_k))
    try:
        raw_value = client.get(key)
        if raw_value is None:
            return None
        logger.info("RAG cache hit: retrieval")
        return json.loads(raw_value)
    except Exception as e:
        logger.debug("Cache get retrieval failed: %s", e)
        return None


def cache_set_retrieval(query_prefix: str, top_k: int, results: List[Dict[str, Any]]) -> None:
    """Store retrieval results in cache. No-op if cache is disabled."""
    client = _get_redis_client()
    if not client:
        return
    key = _build_cache_key("retrieve", query_prefix, str(top_k))
    ttl_seconds = getattr(settings, "cache_ttl_retrieval", 3600)
    try:
        client.setex(key, ttl_seconds, json.dumps(results))
    except Exception as e:
        logger.debug("Cache set retrieval failed: %s", e)


# -----------------------------------------------------------------------------
# LLM response cache (keyed by prompt hash)
# -----------------------------------------------------------------------------


def cache_get_response(prompt_hash: str) -> Optional[str]:
    """Return cached LLM response for this prompt hash if present; otherwise None."""
    client = _get_redis_client()
    if not client:
        return None
    key = _build_cache_key("response", prompt_hash)
    try:
        raw_value = client.get(key)
        if raw_value is None:
            return None
        logger.info("RAG cache hit: response (message served from cache)")
        return raw_value
    except Exception as e:
        logger.debug("Cache get response failed: %s", e)
        return None


def cache_set_response(prompt_hash: str, response: str) -> None:
    """Store LLM response in cache. No-op if cache is disabled."""
    client = _get_redis_client()
    if not client:
        return
    key = _build_cache_key("response", prompt_hash)
    ttl_seconds = getattr(settings, "cache_ttl_response", 3600)
    try:
        client.setex(key, ttl_seconds, response)
    except Exception as e:
        logger.debug("Cache set response failed: %s", e)


def hash_prompt(prompt: str) -> str:
    """Stable hash of prompt for response cache key."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]
