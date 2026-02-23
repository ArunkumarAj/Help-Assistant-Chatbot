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

_redis_client: Optional[Any] = None


def _get_redis():
    """Lazy Redis connection. Returns None if disabled or connection fails."""
    global _redis_client
    if _redis_client is False:
        return None
    if _redis_client is not None:
        return _redis_client
    if not getattr(settings, "cache_enabled", False) or not getattr(settings, "redis_url", ""):
        return None
    try:
        import redis
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        _redis_client.ping()
        logger.info("Redis cache connected: %s", settings.redis_url.split("@")[-1] if "@" in settings.redis_url else settings.redis_url)
        return _redis_client
    except Exception as e:
        logger.warning("Redis cache disabled: %s", e)
        _redis_client = False
        return None


def _key(*parts: str) -> str:
    """Build a cache key from parts; hash long values."""
    out = []
    for p in parts:
        if len(p) > 200:
            out.append(hashlib.sha256(p.encode("utf-8")).hexdigest()[:32])
        else:
            out.append(p)
    return "rag:" + ":".join(out)


def cache_get_embedding(query_prefix: str) -> Optional[List[float]]:
    """Return cached query embedding if present."""
    client = _get_redis()
    if not client:
        return None
    key = _key("embed", query_prefix)
    try:
        raw = client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.debug("Cache get embedding failed: %s", e)
        return None


def cache_set_embedding(query_prefix: str, embedding: List[float]) -> None:
    """Store query embedding in cache."""
    client = _get_redis()
    if not client:
        return
    key = _key("embed", query_prefix)
    ttl = getattr(settings, "cache_ttl_embedding", 86400)
    try:
        client.setex(key, ttl, json.dumps(embedding))
    except Exception as e:
        logger.debug("Cache set embedding failed: %s", e)


def cache_get_retrieval(query_prefix: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
    """Return cached retrieval results (list of hits with _source.text, _source.document_name)."""
    client = _get_redis()
    if not client:
        return None
    key = _key("retrieve", query_prefix, str(top_k))
    try:
        raw = client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.debug("Cache get retrieval failed: %s", e)
        return None


def cache_set_retrieval(query_prefix: str, top_k: int, results: List[Dict[str, Any]]) -> None:
    """Store retrieval results in cache."""
    client = _get_redis()
    if not client:
        return
    key = _key("retrieve", query_prefix, str(top_k))
    ttl = getattr(settings, "cache_ttl_retrieval", 3600)
    try:
        client.setex(key, ttl, json.dumps(results))
    except Exception as e:
        logger.debug("Cache set retrieval failed: %s", e)


def cache_get_response(prompt_hash: str) -> Optional[str]:
    """Return cached LLM response if present."""
    client = _get_redis()
    if not client:
        return None
    key = _key("response", prompt_hash)
    try:
        return client.get(key)
    except Exception as e:
        logger.debug("Cache get response failed: %s", e)
        return None


def cache_set_response(prompt_hash: str, response: str) -> None:
    """Store LLM response in cache."""
    client = _get_redis()
    if not client:
        return
    key = _key("response", prompt_hash)
    ttl = getattr(settings, "cache_ttl_response", 3600)
    try:
        client.setex(key, ttl, response)
    except Exception as e:
        logger.debug("Cache set response failed: %s", e)


def hash_prompt(prompt: str) -> str:
    """Stable hash of prompt for response cache key."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]
