# Redis cache for RAG pipeline

Optional Redis-backed caching for the Help Support Assistant RAG pipeline: **reduce LLM cost**, **improve response speed**, and **avoid recomputing embeddings**.

---

## Benefits

| Benefit | How Redis cache helps |
|--------|------------------------|
| **Reduce LLM cost** | Cached responses (when `temperature=0`) avoid repeated API calls and token usage for the same question + context. |
| **Improve response speed** | Cache hits return in milliseconds instead of running embedding, vector search, and LLM. |
| **Avoid recomputing embeddings** | Query embeddings are cached by text; repeated or similar queries reuse the same vector. |
| **Maintain consistent responses** | With `temperature=0`, the same prompt always returns the same cached answer. |
| **Handle repeated searches in a conversation** | Repeated user questions hit retrieval and/or response cache without re-embedding or re-calling the LLM. |
| **Improve throughput** | Less CPU (embeddings) and fewer outbound LLM requests, so the system handles more concurrent users. |
| **Support session memory without high cost** | Conversation turns that repeat earlier context benefit from cached retrieval and responses without storing full history in the LLM call. |

---

## What is cached

- **Embeddings:** `rag:embed:{hash(query)}` → query embedding vector (TTL: `CACHE_TTL_EMBEDDING`, default 24h).
- **Retrieval:** `rag:retrieve:{hash(query)}:{top_k}` → list of retrieved chunks (TTL: `CACHE_TTL_RETRIEVAL`, default 1h).
- **LLM response:** `rag:response:{hash(prompt)}` → model output (TTL: `CACHE_TTL_RESPONSE`, default 1h).  
  Response cache is **only used when `temperature=0`** so answers stay deterministic and consistent.

---

## Setup

### 1. Install dependency

Already included in `requirements.txt`:

```bash
pip install redis
```

### 2. Enable cache in `.env`

```env
REDIS_URL=redis://localhost:6379/0
CACHE_ENABLED=true
```

Optional TTLs (seconds):

```env
CACHE_TTL_EMBEDDING=86400
CACHE_TTL_RETRIEVAL=3600
CACHE_TTL_RESPONSE=3600
```

If `REDIS_URL` is missing or `CACHE_ENABLED` is not `true`, the app runs without cache (no Redis required).

---

## Local setup

### Option A: Docker (recommended)

```bash
docker run -d --name redis-rag -p 6379:6379 redis:7-alpine
```

Then set in `.env`:

```env
REDIS_URL=redis://localhost:6379/0
CACHE_ENABLED=true
```

### Option B: Redis installed on host

**Windows (WSL or native):**

- Install Redis via WSL: `sudo apt install redis-server` then `redis-server`.
- Or use [Memurai](https://www.memurai.com/) (Redis-compatible) and point `REDIS_URL` to it.

**macOS:**

```bash
brew install redis
brew services start redis
```

**Linux:**

```bash
sudo apt install redis-server
sudo systemctl start redis-server
```

Then:

```env
REDIS_URL=redis://localhost:6379/0
CACHE_ENABLED=true
```

---

## Cloud setup

### Redis Cloud (redis.com)

1. Create a free or paid database at [Redis Cloud](https://redis.com/try-free/).
2. Copy the **Public endpoint** (e.g. `redis-12345.redis.cloud.com:12345`) and, if required, the **default user password**.
3. In `.env`:

```env
REDIS_URL=redis://default:YOUR_PASSWORD@redis-12345.redis.cloud.com:12345
CACHE_ENABLED=true
```

Use TLS if provided (e.g. `rediss://` and port `6380`):

```env
REDIS_URL=rediss://default:YOUR_PASSWORD@redis-12345.redis.cloud.com:6380
CACHE_ENABLED=true
```

### AWS ElastiCache (Redis)

1. Create an ElastiCache Redis cluster (e.g. cache.t3.micro) in the same VPC as your app.
2. Use the cluster endpoint and port in `.env`:

```env
REDIS_URL=redis://your-cluster.xxxxx.cache.amazonaws.com:6379/0
CACHE_ENABLED=true
```

If using AUTH:

```env
REDIS_URL=redis://:YOUR_AUTH_TOKEN@your-cluster.xxxxx.cache.amazonaws.com:6379/0
CACHE_ENABLED=true
```

### Azure Cache for Redis

1. Create an Azure Cache for Redis instance in the Azure portal.
2. Under **Access keys**, copy the **Primary connection string** (or build it from host and key).
3. In `.env` (replace with your host and key):

```env
REDIS_URL=redis://:YOUR_ACCESS_KEY@your-cache.redis.cache.windows.net:6380?ssl=True
CACHE_ENABLED=true
```

---

## Verifying cache

- With cache enabled, the app log should show: `Redis cache connected: ...` at startup (or on first cache use).
- If Redis is down or misconfigured, you’ll see: `Redis cache disabled: ...` and the app continues without cache.

For an overview of how caching fits in the RAG flow, see the [LLM caching architecture diagram](notebooks/README.md#llm-caching-architecture) in the notebooks README.
