"""
RAG evaluation metrics: retrieval, generation, end-to-end, and system.
Minimal dependencies; NLI uses transformers when available.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional NLI (faithfulness)
_nli_pipeline = None

def _get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        try:
            from transformers import pipeline
            _nli_pipeline = pipeline(
                "text-classification",
                model="microsoft/deberta-v3-base-mnli",
                top_k=None,
                device=-1,
            )
        except Exception:
            _nli_pipeline = False
    return _nli_pipeline if _nli_pipeline else None


# --- Retrieval ---

def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _any_gold_in_top_k(retrieved_texts: List[str], gold_passages: List[str], k: int) -> bool:
    if not gold_passages or k <= 0:
        return False
    top = [ _normalize(t) for t in retrieved_texts[:k] ]
    gold_norm = [ _normalize(g) for g in gold_passages ]
    for g in gold_norm:
        for t in top:
            if g in t or t in g or _jaccard_similarity(g, t) > 0.5:
                return True
    return False


def _jaccard_similarity(a: str, b: str) -> float:
    a_set = set(a.split())
    b_set = set(b.split())
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def recall_at_k(retrieved_texts: List[str], gold_passages: List[str], k: int) -> float:
    """Per-query: 1 if any gold passage is in top-k, else 0."""
    return 1.0 if _any_gold_in_top_k(retrieved_texts, gold_passages, k) else 0.0


def mrr_at_k(retrieved_texts: List[str], gold_passages: List[str], k: int) -> float:
    """Reciprocal rank of first relevant passage (binary relevance)."""
    if not gold_passages or k <= 0:
        return 0.0
    gold_norm = [_normalize(g) for g in gold_passages]
    for i, t in enumerate(retrieved_texts[:k]):
        tn = _normalize(t)
        for g in gold_norm:
            if g in tn or tn in g or _jaccard_similarity(g, tn) > 0.5:
                return 1.0 / (i + 1)
    return 0.0


def _relevance_binary(retrieved_text: str, gold_passages: List[str]) -> int:
    tn = _normalize(retrieved_text)
    gold_norm = [_normalize(g) for g in gold_passages]
    for g in gold_norm:
        if g in tn or tn in g or _jaccard_similarity(g, tn) > 0.5:
            return 1
    return 0


def ndcg_at_k(retrieved_texts: List[str], gold_passages: List[str], k: int) -> float:
    """nDCG@k with binary relevance (no graded labels)."""
    if not gold_passages or k <= 0:
        return 0.0
    rel = [_relevance_binary(t, gold_passages) for t in retrieved_texts[:k]]
    dcg = sum(rel[i] / (i + 2) for i in range(len(rel)))  # 1-based rank
    ideal = sorted(rel, reverse=True)
    idcg = sum(ideal[i] / (i + 2) for i in range(len(ideal)))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def coverage(retrieved_texts: List[str], gold_passages: List[str]) -> float:
    """Fraction of gold passages that appear in retrieved set (soft match)."""
    if not gold_passages:
        return 1.0
    gold_norm = [_normalize(g) for g in gold_passages]
    covered = 0
    for g in gold_norm:
        for t in retrieved_texts:
            tn = _normalize(t)
            if g in tn or tn in g or _jaccard_similarity(g, tn) > 0.5:
                covered += 1
                break
    return covered / len(gold_passages)


def redundancy(embeddings: List[List[float]]) -> float:
    """Fraction of retrieved chunk pairs with cosine similarity > 0.95."""
    if not embeddings or len(embeddings) < 2:
        return 0.0
    n = len(embeddings)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = _cosine_sim(embeddings[i], embeddings[j])
            if c > 0.95:
                count += 1
    pairs = n * (n - 1) / 2
    return count / pairs if pairs else 0.0


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na * nb == 0:
        return 0.0
    return dot / (na * nb)


# --- Generation: Faithfulness (NLI) ---

def faithfulness_nli(context: str, answer: str) -> float:
    """
    Sentence-level entailment of answer vs context. Returns average entailment score.
    Uses DeBERTa MNLI; label 'ENTAILMENT' -> 1, else 0 per sentence, then average.
    """
    pipe = _get_nli_pipeline()
    if pipe is None:
        return 0.0
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        sentences = nltk.sent_tokenize(answer.strip())
    except Exception:
        sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
    if not sentences:
        return 1.0
    context_trunc = (context + " ")[:4000]
    scores = []
    for sent in sentences:
        if not sent or len(sent) < 3:
            continue
        # MNLI: premise=context, hypothesis=claim
        result = pipe(context_trunc, sent.strip()[:512])
        if isinstance(result, list) and result:
            labels = result[0] if isinstance(result[0], list) else result
            for item in labels:
                if isinstance(item, dict):
                    lab = item.get("label", "")
                    if "ENTAIL" in lab.upper():
                        scores.append(item.get("score", 0.0))
                        break
                    if "CONTRADICT" in lab.upper():
                        scores.append(0.0)
                        break
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)
    return (sum(scores) / len(scores)) if scores else 1.0


def parse_llm_judge_faithfulness(raw: str) -> Tuple[float, List[str]]:
    """Parse LLM judge JSON: supported_fraction, unsupported_claims."""
    try:
        # strip markdown code block if present
        s = raw.strip()
        for start in ("```json", "```"):
            if s.startswith(start):
                s = s[len(start):].strip()
            if s.endswith("```"):
                s = s[:-3].strip()
        obj = json.loads(s)
        frac = float(obj.get("supported_fraction", 0.0))
        claims = obj.get("unsupported_claims") or []
        if not isinstance(claims, list):
            claims = []
        return max(0.0, min(1.0, frac)), claims
    except Exception:
        return 0.0, []


def hallucination_rate_from_judge(supported_fraction: float, unsupported_claims: List[str]) -> float:
    """1 - supported_fraction, or 1 if any unsupported claim."""
    if unsupported_claims:
        return 1.0
    return 1.0 - supported_fraction


# --- Answer relevance (semantic or judge) ---

def answer_relevance_similarity(
    query_embedding: List[float], answer_embedding: List[float]
) -> float:
    """Cosine similarity between query and answer embeddings."""
    return _cosine_sim(query_embedding, answer_embedding)


def parse_llm_judge_relevance(raw: str) -> float:
    try:
        s = raw.strip()
        for start in ("```json", "```"):
            if s.startswith(start):
                s = s[len(start):].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
        obj = json.loads(s)
        return max(0.0, min(1.0, float(obj.get("relevance_score", 0.0))))
    except Exception:
        return 0.0


# --- Attribution [n] ---

def extract_citations(text: str) -> List[int]:
    """Return list of cited indices (1-based), e.g. [1], [2] -> [1, 2]."""
    indices = []
    for m in re.finditer(r"\[(\d+)\]", text):
        indices.append(int(m.group(1)))
    return indices


def attribution_precision_recall(
    answer: str, context_chunks: List[str]
) -> Tuple[float, float]:
    """
    Precision: fraction of citations [n] that are valid (1-based index in range and claim supported).
    Recall: fraction of context chunks that were cited and used.
    Simplified: precision = valid_citations / total_citations, recall = cited_chunks / len(context).
    """
    cited = extract_citations(answer)
    n_chunks = len(context_chunks)
    if not cited:
        return 0.0, 0.0
    valid = 0
    for n in cited:
        if 1 <= n <= n_chunks:
            valid += 1
    precision = valid / len(cited) if cited else 0.0
    unique_valid = len(set(n for n in cited if 1 <= n <= n_chunks))
    recall = unique_valid / n_chunks if n_chunks else 0.0
    return precision, recall


def context_utilization(answer: str, context: str) -> float:
    """ROUGE-L-like overlap: longest common subsequence ratio (simplified word overlap)."""
    a_words = _normalize(answer).split()
    c_words = _normalize(context).split()
    if not a_words:
        return 0.0
    # fraction of answer words that appear in context
    c_set = set(c_words)
    matches = sum(1 for w in a_words if w in c_set)
    return matches / len(a_words)


def conciseness_score(answer: str, max_target_length: int = 300) -> float:
    """Brevity: 1 if length <= max_target_length, else decay."""
    length = len(answer.split())
    if length <= 0:
        return 0.0
    if length <= max_target_length:
        return 1.0
    return max(0.0, 1.0 - (length - max_target_length) / (2 * max_target_length))


# --- End-to-end ---

def exact_match(predicted: str, ground_truth: str) -> float:
    return 1.0 if _normalize(predicted.strip()) == _normalize(ground_truth.strip()) else 0.0


def _token_f1(pred_tokens: set, gold_tokens: set) -> float:
    if not gold_tokens:
        return 1.0
    prec = len(pred_tokens & gold_tokens) / len(pred_tokens) if pred_tokens else 0.0
    rec = len(pred_tokens & gold_tokens) / len(gold_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def f1_score(predicted: str, ground_truth: str) -> float:
    """Token-level F1."""
    p = set(_normalize(predicted).split())
    g = set(_normalize(ground_truth).split())
    return _token_f1(p, g)


def nugget_f1(predicted: str, nuggets: List[str]) -> float:
    """Nugget-level F1: each nugget is a required fact; match if nugget in predicted."""
    if not nuggets:
        return 1.0
    pred_n = _normalize(predicted)
    matched = sum(1 for n in nuggets if _normalize(n) in pred_n or _jaccard_similarity(_normalize(n), pred_n) > 0.5)
    rec = matched / len(nuggets)
    # precision: nuggets that appear / total "nugget-like" segments? Use recall as proxy when nuggets are gold facts.
    return rec  # simplified: nugget recall as F1 proxy when precision not defined


# --- System ---

def latency_percentiles(latencies_seconds: List[float]) -> Dict[str, float]:
    if not latencies_seconds:
        return {"p50": 0.0, "p95": 0.0}
    sorted_l = sorted(latencies_seconds)
    n = len(sorted_l)
    p50 = sorted_l[min(int(0.5 * n), n - 1)]
    p95 = sorted_l[min(int(0.95 * n), n - 1)]
    return {"p50": p50, "p95": p95}


def estimate_tokens(text: str) -> int:
    """Approximate token count (chars/4)."""
    return max(0, len(text) // 4)


@dataclass
class EvalRunMetadata:
    model: str = ""
    temperature: float = 0.0
    top_k: int = 5
    reranker_on: bool = False


@dataclass
class RetrievalMetrics:
    recall_at_k: float = 0.0
    mrr_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    coverage: float = 0.0
    redundancy: float = 0.0


@dataclass
class GenerationMetrics:
    faithfulness_nli: float = 0.0
    faithfulness_llm: float = 0.0
    hallucination_rate: float = 0.0
    answer_relevance: float = 0.0
    attribution_precision: float = 0.0
    attribution_recall: float = 0.0
    context_utilization: float = 0.0
    conciseness: float = 0.0


@dataclass
class E2EMetrics:
    exact_match: float = 0.0
    f1: float = 0.0
    nugget_f1: float = 0.0


@dataclass
class SystemMetrics:
    latency_retrieve_p50: float = 0.0
    latency_retrieve_p95: float = 0.0
    latency_generate_p50: float = 0.0
    latency_generate_p95: float = 0.0
    tokens_in_total: int = 0
    tokens_out_total: int = 0
    cost_estimate: float = 0.0  # optional


def aggregate_retrieval(
    per_query: List[Tuple[List[str], List[str], List[List[float]], int]]
) -> RetrievalMetrics:
    """per_query: (retrieved_texts, gold_passages, retrieved_embeddings, k)."""
    k = per_query[0][3] if per_query else 5
    recalls = [recall_at_k(r, g, k) for r, g, _, _ in per_query]
    mrrs = [mrr_at_k(r, g, k) for r, g, _, _ in per_query]
    ndcgs = [ndcg_at_k(r, g, k) for r, g, _, _ in per_query]
    coverages = [coverage(r, g) for r, g, _, _ in per_query]
    reds = [redundancy(embs) for _, _, embs, _ in per_query]
    return RetrievalMetrics(
        recall_at_k=statistics.mean(recalls) if recalls else 0.0,
        mrr_at_k=statistics.mean(mrrs) if mrrs else 0.0,
        ndcg_at_k=statistics.mean(ndcgs) if ndcgs else 0.0,
        coverage=statistics.mean(coverages) if coverages else 0.0,
        redundancy=statistics.mean(reds) if reds else 0.0,
    )
