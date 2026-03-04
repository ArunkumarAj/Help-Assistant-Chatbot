"""
RAG evaluation CLI: run retrieval + generation on an eval dataset and write reports.

Usage:
  python -m eval.evaluator --data eval/datasets/eval.jsonl --k 5 --judge nli --out eval/reports/run_YYYYMMDD_HHMM/
  python -m eval.evaluator --data eval/datasets/eval.jsonl --k 5 --judge llm --out eval/reports/run_YYYYMMDD_HHMM/

Outputs: report.json, report.csv, report.md, report.html in the given --out directory.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path when run as python -m eval.evaluator
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.metrics import (
    E2EMetrics,
    EvalRunMetadata,
    GenerationMetrics,
    RetrievalMetrics,
    SystemMetrics,
    aggregate_retrieval,
    answer_relevance_similarity,
    attribution_precision_recall,
    conciseness_score,
    context_utilization,
    exact_match,
    faithfulness_nli,
    f1_score,
    hallucination_rate_from_judge,
    latency_percentiles,
    nugget_f1,
    parse_llm_judge_faithfulness,
    parse_llm_judge_relevance,
    estimate_tokens,
)
from embedding.model import get_embedding_model
from llm.client import get_llm
from services.rag import eval_retrieve_and_build_prompt


# -----------------------------------------------------------------------------
# Data and judge prompts
# -----------------------------------------------------------------------------


def load_eval_data(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_judge_prompt(judge_name: str) -> Dict[str, str]:
    base = Path(__file__).resolve().parent / "judge_prompts"
    path = base / f"{judge_name}.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_llm_judge_faithfulness(context: str, answer: str) -> str:
    t = load_judge_prompt("faithfulness")
    user = (t.get("user") or "").format(context=context[:6000], answer=answer[:3000])
    system = t.get("system") or "Output only valid JSON."
    llm = get_llm(temperature=0.0, max_tokens=1024)
    prompt = f"{system}\n\n{user}"
    return llm.invoke(prompt) or "{}"


def run_llm_judge_relevance(query: str, answer: str) -> str:
    t = load_judge_prompt("relevance")
    user = (t.get("user") or "").format(query=query, answer=answer)
    system = t.get("system") or "Output only valid JSON."
    llm = get_llm(temperature=0.0, max_tokens=256)
    prompt = f"{system}\n\n{user}"
    return llm.invoke(prompt) or "{}"


def run_eval_sync(
    data_path: str,
    top_k: int,
    judge: str,
    out_dir: Path,
) -> Dict[str, Any]:
    rows = load_eval_data(data_path)
    if not rows:
        raise SystemExit("No rows in eval dataset.")

    emb_model = get_embedding_model()
    llm = get_llm(temperature=0.0, top_p=0.9, max_tokens=2000)

    # Per-query data for aggregation
    retrieval_tuples: List[tuple] = []
    lat_retrieve: List[float] = []
    lat_generate: List[float] = []
    tokens_in: List[int] = []
    tokens_out: List[int] = []

    per_query: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        query = row.get("query", "")
        ground_truth = row.get("ground_truth", "")
        gold_passages = row.get("gold_passages") or []
        nuggets = row.get("nuggets") or []

        # Retrieve + build prompt
        start_retrieve = time.perf_counter()
        results, context, prompt = eval_retrieve_and_build_prompt(query, top_k)
        lat_retrieve.append(time.perf_counter() - start_retrieve)

        retrieved_texts = [r["_source"]["text"] for r in results]
        # Embeddings for redundancy
        if results:
            chunk_embs = [emb_model.encode(t).tolist() for t in retrieved_texts]
        else:
            chunk_embs = []

        retrieval_tuples.append((retrieved_texts, gold_passages, chunk_embs, top_k))

        # Generate
        start_generate = time.perf_counter()
        answer = llm.invoke(prompt) or ""
        lat_generate.append(time.perf_counter() - start_generate)

        tokens_in.append(estimate_tokens(prompt))
        tokens_out.append(estimate_tokens(answer))

        # Generation metrics
        ctx_concatenated = "\n\n".join(retrieved_texts)
        faith_nli = faithfulness_nli(ctx_concatenated, answer) if judge == "nli" else 0.0
        faith_llm = 0.0
        unsupported: List[str] = []
        if judge == "llm":
            raw = run_llm_judge_faithfulness(ctx_concatenated, answer)
            faith_llm, unsupported = parse_llm_judge_faithfulness(raw)
            if faith_nli == 0.0:
                faith_nli = faith_llm  # use LLM as fallback for display
            hall = hallucination_rate_from_judge(faith_llm, unsupported)
        else:
            hall = 1.0 - faith_nli  # NLI: hallucination = 1 - faithfulness

        q_emb = emb_model.encode(query).tolist()
        a_emb = emb_model.encode(answer).tolist()
        ans_rel = answer_relevance_similarity(q_emb, a_emb)
        if judge == "llm":
            raw_rel = run_llm_judge_relevance(query, answer)
            ans_rel_llm = parse_llm_judge_relevance(raw_rel)
            ans_rel = max(ans_rel, ans_rel_llm)

        att_prec, att_rec = attribution_precision_recall(answer, retrieved_texts)
        ctx_util = context_utilization(answer, ctx_concatenated)
        conc = conciseness_score(answer)

        # E2E
        em = exact_match(answer, ground_truth) if ground_truth else 0.0
        f1 = f1_score(answer, ground_truth) if ground_truth else 0.0
        nf1 = nugget_f1(answer, nuggets) if nuggets else 0.0

        per_query.append({
            "query": query,
            "answer": answer,
            "ground_truth": ground_truth,
            "retrieved_texts": retrieved_texts,
            "faithfulness_nli": faith_nli,
            "faithfulness_llm": faith_llm,
            "hallucination_rate": hall,
            "answer_relevance": ans_rel,
            "attribution_precision": att_prec,
            "attribution_recall": att_rec,
            "context_utilization": ctx_util,
            "conciseness": conc,
            "exact_match": em,
            "f1": f1,
            "nugget_f1": nf1,
            "latency_retrieve": lat_retrieve[-1],
            "latency_generate": lat_generate[-1],
            "tokens_in": tokens_in[-1],
            "tokens_out": tokens_out[-1],
        })

    # Aggregate retrieval
    ret_metrics = aggregate_retrieval(retrieval_tuples)

    # Aggregate generation / E2E (averages)
    n = len(per_query)
    gen = GenerationMetrics(
        faithfulness_nli=sum(p["faithfulness_nli"] for p in per_query) / n,
        faithfulness_llm=sum(p["faithfulness_llm"] for p in per_query) / n,
        hallucination_rate=sum(p["hallucination_rate"] for p in per_query) / n,
        answer_relevance=sum(p["answer_relevance"] for p in per_query) / n,
        attribution_precision=sum(p["attribution_precision"] for p in per_query) / n,
        attribution_recall=sum(p["attribution_recall"] for p in per_query) / n,
        context_utilization=sum(p["context_utilization"] for p in per_query) / n,
        conciseness=sum(p["conciseness"] for p in per_query) / n,
    )
    e2e = E2EMetrics(
        exact_match=sum(p["exact_match"] for p in per_query) / n,
        f1=sum(p["f1"] for p in per_query) / n,
        nugget_f1=sum(p["nugget_f1"] for p in per_query) / n,
    )
    lat_ret_p = latency_percentiles(lat_retrieve)
    lat_gen_p = latency_percentiles(lat_generate)
    sys_metrics = SystemMetrics(
        latency_retrieve_p50=lat_ret_p["p50"],
        latency_retrieve_p95=lat_ret_p["p95"],
        latency_generate_p50=lat_gen_p["p50"],
        latency_generate_p95=lat_gen_p["p95"],
        tokens_in_total=sum(tokens_in),
        tokens_out_total=sum(tokens_out),
    )

    run_meta = EvalRunMetadata(top_k=top_k, temperature=0.0, reranker_on=False)

    report = {
        "metadata": {
            "run_at": datetime.utcnow().isoformat() + "Z",
            "data_path": data_path,
            "top_k": top_k,
            "judge": judge,
            "n_queries": n,
            "model": getattr(get_llm(), "model", "unknown"),
        },
        "retrieval": {
            "recall_at_k": ret_metrics.recall_at_k,
            "mrr_at_k": ret_metrics.mrr_at_k,
            "ndcg_at_k": ret_metrics.ndcg_at_k,
            "coverage": ret_metrics.coverage,
            "redundancy": ret_metrics.redundancy,
        },
        "generation": {
            "faithfulness_nli": gen.faithfulness_nli,
            "faithfulness_llm": gen.faithfulness_llm,
            "hallucination_rate": gen.hallucination_rate,
            "answer_relevance": gen.answer_relevance,
            "attribution_precision": gen.attribution_precision,
            "attribution_recall": gen.attribution_recall,
            "context_utilization": gen.context_utilization,
            "conciseness": gen.conciseness,
        },
        "e2e": {
            "exact_match": e2e.exact_match,
            "f1": e2e.f1,
            "nugget_f1": e2e.nugget_f1,
        },
        "system": {
            "latency_retrieve_p50_s": sys_metrics.latency_retrieve_p50,
            "latency_retrieve_p95_s": sys_metrics.latency_retrieve_p95,
            "latency_generate_p50_s": sys_metrics.latency_generate_p50,
            "latency_generate_p95_s": sys_metrics.latency_generate_p95,
            "tokens_in_total": sys_metrics.tokens_in_total,
            "tokens_out_total": sys_metrics.tokens_out_total,
        },
        "per_query": per_query,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # CSV (summary row + per-query)
    import csv
    with open(out_dir / "report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["recall_at_k", ret_metrics.recall_at_k])
        w.writerow(["mrr_at_k", ret_metrics.mrr_at_k])
        w.writerow(["ndcg_at_k", ret_metrics.ndcg_at_k])
        w.writerow(["coverage", ret_metrics.coverage])
        w.writerow(["redundancy", ret_metrics.redundancy])
        w.writerow(["faithfulness_nli", gen.faithfulness_nli])
        w.writerow(["faithfulness_llm", gen.faithfulness_llm])
        w.writerow(["hallucination_rate", gen.hallucination_rate])
        w.writerow(["answer_relevance", gen.answer_relevance])
        w.writerow(["attribution_precision", gen.attribution_precision])
        w.writerow(["attribution_recall", gen.attribution_recall])
        w.writerow(["context_utilization", gen.context_utilization])
        w.writerow(["conciseness", gen.conciseness])
        w.writerow(["exact_match", e2e.exact_match])
        w.writerow(["f1", e2e.f1])
        w.writerow(["nugget_f1", e2e.nugget_f1])
        w.writerow(["latency_retrieve_p50_s", sys_metrics.latency_retrieve_p50])
        w.writerow(["latency_retrieve_p95_s", sys_metrics.latency_retrieve_p95])
        w.writerow(["latency_generate_p50_s", sys_metrics.latency_generate_p50])
        w.writerow(["latency_generate_p95_s", sys_metrics.latency_generate_p95])
        w.writerow(["tokens_in_total", sys_metrics.tokens_in_total])
        w.writerow(["tokens_out_total", sys_metrics.tokens_out_total])

    # Markdown
    md = _render_markdown(report)
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(md)

    # HTML (minimal dashboard)
    html = _render_html(report)
    with open(out_dir / "report.html", "w", encoding="utf-8") as f:
        f.write(html)

    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    """Build a markdown summary of the evaluation report."""
    retrieval = report.get("retrieval", {})
    generation = report.get("generation", {})
    e2e = report.get("e2e", {})
    system = report.get("system", {})
    metadata = report.get("metadata", {})
    lines = [
        "# RAG Evaluation Report",
        "",
        f"**Run:** {metadata.get('run_at', '')} | **Queries:** {metadata.get('n_queries', 0)} | **Judge:** {metadata.get('judge', '')} | **top_k:** {metadata.get('top_k', 0)}",
        "",
        "## Retrieval",
        f"- Recall@k: {retrieval.get('recall_at_k', 0):.4f}",
        f"- MRR@k: {retrieval.get('mrr_at_k', 0):.4f}",
        f"- nDCG@k: {retrieval.get('ndcg_at_k', 0):.4f}",
        f"- Coverage: {retrieval.get('coverage', 0):.4f}",
        f"- Redundancy: {retrieval.get('redundancy', 0):.4f}",
        "",
        "## Generation",
        f"- Faithfulness (NLI): {generation.get('faithfulness_nli', 0):.4f}",
        f"- Faithfulness (LLM): {generation.get('faithfulness_llm', 0):.4f}",
        f"- Hallucination rate: {generation.get('hallucination_rate', 0):.4f}",
        f"- Answer relevance: {generation.get('answer_relevance', 0):.4f}",
        f"- Attribution precision: {generation.get('attribution_precision', 0):.4f}",
        f"- Attribution recall: {generation.get('attribution_recall', 0):.4f}",
        f"- Context utilization: {generation.get('context_utilization', 0):.4f}",
        f"- Conciseness: {generation.get('conciseness', 0):.4f}",
        "",
        "## End-to-End",
        f"- Exact match: {e2e.get('exact_match', 0):.4f}",
        f"- F1: {e2e.get('f1', 0):.4f}",
        f"- Nugget F1: {e2e.get('nugget_f1', 0):.4f}",
        "",
        "## System",
        f"- Latency retrieve (p50/p95 s): {system.get('latency_retrieve_p50_s', 0):.4f} / {system.get('latency_retrieve_p95_s', 0):.4f}",
        f"- Latency generate (p50/p95 s): {system.get('latency_generate_p50_s', 0):.4f} / {system.get('latency_generate_p95_s', 0):.4f}",
        f"- Tokens in/out: {system.get('tokens_in_total', 0)} / {system.get('tokens_out_total', 0)}",
        "",
    ]
    return "\n".join(lines)


def _render_html(report: Dict[str, Any]) -> str:
    """Wrap the markdown report in a minimal HTML page."""
    import html
    markdown_body = _render_markdown(report)
    escaped = html.escape(markdown_body)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>RAG Eval Report</title></head><body><pre>{escaped}</pre></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Evaluation CLI")
    parser.add_argument("--data", required=True, help="Path to eval.jsonl")
    parser.add_argument("--k", type=int, default=5, help="top_k for retrieval")
    parser.add_argument("--judge", choices=["nli", "llm"], default="nli", help="Faithfulness judge: nli or llm")
    parser.add_argument("--out", required=True, help="Output directory (e.g. eval/reports/run_YYYYMMDD_HHMM)")
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        raise SystemExit(f"Eval data file not found: {args.data}")

    run_eval_sync(args.data, args.k, args.judge, Path(args.out))
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
