# RAG Evaluation

Quick reference for the RAG evaluation layer. See the main [README.md](README.md#-rag-evaluation) for full details.

## One-command run

From project root:

```bash
# NLI-based faithfulness (no extra LLM calls for judging)
python -m eval.evaluator --data eval/datasets/eval.jsonl --k 5 --judge nli --out eval/reports/run_YYYYMMDD_HHMM/

# LLM-as-judge faithfulness (uses judge prompts)
python -m eval.evaluator --data eval/datasets/eval.jsonl --k 5 --judge llm --out eval/reports/run_YYYYMMDD_HHMM/
```

Replace `run_YYYYMMDD_HHMM` with a timestamp or label. Reports are written as `report.json`, `report.csv`, `report.md`, and `report.html` in the output directory.

## Eval dataset

Edit `eval/datasets/eval.jsonl`: one JSON object per line with at least:

- `query` (required)
- `ground_truth` (optional, for exact match / F1)
- `gold_passages` (optional, list of strings for retrieval metrics)
- `nuggets` (optional, list of strings for nugget F1)

## Optional: latency logging in the main pipeline

Set in `.env`:

```
EVAL_LOGGING_ENABLED=true
```

Then retrieve and generate latencies are logged (e.g. `eval_latency_retrieve_seconds=...`, `eval_latency_generate_seconds=...`).
