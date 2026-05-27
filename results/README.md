# Results

This directory contains released evaluation and latency/throughput artifacts for the Opir paper.

## Evaluation

`evaluation/` contains raw evaluation reports copied from the original guardrail evaluation runs. `evaluation_results.json` and `evaluation_results_summary.csv` are simplified summaries with one row per model/dataset pair and only macro/micro F1 metrics.

Some vLLM baseline files are preserved as raw run outputs even when they are concatenated JSON logs rather than a single strict JSON document.

## Latency

`latency/guardrail_latency_throughput.csv` and `latency/guardrail_latency_throughput.json` contain the reported throughput, p50 latency, and p95 latency measurements across sequence lengths 64, 256, 512, and 1024.

`latency/run_speed_bench.sh` records the command scaffold used for the vLLM speed runs.
