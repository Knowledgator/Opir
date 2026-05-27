# Evaluation Results

Raw evaluation reports:

- `eval_best.json`: Opir multitask large local checkpoint.
- `eval_multi.json`: Opir multitask multilingual local checkpoint.
- `eval_edge_en.json`: Opir edge English local checkpoint.
- `eval_edge_multi.json`: Opir edge multilingual local checkpoint.
- `eval_gliclass_best.json`: expanded GLiClass/Opir evaluation run.
- `eval_gliner2.json`: GLiNER2 baseline run.
- `eval_gliner_opir.json`: GLiNER-family Opir comparison run.
- `eval_wildguard.json`: WildGuard vLLM run.
- `eval_nemotron.json`: Nemotron Safety Guard raw vLLM run.
- `eval_polyguard.json`: PolyGuard-Qwen raw vLLM run.
- `eval_polyguard_small.json`: PolyGuard-Qwen-Smol raw vLLM run.
- `eval_qwen3guard.json`: Qwen3Guard raw vLLM run.
- `eval_results.json`: latest default evaluation output from the original harness.

`evaluation_results.json` and `evaluation_results_summary.csv` contain one row per model/dataset pair with only `model`, `dataset`, `macro_f1`, and `micro_f1`.
