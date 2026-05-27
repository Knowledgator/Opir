# Evaluation Data

Place local evaluation files in this directory when running the harness offline.

The evaluator can also load several public datasets directly from Hugging Face. The Opir benchmark bundle referenced by the paper is available as:

```text
Ihor/OpirSafetyBench
```

Expected local filenames include:

- `oai.jsonl`
- `aegis.json`
- `test.csv`
- `harmbench_contextual_train.jsonl`
- `saferlhf_test.jsonl`
- `beavertails_330k_test.jsonl`
- `xstest_test.jsonl`
- `wildguardmix_prompt_safety.jsonl`
- `wildguardmix_response_safety.jsonl`
- `wildguardmix_response_refusal.jsonl`
- `wildguardmix_prompt_subcategory.jsonl`
- `gliclass_safe_unsafe_classification.jsonl`
- `gliclass_toxicity_classification.jsonl`
- `gliclass_jailbreaking_classification.jsonl`
- `gliclass_unsafe_categorization.jsonl`
