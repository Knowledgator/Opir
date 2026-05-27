# Evaluation Harness

The Opir evaluation harness supports three model families:

- GLiClass models through `evaluation/test_guardrails.py --backend gliclass`.
- GLiNER2 models through `evaluation/test_guardrails.py --backend gliner2`.
- Decoder guardrail models through `evaluation/test_guardrails_vllm.py`.

The harness reports multi-label micro, macro, and weighted F1. For binary safety views it normalizes labels to `safe` and `unsafe` and reports accuracy plus per-label precision, recall, F1, and support.

Local files under `data/eval/` are preferred when available. Otherwise, the loader falls back to public Hugging Face datasets for supported benchmark families.

Important datasets include OpenAI moderation-style safety/category data, Aegis prompt and response safety, SimpleSafetyTests, HarmBench, PKU-SafeRLHF, BeaverTails, XSTest, OR-Bench, ToxicChat, WildGuardMix, PolyGuardPrompts, and JBB-Behaviors.
