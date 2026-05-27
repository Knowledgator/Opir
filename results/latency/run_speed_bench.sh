  python benchmark_guardrail_speed.py \
    --vllm-models nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3 ToxicityPrompts/PolyGuard-Qwen Qwen/Qwen3Guard-Gen-8B ToxicityPrompts/PolyGuard-Qwen-Smol allenai/wildguard\
    --text-lengths 64 256 512 1024 \
    --fixed-batch-size 8 \
    --no-compile \

    # --gliclass-models models/best models/multi_best models/bi_en_best models/bi_multi_best \
    # --gliner2-models fastino/gliguard-LLMGuardrails-300M  hivetrace/gliner-guard-omni\