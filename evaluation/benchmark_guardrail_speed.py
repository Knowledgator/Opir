#!/usr/bin/env python3
"""
Benchmark binary guardrail/classification model speed.

Measures:
  1. Throughput at a fixed batch size over different synthetic sequence lengths.
  2. Latency at batch size 1 over the same sequence lengths.

Examples:
  python evaluation/benchmark_guardrail_speed.py \
    --gliclass-models models_post/final_model \
    --gliner2-models fastino/gliner2-base-v1 \
    --vllm-models nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3

  python evaluation/benchmark_guardrail_speed.py \
    --fixed-batch-size 16 \
    --text-lengths 64 256 512 1024 \
    --n-measure 20 \
    --latency-runs 50
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
try:
    from .test_guardrails_vllm import GUARD_TEMPLATES, build_guard_prompt, detect_guard_template
except ImportError:
    from test_guardrails_vllm import GUARD_TEMPLATES, build_guard_prompt, detect_guard_template
from transformers import AutoTokenizer


DEFAULT_TEXT_LENGTHS = [64, 256, 512, 1024]
DEFAULT_LABELS = ["safe", "unsafe"]
WORDS = (
    "the quick brown fox jumps over the lazy dog technology science health sports "
    "politics business culture education travel food music art research development "
    "innovation data analysis machine learning neural network deep model training "
    "inference classification detection recognition language processing understanding "
    "generation translation computer vision image video audio speech climate environment "
    "energy sustainability economy finance market medical biology chemistry physics "
    "mathematics history philosophy psychology design architecture engineering public "
    "private social political human natural artificial positive negative fast slow "
    "large small new old modern water fire earth air space time city country people "
    "organization company government system process method result output input feature "
    "layer weight gradient loss function parameter network structure algorithm sequence "
    "pattern representation embedding safety policy moderation benign harmful request"
).split()


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def make_text_for_tokens(tokenizer, n_tokens: int, seed: int) -> str:
    rng = random.Random(seed + n_tokens)
    text = " ".join(rng.choices(WORDS, k=max(n_tokens * 3, 1)))
    ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), q))


@dataclass
class BenchResult:
    model: str
    backend: str
    sequence_length: int
    actual_input_tokens: int
    fixed_batch_size: int
    throughput_samples_per_second: float
    throughput_seconds: float
    throughput_samples: int
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float


class Runner:
    def __init__(self, model_name: str, backend: str) -> None:
        self.model_name = model_name
        self.backend = backend
        self.tokenizer = None

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, texts: List[str], batch_size: int) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        self.tokenizer = None

    def count_input_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))


class GLiClassRunner(Runner):
    def __init__(self, model_name: str, device: str, labels: List[str], compile_model: bool, token: Optional[str]) -> None:
        super().__init__(model_name, "gliclass")
        self.device = device
        self.labels = labels
        self.compile_model = compile_model
        self.token = token
        self.pipeline = None

    def load(self) -> None:
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.token,
            add_prefix_space=True,
        )
        model = GLiClassModel.from_pretrained(self.model_name, token=self.token).to(self.device)
        if self.device.startswith("cuda"):
            model = model.to(dtype=torch.float16)
        if self.compile_model:
            model = torch.compile(model)
        self.pipeline = ZeroShotClassificationPipeline(
            model,
            self.tokenizer,
            classification_type="multi-label",
            device=self.device,
            progress_bar=False,
        )

    def predict(self, texts: List[str], batch_size: int) -> None:
        self.pipeline(texts, self.labels, batch_size=batch_size)

    def unload(self) -> None:
        self.pipeline = None
        super().unload()


class GLiNER2Runner(Runner):
    def __init__(self, model_name: str, device: str, labels: List[str], compile_model: bool) -> None:
        super().__init__(model_name, "gliner2")
        self.device = device
        self.labels = labels
        self.compile_model = compile_model
        self.model = None
        self.schema = None

    def load(self) -> None:
        from gliner2 import GLiNER2

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
        self.model = GLiNER2.from_pretrained(self.model_name).to(self.device)
        if self.compile_model and hasattr(self.model, "compile"):
            self.model.compile()
        if hasattr(self.model, "create_schema"):
            self.schema = self.model.create_schema().classification("label", self.labels)

    def predict(self, texts: List[str], batch_size: int) -> None:
        if hasattr(self.model, "batch_classify_text"):
            self.model.batch_classify_text(texts, {"category": self.labels}, batch_size=batch_size)
            return
        for text in texts:
            self.model.extract(text, self.schema)

    def unload(self) -> None:
        self.model = None
        self.schema = None
        super().unload()


class VLLMRunner(Runner):
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        dtype: str,
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
        max_new_tokens: int,
        trust_remote_code: bool,
        use_chat_template: bool,
        guard_template: str,
        token: Optional[str],
    ) -> None:
        super().__init__(model_name, "vllm")
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self.use_chat_template = use_chat_template
        self.guard_template = (
            detect_guard_template(model_name)
            if guard_template == "auto"
            else guard_template
        )
        self.token = token
        self.llm = None
        self.sampling_params = None

    def load(self) -> None:
        from vllm import LLM, SamplingParams

        kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        if self.token:
            kwargs["download_dir"] = None
        print(f"Using guard prompt template: {self.guard_template}")
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_new_tokens,
        )

    def predict(self, texts: List[str], batch_size: int) -> None:
        prompts = [
            self.build_prompt(text)
            for text in texts
        ]
        for start in range(0, len(prompts), batch_size):
            self.llm.generate(prompts[start : start + batch_size], self.sampling_params)

    def build_prompt(self, text: str) -> str:
        return build_guard_prompt(
            self.tokenizer,
            query=text,
            response=None,
            use_chat_template=self.use_chat_template,
            guard_template=self.guard_template,
            wants_response=False,
        )

    def count_input_tokens(self, text: str) -> int:
        prompt = self.build_prompt(text)
        return len(self.tokenizer.encode(prompt, add_special_tokens=True))

    def unload(self) -> None:
        llm = self.llm
        self.llm = None
        self.sampling_params = None
        super().unload()

        if llm is None:
            return

        for owner in (llm, getattr(llm, "llm_engine", None)):
            shutdown = getattr(owner, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass

        llm_engine = getattr(llm, "llm_engine", None)
        model_executor = getattr(llm_engine, "model_executor", None)
        shutdown = getattr(model_executor, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass

        del llm

        try:
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
        except Exception:
            pass

        try:
            from vllm.distributed.parallel_state import destroy_distributed_environment

            destroy_distributed_environment()
        except Exception:
            pass

        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass


def time_call(fn: Callable[[], None]) -> float:
    sync()
    start = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - start


def benchmark_runner(
    runner: Runner,
    text_lengths: List[int],
    fixed_batch_size: int,
    n_measure: int,
    warmup_runs: int,
    latency_runs: int,
    seed: int,
) -> List[BenchResult]:
    print(f"\n{'=' * 88}")
    print(f"{runner.model_name} [{runner.backend}]")
    print(f"{'=' * 88}")

    runner.load()
    texts_by_length = {
        length: make_text_for_tokens(runner.tokenizer, length, seed)
        for length in text_lengths
    }

    warm_text = texts_by_length[text_lengths[-1]]
    print(f"warming up {warmup_runs}x at batch_size={fixed_batch_size}")
    for _ in range(warmup_runs):
        runner.predict([warm_text] * fixed_batch_size, fixed_batch_size)
    sync()

    results = []
    print(
        f"{'seq':>8} {'actual_tok':>11} {'throughput':>14} "
        f"{'lat_mean':>12} {'lat_p50':>10} {'lat_p95':>10}"
    )
    print("-" * 75)

    for length in text_lengths:
        text = texts_by_length[length]
        actual_tokens = runner.count_input_tokens(text)
        throughput_texts = [text] * (fixed_batch_size * n_measure)

        # torch.compile and some backend kernels specialize on input shape. Warm
        # up the exact shapes we measure so one-time compile/setup cost does not
        # pollute throughput or latency percentiles for individual seq lengths.
        for _ in range(warmup_runs):
            runner.predict([text] * fixed_batch_size, fixed_batch_size)
        runner.predict([text], 1)
        sync()

        elapsed = time_call(lambda: runner.predict(throughput_texts, fixed_batch_size))
        throughput = len(throughput_texts) / elapsed

        latencies = []
        for _ in range(latency_runs):
            latencies.append(time_call(lambda: runner.predict([text], 1)) * 1000.0)

        result = BenchResult(
            model=runner.model_name,
            backend=runner.backend,
            sequence_length=length,
            actual_input_tokens=actual_tokens,
            fixed_batch_size=fixed_batch_size,
            throughput_samples_per_second=float(throughput),
            throughput_seconds=float(elapsed),
            throughput_samples=len(throughput_texts),
            latency_mean_ms=float(statistics.mean(latencies)),
            latency_p50_ms=percentile(latencies, 50),
            latency_p95_ms=percentile(latencies, 95),
        )
        results.append(result)
        print(
            f"{length:>8} {actual_tokens:>11} {throughput:>12.2f}/s "
            f"{result.latency_mean_ms:>10.2f}ms "
            f"{result.latency_p50_ms:>8.2f}ms "
            f"{result.latency_p95_ms:>8.2f}ms"
        )

    return results


def make_runners(args: argparse.Namespace) -> List[Runner]:
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    labels = args.labels
    runners: List[Runner] = []
    for model_name in args.gliclass_models:
        runners.append(GLiClassRunner(model_name, device, labels, not args.no_compile, args.token))
    for model_name in args.gliner2_models:
        runners.append(GLiNER2Runner(model_name, device, labels, not args.no_compile))
    for model_name in args.vllm_models:
        runners.append(
            VLLMRunner(
                model_name=model_name,
                tensor_parallel_size=args.tensor_parallel_size,
                dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_new_tokens=args.max_new_tokens,
                trust_remote_code=args.trust_remote_code,
                use_chat_template=not args.no_chat_template,
                guard_template=args.guard_template,
                token=args.token,
            )
        )
    return runners


def print_summary(results: List[BenchResult]) -> None:
    if not results:
        return
    print(f"\n{'=' * 88}")
    print("SUMMARY")
    print(f"{'model':<42} {'backend':<10} {'seq':>7} {'samples/s':>12} {'p50 ms':>10} {'p95 ms':>10}")
    print("-" * 88)
    for result in results:
        short = result.model.split("/")[-1]
        print(
            f"{short:<42} {result.backend:<10} {result.sequence_length:>7} "
            f"{result.throughput_samples_per_second:>12.2f} "
            f"{result.latency_p50_ms:>10.2f} {result.latency_p95_ms:>10.2f}"
        )


def save_json(path: Path, results: List[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump([result.__dict__ for result in results], file, indent=2)
    print(f"\nWrote results to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GLiClass, GLiNER2, and vLLM binary classification speed."
    )
    parser.add_argument("--gliclass-models", nargs="*", default=[])
    parser.add_argument("--gliner2-models", nargs="*", default=[])
    parser.add_argument("--vllm-models", nargs="*", default=[])
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--text-lengths", nargs="+", type=int, default=DEFAULT_TEXT_LENGTHS)
    parser.add_argument("--fixed-batch-size", type=int, default=8)
    parser.add_argument(
        "--n-measure",
        type=int,
        default=20,
        help="Number of fixed-size batches used for throughput measurement.",
    )
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--latency-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--guard-template",
        choices=GUARD_TEMPLATES,
        default="auto",
        help=(
            "Prompt template family for vLLM guard models. 'auto' selects from "
            "the model name, matching test_guardrails_vllm.py."
        ),
    )
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    runners = make_runners(args)
    if not runners:
        raise SystemExit(
            "No models requested. Pass at least one of --gliclass-models, "
            "--gliner2-models, or --vllm-models."
        )

    print(f"Device: {args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Fixed throughput batch size: {args.fixed_batch_size}")
    print(f"Text lengths: {args.text_lengths}")
    print(f"Labels: {args.labels}")

    all_results: List[BenchResult] = []
    for runner in runners:
        try:
            all_results.extend(
                benchmark_runner(
                    runner,
                    text_lengths=args.text_lengths,
                    fixed_batch_size=args.fixed_batch_size,
                    n_measure=args.n_measure,
                    warmup_runs=args.warmup_runs,
                    latency_runs=args.latency_runs,
                    seed=args.seed,
                )
            )
        except Exception as exc:
            print(f"\nERROR benchmarking {runner.model_name} [{runner.backend}]: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            runner.unload()
            cleanup()

    print_summary(all_results)
    if args.output_json is not None:
        save_json(args.output_json, all_results)


if __name__ == "__main__":
    main()
