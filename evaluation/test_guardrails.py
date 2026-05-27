import argparse
import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_MODEL = "knowledgator/opir-multitask-large-v1.0"
DEFAULT_MAX_TEXT_CHARS = 8192
MODEL_BACKENDS = ["gliclass", "gliner2"]
ALL_DATASETS = [
    "oai",
    "oai_safety",
    "aegis_prompt_safety",
    "aegis_response_safety",
    "aegis_categories",
    "simplest",
    "simplesafetytests",
    "harmbench_prompts",
    "harmbench_responses",
    "saferlhf",
    "saferlhf_response_safety",
    "beavertails",
    "xstest",
    "offenseval_2020_safe_unsafe",
    "textdetox_multilingual_toxicity_safe_unsafe",
    "pan12_predator_conversation_safety",
    "med_safety_bench_safe_unsafe",
    "wildguard_prompt_safety",
    "wildguard_response_safety",
    "wildguard_response_refusal",
    "wildguard_prompt_subcategory",
    "polyguard_prompt_safety",
    "polyguard_response_safety",
    "polyguard_response_refusal",
    "polyguard_prompt_subcategory",
    "toxicchat_safe_unsafe_classification",
    "toxicchat_toxicity_classification",
    "toxicchat_jailbreaking_classification",
    "gliclass_safe_unsafe_classification",
    "gliclass_toxicity_classification",
    "gliclass_jailbreaking_classification",
    "gliclass_unsafe_categorization",
    "or_bench_80k",
    "or_bench_hard_1k",
    "or_bench_toxic",
    "jbb_behaviors_behavior",
    "jbb_behaviors_category",
    "jbb_behaviors_safety",
]

AEGIS_DATASETS = {
    "aegis_prompt_safety",
    "aegis_response_safety",
    "aegis_categories",
}

BINARY_SAFETY_DATASETS = {
    "oai_safety",
    "aegis_prompt_safety",
    "aegis_response_safety",
    "simplesafetytests",
    "saferlhf_response_safety",
    "beavertails",
    "xstest",
    "offenseval_2020_safe_unsafe",
    "textdetox_multilingual_toxicity_safe_unsafe",
    "pan12_predator_conversation_safety",
    "med_safety_bench_safe_unsafe",
    "wildguard_prompt_safety",
    "wildguard_response_safety",
    "polyguard_prompt_safety",
    "polyguard_response_safety",
    "toxicchat_safe_unsafe_classification",
    "gliclass_safe_unsafe_classification",
    "or_bench_80k",
    "or_bench_hard_1k",
    "or_bench_toxic",
    "jbb_behaviors_safety",
}

TOXICCHAT_DATASETS = {
    "toxicchat_safe_unsafe_classification",
    "toxicchat_toxicity_classification",
    "toxicchat_jailbreaking_classification",
}

GLICLASS_DATASETS = {
    "gliclass_safe_unsafe_classification",
    "gliclass_toxicity_classification",
    "gliclass_jailbreaking_classification",
    "gliclass_unsafe_categorization",
}

JBB_BEHAVIORS_DATASETS = {
    "jbb_behaviors_behavior",
    "jbb_behaviors_category",
    "jbb_behaviors_safety",
}

SINGLE_LABEL_DATASETS = (
    BINARY_SAFETY_DATASETS
    | TOXICCHAT_DATASETS
    | GLICLASS_DATASETS
    | JBB_BEHAVIORS_DATASETS
)

POLYGUARD_DATASETS = {
    "polyguard_prompt_safety",
    "polyguard_response_safety",
    "polyguard_response_refusal",
    "polyguard_prompt_subcategory",
}

WILDGUARD_DATASETS = {
    "wildguard_prompt_safety",
    "wildguard_response_safety",
    "wildguard_response_refusal",
    "wildguard_prompt_subcategory",
}

WILDGUARD_LOCAL_FILES = {
    "wildguard_prompt_safety": "wildguardmix_prompt_safety.jsonl",
    "wildguard_response_safety": "wildguardmix_response_safety.jsonl",
    "wildguard_response_refusal": "wildguardmix_response_refusal.jsonl",
    "wildguard_prompt_subcategory": "wildguardmix_prompt_subcategory.jsonl",
}

GLICLASS_LOCAL_FILES = {
    "gliclass_safe_unsafe_classification": "gliclass_safe_unsafe_classification.jsonl",
    "gliclass_toxicity_classification": "gliclass_toxicity_classification.jsonl",
    "gliclass_jailbreaking_classification": "gliclass_jailbreaking_classification.jsonl",
    "gliclass_unsafe_categorization": "gliclass_unsafe_categorization.jsonl",
}

SAFETY_CLASSES = ["safe", "unsafe"]
SAFE_LABELS = {"safe", "benign", "unharmful", "safe_and_benign"}
UNSAFE_LABELS = {"unsafe", "harmful"}
NON_SAFETY_LABELS = {"refusal", "compliance"}
OFFENSEVAL_2020_LANGUAGES = ["ar", "da", "en", "gr", "tr"]
OFFENSEVAL_2020_LOCAL_FILE = "offenseval_2020_safe_unsafe.jsonl"
TEXTDETOX_MULTILINGUAL_TOXICITY_LANGUAGES = [
    "en",
    "ru",
    "uk",
    "de",
    "es",
    "am",
    "zh",
    "ar",
    "hi",
    "it",
    "fr",
    "he",
    "hin",
    "tt",
    "ja",
]
TEXTDETOX_MULTILINGUAL_TOXICITY_LOCAL_FILE = (
    "textdetox_multilingual_toxicity_safe_unsafe.jsonl"
)
PAN12_PREDATOR_CONVERSATION_SAFETY_LOCAL_FILE = (
    "pan12_predator_conversation_safety.jsonl"
)
MED_SAFETY_BENCH_SAFE_UNSAFE_LOCAL_FILE = "med_safety_bench_safe_unsafe.jsonl"
PAN12_PREDATOR_CONVERSATION_SAFETY_SAMPLE_SIZE = 10_000
OR_BENCH_DATASETS = {
    "or_bench_80k": {
        "subset": "or-bench-80k",
        "label": "safe",
        "sample_size": 10_000,
    },
    "or_bench_hard_1k": {
        "subset": "or-bench-hard-1k",
        "label": "safe",
        "sample_size": None,
    },
    "or_bench_toxic": {
        "subset": "or-bench-toxic",
        "label": "unsafe",
        "sample_size": None,
    },
}


class GuardrailEvaluator:
    def __init__(
        self,
        model_name: str,
        token: Optional[str],
        device: str,
        threshold: float,
        batch_size: int,
        output_dir: Path,
        backend: str = "gliclass",
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.token = token
        self.device = device
        self.threshold = threshold
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.backend = backend
        self.verbose = verbose
        self.pipeline = None
        self.extractor = None
        self.model_info = self.build_model_info()

    def load_model(self) -> None:
        if self.backend == "gliclass":
            self.load_gliclass_model()
        elif self.backend == "gliner2":
            self.load_gliner2_model()
        else:
            raise ValueError(f"Unknown model backend: {self.backend}")

    def load_gliclass_model(self) -> None:
        import torch
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer

        print(f"Loading GLiClass model: {self.model_name}")
        model = GLiClassModel.from_pretrained(
            self.model_name,
            token=self.token,
        ).to(self.device)

        if self.device.startswith("cuda"):
            model = model.to(dtype=torch.float16)

        self.model_info = self.build_model_info(model)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.token,
            add_prefix_space=True,
        )

        self.pipeline = ZeroShotClassificationPipeline(
            model,
            tokenizer,
            classification_type="multi-label",
            device=self.device,
        )

    def load_gliner2_model(self) -> None:
        try:
            from gliner2 import GLiNER2
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: gliner2. Install it before running "
                "`test_gliclass.py --backend gliner2`."
            ) from exc

        print(f"Loading GLiNER2 model: {self.model_name}")
        self.extractor = GLiNER2.from_pretrained(self.model_name)
        self.extractor.to(self.device)
        self.model_info = self.build_model_info(self.extractor)

    def build_model_info(self, model=None) -> Dict:
        total_parameters = None
        trainable_parameters = None
        parameter_size_bytes = None

        parameter_source = self.find_parameter_source(model)
        if parameter_source is not None:
            total_parameters = 0
            trainable_parameters = 0
            parameter_size_bytes = 0
            for parameter in parameter_source.parameters():
                count = parameter.numel()
                total_parameters += count
                if parameter.requires_grad:
                    trainable_parameters += count
                parameter_size_bytes += count * parameter.element_size()

        parameter_size_mb = None
        if parameter_size_bytes is not None:
            parameter_size_mb = round(parameter_size_bytes / (1024 * 1024), 2)

        return {
            "backend": self.backend,
            "name": self.model_name,
            "device": self.device,
            "parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameters,
            "parameter_size_bytes": parameter_size_bytes,
            "parameter_size_mb": parameter_size_mb,
        }

    @staticmethod
    def find_parameter_source(model):
        if model is None:
            return None
        if hasattr(model, "parameters"):
            return model
        for attribute in ("model", "base_model", "encoder"):
            nested = getattr(model, attribute, None)
            if nested is not None and hasattr(nested, "parameters"):
                return nested
        return None

    @staticmethod
    def load_jsonl(path: Path) -> List[Dict]:
        with path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    @staticmethod
    def load_json(path: Path) -> List[Dict]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def load_csv(path: Path) -> List[Dict]:
        with path.open("r", newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def load_hf(
        self,
        path: str,
        subset: Optional[str] = None,
        split: Optional[str] = None,
    ):
        from datasets import load_dataset

        if split is None:
            raise ValueError("split must be specified for load_hf")
        dataset = load_dataset(path, subset, token=self.token)
        return [dict(item) for item in dataset[split]]

    @staticmethod
    def process_oai_dataset(items: Iterable[Dict]) -> Dict[str, List[Dict]]:
        category_labels = [
            "sexual",
            "hate",
            "violence",
            "harassment",
            "self-harm",
            "sexual/minors",
            "hate/threatening",
            "violence/graphic",
            "safe",
        ]
        safety_labels = ["safe", "unsafe"]
        flag_map = [
            ("S", "sexual"),
            ("H", "hate"),
            ("V", "violence"),
            ("HR", "harassment"),
            ("SH", "self-harm"),
            ("S3", "sexual/minors"),
            ("H2", "hate/threatening"),
            ("V2", "violence/graphic"),
        ]
        required = [key for key, _ in flag_map]

        multi_label: List[Dict] = []
        binary: List[Dict] = []

        for item in items:
            if "prompt" not in item or not all(key in item for key in required):
                continue

            categories = [label for key, label in flag_map if int(item[key]) == 1]
            true_category_labels = categories if categories else ["safe"]
            is_unsafe = bool(categories)

            multi_label.append({
                "text": item["prompt"],
                "true_labels": true_category_labels,
                "all_labels": category_labels,
            })
            binary.append({
                "text": item["prompt"],
                "true_labels": ["unsafe" if is_unsafe else "safe"],
                "all_labels": safety_labels,
            })

        return {"oai": multi_label, "oai_safety": binary}

    @staticmethod
    def process_aegis(items: Iterable[Dict]) -> Dict[str, List[Dict]]:
        items = list(items)
        safety_labels = ["safe", "unsafe"]
        all_category_labels = sorted({
            category.strip()
            for item in items
            for category in (item.get("violated_categories") or "").split(",")
            if category.strip()
        })

        prompt_safety: List[Dict] = []
        response_safety: List[Dict] = []
        categories: List[Dict] = []

        for item in items:
            prompt = item.get("prompt") or ""
            response = item.get("response") or ""
            prompt_label = item.get("prompt_label")
            response_label = item.get("response_label")
            violated_str = item.get("violated_categories") or ""

            if prompt and prompt_label in ("safe", "unsafe"):
                prompt_safety.append({
                    "text": prompt,
                    "true_labels": [prompt_label],
                    "all_labels": safety_labels,
                })

            if response and response_label in ("safe", "unsafe"):
                response_safety.append({
                    "text": response,
                    "true_labels": [response_label],
                    "all_labels": safety_labels,
                })

            violated = [c.strip() for c in violated_str.split(",") if c.strip()]
            if prompt and violated:
                categories.append({
                    "text": prompt,
                    "true_labels": violated,
                    "all_labels": all_category_labels,
                })

        return {
            "aegis_prompt_safety": prompt_safety,
            "aegis_response_safety": response_safety,
            "aegis_categories": categories,
        }

    @staticmethod
    def process_simplest(items: Iterable[Dict]) -> List[Dict]:
        items = list(items)
        all_labels = sorted({item["harm_area"] for item in items})
        return [
            {
                "text": item["prompts_final"],
                "true_labels": [item["harm_area"]],
                "all_labels": all_labels,
            }
            for item in items
        ]

    @staticmethod
    def process_simplesafetytests(items: Iterable[Dict]) -> List[Dict]:
        items = list(items)
        all_labels = sorted({item["harm_area"] for item in items})
        return [
            {
                "text": item["prompt"],
                "true_labels": [item["harm_area"]],
                "all_labels": all_labels,
            }
            for item in items
        ]

    @staticmethod
    def process_harmbench(items: Iterable[Dict]) -> Dict[str, List[Dict]]:
        items = list(items)
        all_labels = sorted({item["category"] for item in items})
        prompts = []
        responses = []

        for item in items:
            true_labels = [item["category"]]
            prompts.append(
                {
                    "text": item["prompt"],
                    "true_labels": true_labels,
                    "all_labels": all_labels,
                }
            )
            responses.append(
                {
                    "text": item["context"],
                    "true_labels": true_labels,
                    "all_labels": all_labels,
                }
            )

        return {
            "harmbench_prompts": prompts,
            "harmbench_responses": responses,
        }

    @staticmethod
    def process_saferlhf(items: Iterable[Dict]) -> Dict[str, List[Dict]]:
        items = list(items)
        all_labels = set()

        for item in items:
            for category, value in item["response_0_harm_category"].items():
                if value:
                    all_labels.add(category)
            for category, value in item["response_1_harm_category"].items():
                if value:
                    all_labels.add(category)

        all_labels.add("safe")
        all_labels = sorted(all_labels)
        harm_category_examples = []

        for response_key, category_key in [
            ("response_0", "response_0_harm_category"),
            ("response_1", "response_1_harm_category"),
        ]:
            for item in items:
                true_labels = [
                    category
                    for category, value in item[category_key].items()
                    if value
                ]
                if not true_labels:
                    true_labels.append("safe")
                harm_category_examples.append(
                    {
                        "text": item[response_key],
                        "true_labels": true_labels,
                        "all_labels": all_labels,
                    }
                )

        safety_labels = ["safe", "unsafe"]
        response_safety_examples = []
        for item in items:
            prompt = GuardrailEvaluator.clean_text(item.get("prompt"))
            for response_key, safe_key in [
                ("response_0", "is_response_0_safe"),
                ("response_1", "is_response_1_safe"),
            ]:
                response = GuardrailEvaluator.clean_text(item.get(response_key))
                is_safe = item.get(safe_key)
                if not response or is_safe is None:
                    continue
                label = "safe" if is_safe else "unsafe"
                text = f"User prompt:\n{prompt}\n\nAssistant response:\n{response}"
                response_safety_examples.append({
                    "text": text,
                    "true_labels": [label],
                    "all_labels": safety_labels,
                })

        return {
            "saferlhf": harm_category_examples,
            "saferlhf_response_safety": response_safety_examples,
        }

    @staticmethod
    def process_beavertails(items: Iterable[Dict]) -> List[Dict]:
        items = list(items)
        all_labels = set()

        for item in items:
            for category, value in item["category"].items():
                if value:
                    all_labels.add(category)

        all_labels.add("safe")
        all_labels = sorted(all_labels)
        processed = []

        for item in items:
            true_labels = [
                category
                for category, value in item["category"].items()
                if value
            ]
            if not true_labels:
                true_labels.append("safe")
            processed.append(
                {
                    "text": item["response"],
                    "true_labels": true_labels,
                    "all_labels": all_labels,
                }
            )

        return processed

    @staticmethod
    def process_xstest(items: Iterable[Dict]) -> List[Dict]:
        items = list(items)
        all_labels = ["safe", "unsafe"]

        return [
            {
                "text": item["prompt"],
                "true_labels": [item["label"]],
                "all_labels": all_labels,
            }
            for item in items
        ]

    @staticmethod
    def offenseval_2020_label(value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, int):
            return "safe" if value == 0 else "unsafe" if value == 1 else None
        normalized = GuardrailEvaluator.clean_text(value).upper()
        if normalized == "NOT":
            return "safe"
        if normalized == "OFF":
            return "unsafe"
        return None

    @staticmethod
    def process_offenseval_2020(
        items: Iterable[Dict],
        language: Optional[str] = None,
    ) -> List[Dict]:
        processed = []
        for item in items:
            text = GuardrailEvaluator.clean_text(item.get("text"))
            label = GuardrailEvaluator.offenseval_2020_label(item.get("subtask_a"))
            if not text or not label:
                continue
            row = {
                "text": text,
                "true_labels": [label],
                "all_labels": SAFETY_CLASSES,
            }
            item_language = item.get("language") or language
            if item_language:
                row["language"] = item_language
            if item.get("original_id") is not None:
                row["source_id"] = item.get("original_id")
            processed.append(row)
        return processed

    @staticmethod
    def textdetox_toxicity_label(value) -> Optional[str]:
        flag = GuardrailEvaluator.int_flag(value)
        if flag == 0:
            return "safe"
        if flag == 1:
            return "unsafe"
        return None

    @staticmethod
    def process_textdetox_multilingual_toxicity(
        items: Iterable[Dict],
        language: Optional[str] = None,
    ) -> List[Dict]:
        processed = []
        for item in items:
            text = GuardrailEvaluator.clean_text(item.get("text"))
            label = GuardrailEvaluator.textdetox_toxicity_label(item.get("toxic"))
            if not text or not label:
                continue
            row = {
                "text": text,
                "true_labels": [label],
                "all_labels": SAFETY_CLASSES,
            }
            item_language = item.get("language") or language
            if item_language:
                row["language"] = item_language
            processed.append(row)
        return processed

    @staticmethod
    def int_flag(value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def process_toxicchat(items: Iterable[Dict]) -> Dict[str, List[Dict]]:
        safety_labels = ["safe", "unsafe"]
        toxicity_labels = ["non_toxic", "toxic"]
        jailbreaking_labels = ["not_jailbreaking", "jailbreaking"]
        processed = {
            "toxicchat_safe_unsafe_classification": [],
            "toxicchat_toxicity_classification": [],
            "toxicchat_jailbreaking_classification": [],
        }

        for item in items:
            text = GuardrailEvaluator.clean_text(item.get("user_input"))
            toxicity = GuardrailEvaluator.int_flag(item.get("toxicity"))
            jailbreaking = GuardrailEvaluator.int_flag(item.get("jailbreaking"))
            if not text or toxicity not in {0, 1} or jailbreaking not in {0, 1}:
                continue

            processed["toxicchat_toxicity_classification"].append({
                "text": text,
                "true_labels": ["toxic" if toxicity else "non_toxic"],
                "all_labels": toxicity_labels,
            })
            processed["toxicchat_jailbreaking_classification"].append({
                "text": text,
                "true_labels": ["jailbreaking" if jailbreaking else "not_jailbreaking"],
                "all_labels": jailbreaking_labels,
            })
            processed["toxicchat_safe_unsafe_classification"].append({
                "text": text,
                "true_labels": ["unsafe" if toxicity and jailbreaking else "safe"],
                "all_labels": safety_labels,
            })

        return processed

    @staticmethod
    def process_or_bench(items: Iterable[Dict], label: str) -> List[Dict]:
        processed = []
        for item in items:
            prompt = GuardrailEvaluator.clean_text(item.get("prompt"))
            if not prompt:
                continue
            row = {
                "text": prompt,
                "true_labels": [label],
                "all_labels": SAFETY_CLASSES,
            }
            category = GuardrailEvaluator.clean_text(item.get("category"))
            if category:
                row["category"] = category
            processed.append(row)
        return processed

    @staticmethod
    def first_present(item: Dict, keys: Sequence[str]) -> str:
        for key in keys:
            value = GuardrailEvaluator.clean_text(item.get(key))
            if value:
                return value
        return ""

    @staticmethod
    def process_jbb_behaviors(
        harmful_items: Iterable[Dict],
        benign_items: Iterable[Dict],
    ) -> Dict[str, List[Dict]]:
        harmful_items = list(harmful_items)
        benign_items = list(benign_items)
        behavior_labels = sorted({
            GuardrailEvaluator.first_present(item, ("Behavior", "behavior"))
            for item in harmful_items
            if GuardrailEvaluator.first_present(item, ("Behavior", "behavior"))
        })
        category_labels = sorted({
            GuardrailEvaluator.first_present(item, ("Category", "category"))
            for item in harmful_items
            if GuardrailEvaluator.first_present(item, ("Category", "category"))
        })
        processed = {
            "jbb_behaviors_behavior": [],
            "jbb_behaviors_category": [],
            "jbb_behaviors_safety": [],
        }

        for item in harmful_items:
            goal = GuardrailEvaluator.first_present(item, ("Goal", "goal"))
            behavior = GuardrailEvaluator.first_present(item, ("Behavior", "behavior"))
            category = GuardrailEvaluator.first_present(item, ("Category", "category"))
            if not goal:
                continue
            if behavior:
                processed["jbb_behaviors_behavior"].append({
                    "text": goal,
                    "true_labels": [behavior],
                    "all_labels": behavior_labels,
                })
            if category:
                processed["jbb_behaviors_category"].append({
                    "text": goal,
                    "true_labels": [category],
                    "all_labels": category_labels,
                })
            processed["jbb_behaviors_safety"].append({
                "text": goal,
                "true_labels": ["unsafe"],
                "all_labels": SAFETY_CLASSES,
            })

        for item in benign_items:
            goal = GuardrailEvaluator.first_present(item, ("Goal", "goal"))
            if not goal:
                continue
            processed["jbb_behaviors_safety"].append({
                "text": goal,
                "true_labels": ["safe"],
                "all_labels": SAFETY_CLASSES,
            })

        return processed

    @staticmethod
    def clean_text(value) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.strip()

    @staticmethod
    def wildguard_safety_label(value) -> Optional[str]:
        label = GuardrailEvaluator.clean_text(value).lower()
        if label == "harmful":
            return "unsafe"
        if label == "unharmful":
            return "safe"
        return None

    @staticmethod
    def wildguard_refusal_label(value) -> Optional[str]:
        label = GuardrailEvaluator.clean_text(value).lower()
        if label in {"refusal", "compliance"}:
            return label
        return None

    @staticmethod
    def wildguard_response_text(item: Dict) -> str:
        prompt = GuardrailEvaluator.clean_text(item.get("prompt"))
        response = GuardrailEvaluator.clean_text(item.get("response"))
        if not response:
            return ""
        return f"User prompt:\n{prompt}\n\nAssistant response:\n{response}"

    @staticmethod
    def process_wildguardmix(items: Iterable[Dict]) -> Dict[str, List[Dict]]:
        items = list(items)
        safety_labels = ["safe", "unsafe"]
        refusal_labels = ["refusal", "compliance"]
        subcategory_labels = sorted(
            {
                GuardrailEvaluator.clean_text(item.get("subcategory"))
                for item in items
                if GuardrailEvaluator.clean_text(item.get("prompt"))
                and GuardrailEvaluator.clean_text(item.get("subcategory"))
            }
        )
        processed = {
            "wildguard_prompt_safety": [],
            "wildguard_response_safety": [],
            "wildguard_response_refusal": [],
            "wildguard_prompt_subcategory": [],
        }

        for item in items:
            prompt = GuardrailEvaluator.clean_text(item.get("prompt"))
            response_text = GuardrailEvaluator.wildguard_response_text(item)

            prompt_safety = GuardrailEvaluator.wildguard_safety_label(
                item.get("prompt_harm_label")
            )
            if prompt and prompt_safety:
                processed["wildguard_prompt_safety"].append(
                    {
                        "text": prompt,
                        "true_labels": [prompt_safety],
                        "all_labels": safety_labels,
                    }
                )

            response_safety = GuardrailEvaluator.wildguard_safety_label(
                item.get("response_harm_label")
            )
            if response_text and response_safety:
                processed["wildguard_response_safety"].append(
                    {
                        "text": response_text,
                        "true_labels": [response_safety],
                        "all_labels": safety_labels,
                    }
                )

            response_refusal = GuardrailEvaluator.wildguard_refusal_label(
                item.get("response_refusal_label")
            )
            if response_text and response_refusal:
                processed["wildguard_response_refusal"].append(
                    {
                        "text": response_text,
                        "true_labels": [response_refusal],
                        "all_labels": refusal_labels,
                    }
                )

            subcategory = GuardrailEvaluator.clean_text(item.get("subcategory"))
            if prompt and subcategory:
                processed["wildguard_prompt_subcategory"].append(
                    {
                        "text": prompt,
                        "true_labels": [subcategory],
                        "all_labels": subcategory_labels,
                    }
                )

        return processed

    @staticmethod
    def process_polyguardprompts(items: Iterable[Dict]) -> Dict[str, List[Dict]]:
        items = list(items)
        safety_labels = ["safe", "unsafe"]
        refusal_labels = ["compliance", "refusal"]
        subcategory_labels = sorted({
            GuardrailEvaluator.clean_text(item.get("subcategory"))
            for item in items
            if GuardrailEvaluator.clean_text(item.get("subcategory"))
        })
        processed: Dict[str, List[Dict]] = {
            "polyguard_prompt_safety": [],
            "polyguard_response_safety": [],
            "polyguard_response_refusal": [],
            "polyguard_prompt_subcategory": [],
        }

        for item in items:
            prompt = GuardrailEvaluator.clean_text(item.get("prompt"))
            response_text = GuardrailEvaluator.wildguard_response_text(item)

            prompt_label = GuardrailEvaluator.clean_text(item.get("prompt_label")).lower()
            if prompt and prompt_label in {"safe", "unsafe"}:
                processed["polyguard_prompt_safety"].append({
                    "text": prompt,
                    "true_labels": [prompt_label],
                    "all_labels": safety_labels,
                })

            response_label = GuardrailEvaluator.clean_text(item.get("response_label")).lower()
            if response_text and response_label in {"safe", "unsafe"}:
                processed["polyguard_response_safety"].append({
                    "text": response_text,
                    "true_labels": [response_label],
                    "all_labels": safety_labels,
                })

            refusal_label = GuardrailEvaluator.clean_text(item.get("response_refusal_label")).lower()
            if response_text and refusal_label in {"compliance", "refusal"}:
                processed["polyguard_response_refusal"].append({
                    "text": response_text,
                    "true_labels": [refusal_label],
                    "all_labels": refusal_labels,
                })

            subcategory = GuardrailEvaluator.clean_text(item.get("subcategory"))
            if prompt and subcategory:
                processed["polyguard_prompt_subcategory"].append({
                    "text": prompt,
                    "true_labels": [subcategory],
                    "all_labels": subcategory_labels,
                })

        return processed

    def parse_gliner2_labels(self, value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            label = value.get("label")
            confidence = value.get("confidence", value.get("score"))
            if self.verbose and label and confidence is not None:
                print(f"{confidence:.4f}\t{label}")
            if label and (confidence is None or confidence >= self.threshold):
                return [label]
            return []
        if isinstance(value, list):
            selected = []
            for item in value:
                if isinstance(item, str):
                    selected.append(item)
                elif isinstance(item, dict) and item.get("label"):
                    confidence = item.get("confidence", item.get("score"))
                    if self.verbose and confidence is not None:
                        print(f"{confidence:.4f}\t{item['label']}")
                    if confidence is None or confidence >= self.threshold:
                        selected.append(item["label"])
            return selected
        return []

    def predict(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        multi_label: bool = True,
    ) -> List[List[str]]:
        if self.backend == "gliner2":
            return self.predict_gliner2(texts, labels, multi_label=multi_label)
        return self.predict_gliclass(texts, labels, multi_label=multi_label)

    def predict_gliclass(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        multi_label: bool = True,
    ) -> List[List[str]]:
        if self.pipeline is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        classification_type = "multi-label" if multi_label else "single-label"
        original_type = self.pipeline.pipe.classification_type
        self.pipeline.pipe.classification_type = classification_type
        try:
            results = self.pipeline(
                list(texts),
                list(labels),
                batch_size=self.batch_size,
            )
        finally:
            self.pipeline.pipe.classification_type = original_type

        predictions = []

        for result in results:
            selected = []
            for prediction in result:
                if self.verbose:
                    print(f"{prediction['score']:.4f}\t{prediction['label']}")
                if prediction["score"] >= self.threshold:
                    selected.append(prediction["label"])
            predictions.append(selected)

        return predictions

    def predict_gliner2(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        multi_label: bool = True,
    ) -> List[List[str]]:
        if self.extractor is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        schema = {
            "labels": {
                "labels": list(labels),
                "multi_label": multi_label,
                "cls_threshold": self.threshold,
            }
        }
        predictions = []

        for text in texts:
            try:
                result = self.extractor.classify_text(
                    text,
                    schema,
                    include_confidence=True,
                )
            except TypeError:
                result = self.extractor.classify_text(text, schema)

            selected = []
            raw_labels = result.get("labels") if isinstance(result, dict) else result
            selected.extend(self.parse_gliner2_labels(raw_labels))

            predictions.append(selected)

        return predictions

    @staticmethod
    def evaluate(
        predictions: Sequence[Sequence[str]],
        true_labels: Sequence[Sequence[str]],
        all_labels: Sequence[str],
    ) -> Dict[str, float]:
        from sklearn.metrics import f1_score
        from sklearn.preprocessing import MultiLabelBinarizer

        mlb = MultiLabelBinarizer(classes=list(all_labels))
        y_true = mlb.fit_transform(true_labels)
        y_pred = mlb.transform(predictions)

        return {
            "micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
        }

    @staticmethod
    def normalize_label(label: str) -> str:
        return str(label).strip().lower()

    @classmethod
    def labels_to_safety(
        cls,
        labels: Sequence[str],
        dataset_name: str,
    ) -> Optional[str]:
        normalized = {cls.normalize_label(label) for label in labels if label}
        if not normalized:
            return "safe"

        if dataset_name == "xstest":
            if "unsafe" in normalized:
                return "unsafe"
            if "safe" in normalized:
                return "safe"
            return "unsafe"

        if normalized <= NON_SAFETY_LABELS:
            return None
        if normalized & UNSAFE_LABELS:
            return "unsafe"
        if normalized - SAFE_LABELS - NON_SAFETY_LABELS:
            return "unsafe"
        if normalized & SAFE_LABELS:
            return "safe"
        return None

    @classmethod
    def evaluate_binary_safety(
        cls,
        dataset_name: str,
        predictions: Sequence[Sequence[str]],
        true_labels: Sequence[Sequence[str]],
    ) -> Dict:
        from collections import Counter
        from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

        y_true = []
        y_pred = []
        skipped = 0

        for predicted, expected in zip(predictions, true_labels):
            expected_safety = cls.labels_to_safety(expected, dataset_name)
            if expected_safety is None:
                skipped += 1
                continue

            predicted_safety = cls.labels_to_safety(predicted, dataset_name)
            if predicted_safety is None:
                predicted_safety = "safe"

            y_true.append(expected_safety)
            y_pred.append(predicted_safety)

        if not y_true:
            return {
                "available": False,
                "reason": "No safe/unsafe signal could be derived from this dataset's labels.",
                "skipped_examples": skipped,
            }

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=SAFETY_CLASSES,
            zero_division=0,
        )
        per_label = {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(SAFETY_CLASSES)
        }

        return {
            "available": True,
            "examples": len(y_true),
            "skipped_examples": skipped,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "micro": float(f1_score(
                y_true,
                y_pred,
                labels=SAFETY_CLASSES,
                average="micro",
                zero_division=0,
            )),
            "macro": float(f1_score(
                y_true,
                y_pred,
                labels=SAFETY_CLASSES,
                average="macro",
                zero_division=0,
            )),
            "weighted": float(f1_score(
                y_true,
                y_pred,
                labels=SAFETY_CLASSES,
                average="weighted",
                zero_division=0,
            )),
            "per_label": per_label,
            "true_label_counts": dict(Counter(y_true)),
            "predicted_label_counts": dict(Counter(y_pred)),
        }

    def dump_results(
        self,
        name: str,
        texts: Sequence[str],
        true_labels: Sequence[Sequence[str]],
        predictions: Sequence[Sequence[str]],
        all_labels: Sequence[str],
        binary_true: Optional[Sequence[Optional[str]]] = None,
        binary_predicted: Optional[Sequence[Optional[str]]] = None,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"debug_{name}.jsonl"

        with path.open("w", encoding="utf-8") as file:
            for index, (text, expected, predicted) in enumerate(
                zip(texts, true_labels, predictions)
            ):
                row = {
                    "text": text,
                    "true_labels": list(expected),
                    "predicted": list(predicted),
                    "all_labels": list(all_labels),
                }
                if binary_true is not None and binary_predicted is not None:
                    row["binary_true"] = binary_true[index]
                    row["binary_predicted"] = binary_predicted[index]

                file.write(
                    json.dumps(row, ensure_ascii=False)
                    + "\n"
                )

        print(f"Dumped {name} predictions to {path}")

    def run_dataset(self, name: str, examples: List[Dict], limit: Optional[int]) -> Dict:
        if limit is not None:
            examples = examples[:limit]
        if not examples:
            raise ValueError(f"{name} produced no examples.")

        texts = [item["text"] for item in examples]
        true_labels = [item["true_labels"] for item in examples]
        all_labels = list(examples[0]["all_labels"])
        random.shuffle(all_labels)

        multi_label = name not in SINGLE_LABEL_DATASETS
        print(f"\nEvaluating {name}: {len(texts)} examples, {len(all_labels)} labels")
        predictions = self.predict(texts, all_labels, multi_label=multi_label)
        metrics = self.evaluate(predictions, true_labels, all_labels)
        print(f"{name} results: {metrics}")

        binary_safety = None
        binary_true = None
        binary_predicted = None

        if name in BINARY_SAFETY_DATASETS:
            binary_safety = self.evaluate_binary_safety(name, predictions, true_labels)
            binary_true = [
                self.labels_to_safety(expected, name)
                for expected in true_labels
            ]
            binary_predicted = [
                "safe" if self.labels_to_safety(preds, name) is None
                else self.labels_to_safety(preds, name)
                for preds in predictions
            ]
            if binary_safety["available"]:
                print(
                    f"{name} binary safety: accuracy={binary_safety['accuracy']:.4f} "
                    f"macro={binary_safety['macro']:.4f} "
                    f"weighted={binary_safety['weighted']:.4f}"
                )

        self.dump_results(
            name,
            texts,
            true_labels,
            predictions,
            all_labels,
            binary_true=binary_true,
            binary_predicted=binary_predicted,
        )
        result: Dict = {
            "examples": len(texts),
            "label_count": len(all_labels),
            "labels": all_labels,
            "metrics": metrics,
        }
        if binary_safety is not None:
            result["binary_safety"] = binary_safety
        return result

    def save_eval_report(
        self,
        path: Path,
        dataset_names: Sequence[str],
        results: Dict[str, Dict],
        limit: Optional[int],
        seed: int,
        max_text_chars: Optional[int],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model_info,
            "evaluation": {
                "datasets": list(dataset_names),
                "threshold": self.threshold,
                "batch_size": self.batch_size,
                "limit": limit,
                "max_text_chars": max_text_chars,
                "seed": seed,
            },
            "results": results,
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(f"Saved evaluation report to {path}")


def parse_dataset_names(value: str) -> List[str]:
    if value == "all":
        return ALL_DATASETS

    names = [name.strip() for name in value.split(",") if name.strip()]
    invalid = sorted(set(names) - set(ALL_DATASETS))
    if invalid:
        raise ValueError(f"Unknown dataset name(s): {', '.join(invalid)}")
    return names


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def truncate_examples_text(
    examples: Sequence[Dict],
    max_text_chars: Optional[int],
) -> List[Dict]:
    if max_text_chars is None:
        return list(examples)

    truncated = []
    for item in examples:
        copied = dict(item)
        text = copied.get("text")
        if isinstance(text, str):
            copied["text"] = text[:max_text_chars]
        truncated.append(copied)
    return truncated


def load_requested_datasets(
    evaluator: GuardrailEvaluator,
    dataset_names: Sequence[str],
    data_dir: Path,
    max_text_chars: Optional[int] = DEFAULT_MAX_TEXT_CHARS,
) -> Dict[str, List[Dict]]:
    loaded = {}
    requested = set(dataset_names)

    oai_requested = requested & {"oai", "oai_safety"}
    if oai_requested:
        oai_processed = evaluator.process_oai_dataset(
            evaluator.load_jsonl(data_dir / "oai.jsonl")
        )
        for name in oai_requested:
            loaded[name] = oai_processed[name]

    if requested & AEGIS_DATASETS:
        aegis_processed = evaluator.process_aegis(
            evaluator.load_json(data_dir / "aegis.json")
        )
        for name in AEGIS_DATASETS & requested:
            loaded[name] = aegis_processed[name]

    if "simplest" in requested:
        loaded["simplest"] = evaluator.process_simplest(
            evaluator.load_csv(data_dir / "test.csv")
        )

    if "simplesafetytests" in requested:
        local_path = data_dir / "test.csv"
        if local_path.exists():
            sst = evaluator.load_csv(local_path)
        else:
            sst = evaluator.load_hf("Bertievidgen/SimpleSafetyTests", split="test")
        loaded["simplesafetytests"] = evaluator.process_simplesafetytests(sst)

    if requested & {"harmbench_prompts", "harmbench_responses"}:
        local_path = data_dir / "harmbench_contextual_train.jsonl"
        if local_path.exists():
            harmbench = evaluator.load_jsonl(local_path)
        else:
            harmbench = evaluator.load_hf(
                "walledai/HarmBench",
                subset="contextual",
                split="train",
            )
        loaded.update(evaluator.process_harmbench(harmbench))

    saferlhf_requested = requested & {"saferlhf", "saferlhf_response_safety"}
    if saferlhf_requested:
        local_path = data_dir / "saferlhf_test.jsonl"
        if local_path.exists():
            saferlhf_raw = evaluator.load_jsonl(local_path)
        else:
            saferlhf_raw = evaluator.load_hf(
                "PKU-Alignment/PKU-SafeRLHF",
                subset="default",
                split="test",
            )
        all_saferlhf = evaluator.process_saferlhf(saferlhf_raw)
        for name in saferlhf_requested:
            loaded[name] = all_saferlhf[name]

    if "beavertails" in requested:
        local_path = data_dir / "beavertails_330k_test.jsonl"
        if local_path.exists():
            beavertails = evaluator.load_jsonl(local_path)
        else:
            beavertails = evaluator.load_hf(
                "PKU-Alignment/BeaverTails",
                split="330k_test",
            )
        loaded["beavertails"] = evaluator.process_beavertails(beavertails)

    if "xstest" in requested:
        local_path = data_dir / "xstest_test.jsonl"
        if local_path.exists():
            xstest = evaluator.load_jsonl(local_path)
        else:
            xstest = evaluator.load_hf("walledai/XSTest", split="test")
        loaded["xstest"] = evaluator.process_xstest(xstest)

    if "offenseval_2020_safe_unsafe" in requested:
        local_path = data_dir / OFFENSEVAL_2020_LOCAL_FILE
        if local_path.exists():
            loaded["offenseval_2020_safe_unsafe"] = evaluator.load_jsonl(local_path)
        else:
            offenseval_rows = []
            for language in OFFENSEVAL_2020_LANGUAGES:
                language_rows = evaluator.load_hf(
                    "strombergnlp/offenseval_2020",
                    subset=language,
                    split="test",
                )
                offenseval_rows.extend(
                    evaluator.process_offenseval_2020(language_rows, language)
                )
            loaded["offenseval_2020_safe_unsafe"] = offenseval_rows

    if "textdetox_multilingual_toxicity_safe_unsafe" in requested:
        local_path = data_dir / TEXTDETOX_MULTILINGUAL_TOXICITY_LOCAL_FILE
        if local_path.exists():
            loaded["textdetox_multilingual_toxicity_safe_unsafe"] = (
                evaluator.load_jsonl(local_path)
            )
        else:
            textdetox_rows = []
            for language in TEXTDETOX_MULTILINGUAL_TOXICITY_LANGUAGES:
                language_rows = evaluator.load_hf(
                    "textdetox/multilingual_toxicity_dataset",
                    split=language,
                )
                textdetox_rows.extend(
                    evaluator.process_textdetox_multilingual_toxicity(
                        language_rows,
                        language,
                    )
                )
            loaded["textdetox_multilingual_toxicity_safe_unsafe"] = textdetox_rows

    if "pan12_predator_conversation_safety" in requested:
        local_path = data_dir / PAN12_PREDATOR_CONVERSATION_SAFETY_LOCAL_FILE
        if not local_path.exists():
            raise FileNotFoundError(
                "Missing PAN12 eval file. Generate it with "
                "`python scripts/prepare_pan12_eval.py` after downloading and "
                f"unarchiving the dataset: {local_path}"
            )
        loaded["pan12_predator_conversation_safety"] = evaluator.load_jsonl(
            local_path
        )
        if (
            len(loaded["pan12_predator_conversation_safety"])
            > PAN12_PREDATOR_CONVERSATION_SAFETY_SAMPLE_SIZE
        ):
            loaded["pan12_predator_conversation_safety"] = random.sample(
                loaded["pan12_predator_conversation_safety"],
                PAN12_PREDATOR_CONVERSATION_SAFETY_SAMPLE_SIZE,
            )

    if "med_safety_bench_safe_unsafe" in requested:
        local_path = data_dir / MED_SAFETY_BENCH_SAFE_UNSAFE_LOCAL_FILE
        if not local_path.exists():
            raise FileNotFoundError(
                "Missing Med Safety Bench eval file. Generate it with "
                f"`python scripts/prepare_med_safety_bench_eval.py`: {local_path}"
            )
        loaded["med_safety_bench_safe_unsafe"] = evaluator.load_jsonl(local_path)

    toxicchat_requested = requested & TOXICCHAT_DATASETS
    if toxicchat_requested:
        toxicchat_raw = evaluator.load_hf(
            "lmsys/toxic-chat",
            subset="toxicchat0124",
            split="test",
        )
        all_toxicchat = evaluator.process_toxicchat(toxicchat_raw)
        for name in toxicchat_requested:
            loaded[name] = all_toxicchat[name]

    or_bench_requested = requested & set(OR_BENCH_DATASETS)
    for name in or_bench_requested:
        config = OR_BENCH_DATASETS[name]
        rows = evaluator.load_hf(
            "bench-llm/or-bench",
            subset=config["subset"],
            split="train",
        )
        sample_size = config["sample_size"]
        if sample_size is not None and len(rows) > sample_size:
            rows = random.sample(rows, sample_size)
        loaded[name] = evaluator.process_or_bench(rows, config["label"])

    jbb_behaviors_requested = requested & JBB_BEHAVIORS_DATASETS
    if jbb_behaviors_requested:
        harmful_behaviors = evaluator.load_hf(
            "JailbreakBench/JBB-Behaviors",
            subset="behaviors",
            split="harmful",
        )
        benign_behaviors = evaluator.load_hf(
            "JailbreakBench/JBB-Behaviors",
            subset="behaviors",
            split="benign",
        )
        all_jbb_behaviors = evaluator.process_jbb_behaviors(
            harmful_behaviors,
            benign_behaviors,
        )
        for name in jbb_behaviors_requested:
            loaded[name] = all_jbb_behaviors[name]

    gliclass_requested = requested & GLICLASS_DATASETS
    for name in gliclass_requested:
        path = data_dir / GLICLASS_LOCAL_FILES[name]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing local GLiClass eval file for {name}: {path}"
            )
        loaded[name] = evaluator.load_jsonl(path)

    wildguard_requested = requested & WILDGUARD_DATASETS
    if wildguard_requested:
        local_wildguard = {}
        for name in wildguard_requested:
            path = data_dir / WILDGUARD_LOCAL_FILES[name]
            if path.exists():
                local_wildguard[name] = evaluator.load_jsonl(path)

        if local_wildguard.keys() == wildguard_requested:
            loaded.update(local_wildguard)
        else:
            wildguardmix = evaluator.load_hf(
                "allenai/wildguardmix",
                subset="wildguardtest",
                split="test",
            )
            loaded.update(evaluator.process_wildguardmix(wildguardmix))

    polyguard_requested = requested & POLYGUARD_DATASETS
    if polyguard_requested:
        polyguardprompts = evaluator.load_hf(
            "ToxicityPrompts/PolyGuardPrompts",
            split="test",
        )
        all_polyguard = evaluator.process_polyguardprompts(polyguardprompts)
        for name in polyguard_requested:
            loaded[name] = all_polyguard[name]

    return {
        name: truncate_examples_text(loaded[name], max_text_chars)
        for name in dataset_names
    }


def default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GLiClass or GLiNER2 on guardrail datasets."
    )
    parser.add_argument(
        "--backend",
        choices=MODEL_BACKENDS,
        default="gliclass",
        help="Model backend to use for zero-shot classification.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/eval"),
        help=(
            "Directory containing local eval files such as oai.jsonl, aegis.json, "
            "wildguardmix_*.jsonl, and gliclass_*.jsonl."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help=(
            "JSON file for run-level metrics and model metadata. "
            "Defaults to OUTPUT_DIR/eval_results.json."
        ),
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help=(
            "Comma-separated names, or 'all'. Available: "
            + ", ".join(ALL_DATASETS)
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-text-chars",
        type=positive_int,
        default=DEFAULT_MAX_TEXT_CHARS,
        help=(
            "Maximum number of characters kept from each dataset text input. "
            "Defaults to 8192."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=default_device(),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    dataset_names = parse_dataset_names(args.datasets)

    evaluator = GuardrailEvaluator(
        model_name=args.model,
        token=args.token,
        device=args.device,
        threshold=args.threshold,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        backend=args.backend,
        verbose=args.verbose,
    )
    evaluator.load_model()

    results_file = args.results_file or (args.output_dir / "eval_results.json")
    datasets = load_requested_datasets(
        evaluator,
        dataset_names,
        args.data_dir,
        max_text_chars=args.max_text_chars,
    )
    summary = {}
    for name in dataset_names:
        summary[name] = evaluator.run_dataset(name, datasets[name], args.limit)
        evaluator.save_eval_report(
            path=results_file,
            dataset_names=dataset_names,
            results=summary,
            limit=args.limit,
            seed=args.seed,
            max_text_chars=args.max_text_chars,
        )

    print("\nSummary")
    for name, result in summary.items():
        binary_safety = result.get("binary_safety")
        metrics = result["metrics"]
        if binary_safety is not None and binary_safety["available"]:
            line = (
                f"{name}: binary_macro={binary_safety['macro']:.4f} "
                f"binary_accuracy={binary_safety['accuracy']:.4f} "
                f"(micro={metrics['micro']:.4f} macro={metrics['macro']:.4f})"
            )
        else:
            line = (
                f"{name}: micro={metrics['micro']:.4f} "
                f"macro={metrics['macro']:.4f} "
                f"weighted={metrics['weighted']:.4f}"
            )
        print(line)


if __name__ == "__main__":
    main()
