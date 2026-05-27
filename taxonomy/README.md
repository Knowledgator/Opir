# Guardrail Taxonomy

This directory contains the Opir guardrail taxonomy used for multi-label safety classification. The taxonomy is stored in [`guardrail_taxonomy.json`](guardrail_taxonomy.json) as a three-level hierarchy:

- Level 1: broad safety domains, such as `toxicity`, `cybersecurity`, or `safe_and_benign`.
- Level 2: policy subcategories within each domain.
- Level 3: concrete leaf labels used for fine-grained classification.

The labels are written as stable snake-case identifiers so they can be passed directly to zero-shot classifiers, mapped to policy actions, or converted into user-facing names by downstream applications.

## Summary

| Level | Description | Count |
|---|---|---:|
| 1 | Top-level safety domains | 16 |
| 2 | Mid-level subcategories | 126 |
| 3 | Leaf labels | 854 |

Subcategories per top-level domain range from 2 to 22. Leaf labels per subcategory range from 3 to 16.

## Top-Level Domains

| Domain | Subcategories | Leaf labels |
|---|---:|---:|
| `toxicity` | 6 | 41 |
| `violence_and_physical_harm` | 5 | 30 |
| `self_harm_and_suicide` | 5 | 30 |
| `sexual_content` | 5 | 30 |
| `child_safety` | 5 | 30 |
| `personal_information_privacy_and_intellectual_property` | 18 | 129 |
| `cybersecurity` | 6 | 36 |
| `criminal_and_illegal_activity` | 7 | 46 |
| `regulated_goods_and_advice` | 6 | 33 |
| `biological_medical_and_environmental_harm` | 22 | 177 |
| `weapons_of_mass_destruction` | 8 | 67 |
| `information_integrity_and_manipulation` | 10 | 60 |
| `ai_system_security_and_reliability` | 12 | 79 |
| `bias_fairness_and_representation` | 5 | 30 |
| `other_or_uncertain` | 2 | 12 |
| `safe_and_benign` | 4 | 24 |

