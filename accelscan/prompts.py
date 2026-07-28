"""Prompt templates for stage-2 LLM extraction.

PROMPT_VERSION is stamped into every mentions row; bump it on any wording
change so downstream tables never silently mix prompt variants.
"""

PROMPT_VERSION = 'p1'

SYSTEM_PROMPT = """\
You extract hardware-accelerator mentions from scientific paper passages.

For each accelerator mention in the passage (GPU, TPU, FPGA, NPU, or other \
compute accelerator), emit one JSON mention object. If the passage contains \
no accelerator mention (for example, a registry false positive such as the \
"K80 antibody", "BMW M2", or a graph convolutional network "GCN"), return \
{"mentions": []} or use accelerator_subtype "not-an-accelerator".

Field rules:
- model_raw: copy the surface form verbatim (e.g. "V100-SXM2", "RTX3090").
- model_normalized: the canonical marketing name (e.g. "NVIDIA Tesla V100", \
"NVIDIA GeForce RTX 3090", "Google TPU v3", "AMD Instinct MI250X"); null when \
only a generic term like "GPU" appears.
- device_count: the number of devices of that model this passage reports \
(e.g. "8 V100s" -> 8, "a single GPU" -> 1, "four nodes with 4 GPUs each" -> \
16 with basis "inferred"); null if unstated.
- memory_gb: per-device memory when stated ("V100 32GB" -> 32); null otherwise.
- usage_context, exactly one of:
  - used-in-this-work: the authors ran computations for THIS paper on it.
  - comparison-or-related-work: mentioned as prior/related work, a baseline \
another paper used, or a spec comparison the authors did not run.
  - object-of-study: the hardware itself is what the paper studies, \
benchmarks, simulates, or designs against.
  - speculative-future: announced/future hardware, or hypothetical use.
- evidence_quote: the shortest verbatim span that justifies the labels.

Emit one mention per distinct model in the passage. A generic "GPU"/"TPU" \
mention gets its own mention object only when no specific model in the same \
passage covers it (e.g. "8 GPUs" with no model named).

Examples:

Passage: "All models were trained on 4 NVIDIA A100 (80 GB) GPUs using PyTorch."
{"mentions": [{"model_raw": "NVIDIA A100", "model_normalized": "NVIDIA A100", \
"manufacturer": "nvidia", "accelerator_subtype": "datacenter-gpu", \
"device_count": 4, "device_count_basis": "explicit", "memory_gb": 80.0, \
"usage_context": "used-in-this-work", "evidence_quote": "trained on 4 NVIDIA \
A100 (80 GB) GPUs"}]}

Passage: "Unlike prior work which required 64 TPUv3 cores (Smith et al.), our \
method runs on a single consumer GPU (GTX 1080 Ti)."
{"mentions": [{"model_raw": "TPUv3", "model_normalized": "Google TPU v3", \
"manufacturer": "google", "accelerator_subtype": "tpu", "device_count": 64, \
"device_count_basis": "explicit", "memory_gb": null, "usage_context": \
"comparison-or-related-work", "evidence_quote": "prior work which required 64 \
TPUv3 cores"}, {"model_raw": "GTX 1080 Ti", "model_normalized": "NVIDIA \
GeForce GTX 1080 Ti", "manufacturer": "nvidia", "accelerator_subtype": \
"consumer-gpu", "device_count": 1, "device_count_basis": "explicit", \
"memory_gb": null, "usage_context": "used-in-this-work", "evidence_quote": \
"our method runs on a single consumer GPU (GTX 1080 Ti)"}]}

Passage: "We propose an optimized roofline model of the Fermi GPU \
architecture and validate it on a Tesla C2050."
{"mentions": [{"model_raw": "Fermi", "model_normalized": "NVIDIA Fermi", \
"manufacturer": "nvidia", "accelerator_subtype": "datacenter-gpu", \
"device_count": null, "device_count_basis": null, "memory_gb": null, \
"usage_context": "object-of-study", "evidence_quote": "roofline model of the \
Fermi GPU architecture"}, {"model_raw": "Tesla C2050", "model_normalized": \
"NVIDIA Tesla C2050", "manufacturer": "nvidia", "accelerator_subtype": \
"datacenter-gpu", "device_count": 1, "device_count_basis": "inferred", \
"memory_gb": null, "usage_context": "used-in-this-work", "evidence_quote": \
"validate it on a Tesla C2050"}]}

Passage: "K80 cells were incubated with the antibody for 24 hours."
{"mentions": []}
"""

USER_TEMPLATE = """\
section: {section}
registry hints: {hints}
passage:
{passage}"""


def build_user_prompt(section: str | None, surfaces: list[str], passage: str) -> str:
    return USER_TEMPLATE.format(
        section=section or '(unknown)',
        hints=', '.join(dict.fromkeys(surfaces)) or '(none)',
        passage=passage,
    )
