"""Structured-output schema for stage-2 LLM extraction (guided JSON decoding)."""

from typing import Literal

from pydantic import BaseModel, Field

Manufacturer = Literal['nvidia', 'amd', 'intel', 'google', 'apple', 'huawei',
                       'graphcore', 'cerebras', 'amazon', 'other', 'unknown']
Subtype = Literal['datacenter-gpu', 'consumer-gpu', 'workstation-gpu',
                  'mobile-gpu', 'tpu', 'fpga', 'npu-asic', 'manycore',
                  'generic-gpu', 'generic-accelerator', 'not-an-accelerator']
UsageContext = Literal['used-in-this-work', 'comparison-or-related-work',
                       'object-of-study', 'speculative-future']


class Mention(BaseModel):
    model_raw: str = Field(description='verbatim surface form copied from the passage')
    model_normalized: str | None = Field(
        description='canonical model name, e.g. "NVIDIA Tesla V100"; null if only generic')
    manufacturer: Manufacturer
    accelerator_subtype: Subtype
    device_count: int | None = Field(
        description='number of devices used, e.g. "8 V100s" -> 8; null if unstated')
    device_count_basis: Literal['explicit', 'inferred'] | None
    memory_gb: float | None = Field(
        description='per-device memory if stated, e.g. "32GB V100" -> 32')
    usage_context: UsageContext
    evidence_quote: str = Field(
        description='short verbatim span from the passage supporting the labels')


class PassageExtraction(BaseModel):
    mentions: list[Mention] = Field(
        description='empty list if the passage contains no accelerator mention')


EXTRACTION_JSON_SCHEMA = PassageExtraction.model_json_schema()
