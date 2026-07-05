"""
Medical QA System - Generator Model State.

Dataclass that captures the runtime state of the loaded Gemma generator with
its PEFT LoRA adapter applied.  Used internally by :class:`GemmaService`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GeneratorModelState:
    """
    Holds the runtime state of the Gemma-2B-IT generator with LoRA.

    Attributes:
        base_model:  The base AutoModelForCausalLM object (before PEFT wrapping).
        peft_model:  The PeftModel wrapping the base model with LoRA applied.
        tokenizer:   The AutoTokenizer loaded from the LoRA adapter directory.
        model_path:  Filesystem path to the base model weights.
        lora_path:   Filesystem path to the LoRA adapter artefacts.
        device:      Compute device string (``"cuda"`` or ``"cpu"``).
        is_loaded:   Whether the model has been successfully loaded.
        load_time:   Seconds elapsed during model loading.
    """

    base_model: Any
    peft_model: Any
    tokenizer: Any
    model_path: str
    lora_path: str
    device: str
    is_loaded: bool = field(default=False)
    load_time: float = field(default=0.0)
