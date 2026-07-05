"""
Medical QA System - Gemma Generator Service.

Loads the ``google/gemma-2b-it`` base model (from local cache or Hugging Face)
and applies the pre-trained LoRA adapter from ``models/gemma_medical_qa_final/``
using the PEFT library.

The service automatically detects CUDA and uses half-precision (float16) on
GPU for memory efficiency.  On CPU the model runs in full precision (float32).
Generation is performed with ``torch.no_grad()`` for inference-only mode.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

from app.core.config.settings import Settings
from app.services.logger_service import get_logger

logger = get_logger()


class GemmaService:
    """
    Service that manages the Gemma-2B-IT generator model with LoRA adapter.

    Responsibilities:
      - Load the base causal language model from disk or HuggingFace.
      - Apply the PEFT LoRA adapter from the ``gemma_medical_qa_final/`` artefact.
      - Run text generation for a given prompt string.
      - Handle CUDA / CPU device placement automatically.

    Attributes:
        _settings:   Application configuration.
        _model:      The loaded PEFT model (None until :meth:`load`).
        _tokenizer:  The loaded tokeniser (None until :meth:`load`).
        _device:     Resolved device string (``"cuda"`` or ``"cpu"``).
        _is_loaded:  Whether the model has been successfully loaded.
        _load_time:  Seconds taken to load the model.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialise GemmaService without loading the model.

        Args:
            settings: The application settings object.
        """
        self._settings = settings
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None
        self._device: str = "cpu"
        self._is_loaded: bool = False
        self._load_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, device: Optional[str] = None) -> None:
        """
        Load the base model and apply the LoRA adapter.

        Steps performed:
        1. Resolve and record the compute device.
        2. Load the tokeniser from the LoRA adapter directory (which contains
           the fine-tuned tokeniser config).
        3. Load the base ``gemma-2b-it`` weights from the local cache or
           download them from Hugging Face and save them for future use.
        4. Apply the LoRA adapter via
           ``peft.PeftModel.from_pretrained(base_model, lora_path)``.
        5. Set the model to evaluation mode.

        Args:
            device: Optional device override (``"cuda"`` or ``"cpu"``).
                    If omitted, determined by :meth:`ConfigurationService.get_device`.

        Raises:
            RuntimeError: If the model or tokeniser cannot be loaded.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        t_start = time.perf_counter()

        # --- Resolve device ---
        if device:
            self._device = device
        else:
            from app.services.configuration_service import ConfigurationService
            self._device = ConfigurationService.get_device(self._settings)

        logger.log_model(f"Loading Gemma model on device: {self._device}")

        lora_path = Path(self._settings.LORA_PATH)
        model_path = Path(self._settings.MODEL_PATH)

        # --- Load tokeniser from fine-tuned artefacts ---
        try:
            logger.log_model(f"Loading tokeniser from LoRA path: {lora_path.resolve()}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(lora_path),
                local_files_only=True,
            )
            # Ensure pad token is defined
            if self._tokenizer.pad_token is None:  # type: ignore[union-attr]
                self._tokenizer.pad_token = self._tokenizer.eos_token  # type: ignore[union-attr]
            logger.log_model("Tokeniser loaded successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokeniser from '{lora_path}': {exc}"
            ) from exc

        # --- Load base model ---
        dtype = (
            __import__("torch").float16
            if self._device == "cuda"
            else __import__("torch").float32
        )

        try:
            if model_path.is_dir():
                logger.log_model(
                    f"Loading base model from local cache: {model_path.resolve()}"
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=dtype,
                    device_map="auto" if self._device == "cuda" else None,
                    local_files_only=True,
                )
                logger.log_model("Base model loaded from local cache.")
            else:
                hf_model_id = self._settings.HF_GENERATOR_MODEL_ID
                logger.log_model(
                    f"Base model not found locally.  "
                    f"Downloading '{hf_model_id}' from Hugging Face Hub …  "
                    "This may take 10–20 minutes on the first run."
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_id,
                    torch_dtype=dtype,
                    device_map="auto" if self._device == "cuda" else None,
                )
                # Save for future use
                logger.log_model(
                    f"Saving base model to: {model_path.resolve()} …"
                )
                model_path.mkdir(parents=True, exist_ok=True)
                base_model.save_pretrained(str(model_path))
                self._tokenizer.save_pretrained(str(model_path))  # type: ignore[union-attr]
                logger.log_model("Base model saved to local cache.")

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load base generator model from "
                f"'{model_path}' / '{self._settings.HF_GENERATOR_MODEL_ID}': {exc}"
            ) from exc

        # --- Apply LoRA adapter ---
        try:
            logger.log_model(
                f"Applying LoRA adapter from: {lora_path.resolve()}"
            )
            self._model = PeftModel.from_pretrained(
                base_model,
                str(lora_path),
                is_trainable=False,
            )
            logger.log_model("LoRA adapter applied successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to apply LoRA adapter from '{lora_path}': {exc}"
            ) from exc

        # --- Finalise ---
        self._model.eval()  # type: ignore[union-attr]
        if self._device == "cpu":
            self._model = self._model.float()  # type: ignore[union-attr]

        self._load_time = time.perf_counter() - t_start
        self._is_loaded = True
        logger.log_model(
            f"Gemma model with LoRA adapter ready.  "
            f"Device: {self._device}, "
            f"Load time: {self._load_time:.2f}s"
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> Tuple[str, float]:
        """
        Generate an answer for the given *prompt*.

        Tokenises the prompt, runs the model with ``torch.no_grad()``, and
        decodes only the newly generated tokens (not the input prompt).

        Args:
            prompt: The fully-formatted prompt string from
                    :class:`PromptBuilderService`.

        Returns:
            A tuple ``(answer_text, generation_time_seconds)`` where
            *answer_text* is the stripped model output and
            *generation_time_seconds* is the wall-clock elapsed time.

        Raises:
            RuntimeError: If the model has not been loaded.
            TimeoutError: If generation exceeds 300 seconds.
        """
        import torch

        self._assert_loaded()

        t_start = time.perf_counter()

        try:
            # Tokenise
            inputs = self._tokenizer(  # type: ignore[call-arg]
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False,
            )
            input_ids = inputs["input_ids"].to(self._device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._device)

            prompt_length = input_ids.shape[1]

            # Generate
            with torch.no_grad():
                output_ids = self._model.generate(  # type: ignore[union-attr]
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self._settings.MAX_NEW_TOKENS,
                    temperature=self._settings.TEMPERATURE,
                    top_p=self._settings.TOP_P,
                    top_k=self._settings.TOP_K,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,  # type: ignore[union-attr]
                    eos_token_id=self._tokenizer.eos_token_id,  # type: ignore[union-attr]
                    repetition_penalty=1.1,
                )

            # Decode only newly generated tokens
            new_tokens = output_ids[0][prompt_length:]
            answer = self._tokenizer.decode(  # type: ignore[union-attr]
                new_tokens, skip_special_tokens=True
            ).strip()

            elapsed = time.perf_counter() - t_start

            # Timeout guard
            if elapsed > 300:
                raise TimeoutError(
                    f"Generation exceeded 300 seconds (took {elapsed:.1f}s). "
                    "Consider reducing MAX_NEW_TOKENS or switching to a GPU device."
                )

            logger.log_model(
                f"Generation complete.  "
                f"Input tokens: {prompt_length}, "
                f"Output tokens: {len(new_tokens)}, "
                f"Time: {elapsed:.2f}s"
            )
            return answer, elapsed

        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "CUDA out of memory" in str(exc):
                raise RuntimeError(
                    "GPU ran out of memory during generation.  "
                    "Reduce MAX_NEW_TOKENS or switch DEVICE to 'cpu'."
                ) from exc
            raise

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the Gemma model has been successfully loaded."""
        return self._is_loaded

    @property
    def device(self) -> str:
        """The compute device the model is running on."""
        return self._device

    @property
    def load_time(self) -> float:
        """Seconds elapsed during model loading."""
        return self._load_time

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_loaded(self) -> None:
        """Raise RuntimeError if the model has not been loaded."""
        if not self._is_loaded or self._model is None:
            raise RuntimeError(
                "GemmaService.load() must be called before generating text."
            )
