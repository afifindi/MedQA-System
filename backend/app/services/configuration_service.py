"""
Medical QA System - Configuration Validation Service.

Validates that all required model artefacts and configuration paths exist
before the application starts serving requests.  Also determines the
appropriate compute device (CUDA vs CPU) based on runtime availability.

All errors include actionable messages telling the operator exactly what is
missing and how to resolve it.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config.settings import Settings
from app.services.logger_service import get_logger

logger = get_logger()


class ConfigurationService:
    """
    Validates application configuration and resolves the compute device.

    This service is called once during application startup, before any ML
    models are loaded.  It raises ``FileNotFoundError`` if mandatory artefacts
    are absent so that the operator receives a clear error rather than a
    cryptic runtime exception later.

    Methods:
        validate_all: Check all required paths and log warnings for optional ones.
        get_device:   Resolve the compute device string from settings.
    """

    @staticmethod
    def validate_all(settings: Settings) -> None:
        """
        Validate all required model artefact paths.

        Checks the following in order:
          1. FAISS index file exists.
          2. Knowledge base CSV exists.
          3. Knowledge base embeddings .npy exists.
          4. LoRA adapter directory exists.
          5. ``adapter_config.json`` inside the LoRA directory exists.
          6. Warns (does not raise) if the base model directories are absent
             because they will be auto-downloaded on first startup.

        Args:
            settings: The application settings object.

        Raises:
            FileNotFoundError: If any mandatory artefact is missing.
        """
        logger.log_startup("Starting configuration validation …")

        # --- FAISS Index ---
        faiss_path = Path(settings.FAISS_PATH)
        if not faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{faiss_path.resolve()}'.  "
                "Make sure the file 'faiss_index.bin' is present inside "
                "the models/gemma_medical_qa_final/ directory.  "
                "Do NOT regenerate the index; it must match the training artefacts."
            )
        logger.log_startup(f"✓ FAISS index found: {faiss_path.resolve()}")

        # --- Knowledge Base CSV ---
        kb_path = Path(settings.KNOWLEDGE_BASE_PATH)
        if not kb_path.exists():
            raise FileNotFoundError(
                f"Knowledge base CSV not found at '{kb_path.resolve()}'.  "
                "Ensure 'knowledge_base.csv' is inside models/gemma_medical_qa_final/."
            )
        logger.log_startup(f"✓ Knowledge base CSV found: {kb_path.resolve()}")

        # --- Knowledge Base Embeddings ---
        emb_path = Path(settings.KB_EMBEDDINGS_PATH)
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Knowledge base embeddings not found at '{emb_path.resolve()}'.  "
                "Ensure 'kb_embeddings.npy' is inside models/gemma_medical_qa_final/."
            )
        logger.log_startup(f"✓ KB embeddings found: {emb_path.resolve()}")

        # --- LoRA Adapter Directory ---
        lora_dir = Path(settings.LORA_PATH)
        if not lora_dir.is_dir():
            raise FileNotFoundError(
                f"LoRA adapter directory not found at '{lora_dir.resolve()}'.  "
                "The directory 'models/gemma_medical_qa_final/' must exist and "
                "contain adapter_model.safetensors and adapter_config.json."
            )
        logger.log_startup(f"✓ LoRA adapter directory found: {lora_dir.resolve()}")

        # --- adapter_config.json ---
        adapter_cfg = lora_dir / "adapter_config.json"
        if not adapter_cfg.exists():
            raise FileNotFoundError(
                f"'adapter_config.json' not found inside '{lora_dir.resolve()}'.  "
                "This file is required to load the PEFT LoRA adapter.  "
                "Restore the complete models/gemma_medical_qa_final/ directory."
            )
        logger.log_startup(f"✓ adapter_config.json found: {adapter_cfg.resolve()}")

        # --- Base Generator Model (optional – will be downloaded if missing) ---
        model_path = Path(settings.MODEL_PATH)
        if not model_path.is_dir():
            logger.log_startup(
                f"⚠ Base generator model not found at '{model_path.resolve()}'.  "
                f"It will be automatically downloaded from Hugging Face "
                f"({settings.HF_GENERATOR_MODEL_ID}) on first startup.  "
                "This may take several minutes depending on network speed."
            )
        else:
            logger.log_startup(f"✓ Base generator model found: {model_path.resolve()}")

        # --- Base Embedding Model (optional – will be downloaded if missing) ---
        embed_model_path = Path(settings.EMBEDDING_MODEL_PATH)
        if not embed_model_path.is_dir():
            logger.log_startup(
                f"⚠ Embedding model not found at '{embed_model_path.resolve()}'.  "
                f"It will be automatically downloaded from Hugging Face "
                f"({settings.HF_EMBEDDING_MODEL_ID}) on first startup."
            )
        else:
            logger.log_startup(
                f"✓ Embedding model found: {embed_model_path.resolve()}"
            )

        logger.log_startup("Configuration validation passed. ✓")

    @staticmethod
    def get_device(settings: Settings) -> str:
        """
        Resolve the compute device based on settings and hardware availability.

        If ``settings.DEVICE`` is ``"auto"``, CUDA is selected when available,
        otherwise CPU is used.  Explicit ``"cuda"`` or ``"cpu"`` values bypass
        the auto-detection.

        Args:
            settings: The application settings object.

        Returns:
            A device string suitable for use with PyTorch (``"cuda"`` or ``"cpu"``).
        """
        import torch  # Local import to avoid mandatory torch at module import time

        if settings.DEVICE == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                logger.log_startup(f"CUDA available – using GPU: {gpu_name}")
            else:
                device = "cpu"
                logger.log_startup(
                    "CUDA not available – using CPU.  "
                    "Inference will be significantly slower (~5–10 min/query)."
                )
        else:
            device = settings.DEVICE
            logger.log_startup(f"Device forced to '{device}' via DEVICE setting.")

        return device
