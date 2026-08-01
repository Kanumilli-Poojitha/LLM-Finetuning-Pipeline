"""
model_utils.py

Utilities for loading a quantized (QLoRA-ready) base causal LM + tokenizer,
and for injecting trainable LoRA adapters via the `peft` library.
"""

import logging

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
}


def _bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def build_bnb_config(quant_cfg: dict) -> BitsAndBytesConfig:
    """Builds a BitsAndBytesConfig for 4-bit (QLoRA) quantization from config dict.

    bnb_4bit_compute_dtype: "auto" resolves to bf16 on GPUs that support it
    (Ampere/Ada/Hopper) and float16 elsewhere. An explicit "bfloat16" on
    unsupported hardware (e.g. Turing T4) is auto-corrected to float16 with
    a warning rather than failing.
    """
    requested = quant_cfg.get("bnb_4bit_compute_dtype", "auto")
    if requested == "auto":
        compute_dtype = torch.bfloat16 if _bf16_supported() else torch.float16
    else:
        compute_dtype = _DTYPE_MAP.get(requested, torch.bfloat16)
        if compute_dtype is torch.bfloat16 and not _bf16_supported():
            logger.warning(
                f"bnb_4bit_compute_dtype={requested!r} was requested but this GPU "
                "doesn't support bf16 — falling back to float16."
            )
            compute_dtype = torch.float16

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def resolve_precision(precision: str) -> dict:
    """Resolves the training.precision config value to TrainingArguments-style
    {"fp16": bool, "bf16": bool} flags. "auto" picks bf16 on GPUs that support
    it and float16 elsewhere; an explicit "bfloat16" on unsupported hardware
    is auto-corrected to float16 with a warning rather than failing.
    """
    normalized = precision.lower()
    if normalized == "auto":
        bf16 = _bf16_supported()
        return {"fp16": not bf16, "bf16": bf16}
    if normalized in ("bfloat16", "bf16"):
        if not _bf16_supported():
            logger.warning(f"precision={precision!r} was requested but this GPU "
                            "doesn't support bf16 — falling back to float16.")
            return {"fp16": True, "bf16": False}
        return {"fp16": False, "bf16": True}
    if normalized in ("float16", "fp16"):
        return {"fp16": True, "bf16": False}
    if normalized in ("float32", "fp32"):
        return {"fp16": False, "bf16": False}
    raise ValueError(f"Unrecognized training.precision value: {precision!r}")


def load_model_and_tokenizer(model_id: str, quant_cfg: dict, trust_remote_code: bool = False):
    """
    Loads the tokenizer and the base model in 4-bit quantization.

    Args:
        model_id (str): Hugging Face model repository ID.
        quant_cfg (dict): quantization sub-section of the training config.
        trust_remote_code (bool): whether to trust custom modeling code from
            the Hub repo. Leave False for architectures transformers already
            has built in natively (e.g. Phi-3) — only needed for models that
            ship code transformers doesn't bundle.

    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        # Causal LMs frequently ship without a pad token; reuse EOS.
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding is required for correct batched generation with causal LMs.
    tokenizer.padding_side = "left"

    bnb_config = build_bnb_config(quant_cfg)

    logger.info(f"Loading base model {model_id} in 4-bit (QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def apply_lora_adapters(model, lora_config_dict: dict):
    """
    Prepares the quantized model for training and injects LoRA adapters.

    Args:
        model: The quantized base model.
        lora_config_dict (dict): Dictionary containing r, lora_alpha,
            lora_dropout, bias, task_type, target_modules.

    Returns:
        PeftModel: The model ready for fine-tuning.
    """
    logger.info("Preparing quantized model for k-bit training...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=lora_config_dict.get("r", 16),
        lora_alpha=lora_config_dict.get("lora_alpha", 32),
        lora_dropout=lora_config_dict.get("lora_dropout", 0.05),
        bias=lora_config_dict.get("bias", "none"),
        task_type=lora_config_dict.get("task_type", "CAUSAL_LM"),
        target_modules=lora_config_dict.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    )

    logger.info(f"Injecting LoRA adapters: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
                f"targets={lora_config.target_modules}")
    peft_model = get_peft_model(model, lora_config)
    print_trainable_parameters(peft_model)

    return peft_model


def print_trainable_parameters(model) -> None:
    """Prints the number and percentage of trainable parameters in a PEFT model."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    pct = 100 * trainable_params / all_params if all_params else 0.0
    logger.info(
        f"Trainable params: {trainable_params:,} / {all_params:,} ({pct:.4f}%)"
    )


def load_peft_model_for_inference(base_model_id: str, adapter_path: str, quant_cfg: dict,
                                   trust_remote_code: bool = False):
    """
    Loads the frozen base model + tokenizer, then attaches a trained LoRA
    adapter for inference/evaluation (no merging — adapter stays separate).
    """
    model, tokenizer = load_model_and_tokenizer(base_model_id, quant_cfg, trust_remote_code)
    logger.info(f"Attaching LoRA adapter from {adapter_path}...")
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    peft_model.eval()
    return peft_model, tokenizer


def merge_and_save(base_model_id: str, adapter_path: str, output_dir: str,
                    trust_remote_code: bool = False):
    """
    Loads the base model in full precision, applies the LoRA adapter, and
    bakes the adapter weights into the base weights permanently via
    `merge_and_unload()`, saving a single standalone model for deployment.

    Note: this loads the base model WITHOUT 4-bit quantization, since
    merging requires full-precision (fp16/bf16) weights.
    """
    logger.info(f"Loading full-precision base model {base_model_id} for merging...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)

    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    logger.info("Merging LoRA adapter into base weights...")
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Merged model saved to {output_dir}")
