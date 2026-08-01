"""
train.py

Main training entrypoint. Loads config from YAML, prepares the quantized
model + LoRA adapters (with mixed precision auto-detected for whatever GPU
is present), wraps everything in trl.SFTTrainer, and runs supervised
fine-tuning with TensorBoard tracking and periodic checkpointing.

Run as a module from the project root so imports resolve correctly:
    python -m src.train --config configs/training_config.yaml

CLI flags let you override individual hyperparameters for quick
experiments; with no flags passed, every value comes purely from the YAML
config, which remains the single source of truth.
    python -m src.train --config configs/training_config.yaml --learning_rate 1e-4 --lora_r 32
"""

import argparse
import logging
import os

import torch
import yaml
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from src.model_utils import apply_lora_adapters, load_model_and_tokenizer, resolve_precision

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Defensive guard: even though report_to defaults to "tensorboard", make sure
# no dependency (transformers, trl) tries to auto-initialize a W&B run.
os.environ.setdefault("WANDB_DISABLED", "true")


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Lets command-line flags override individual YAML values without editing the file.
    With no flags passed, config is used exactly as loaded from YAML."""
    overrides = {
        "learning_rate": ("training", "learning_rate"),
        "batch_size": ("training", "per_device_train_batch_size"),
        "grad_accum": ("training", "gradient_accumulation_steps"),
        "num_epochs": ("training", "num_train_epochs"),
        "max_steps": ("training", "max_steps"),
        "lora_r": ("lora", "r"),
        "lora_alpha": ("lora", "lora_alpha"),
        "output_dir": ("training", "output_dir"),
    }
    for arg_name, (section, key) in overrides.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config[section][key] = value
    return config


def build_training_arguments(config: dict) -> SFTConfig:
    t_cfg = config["training"]
    ds_cfg = config["dataset"]
    precision = resolve_precision(t_cfg.get("precision", "auto"))

    return SFTConfig(
        output_dir=t_cfg["output_dir"],
        logging_dir=os.path.join(t_cfg["output_dir"], "runs"),
        num_train_epochs=t_cfg["num_train_epochs"],
        max_steps=t_cfg.get("max_steps", -1),
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        learning_rate=float(t_cfg["learning_rate"]),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.03),
        optim=t_cfg.get("optim", "paged_adamw_8bit"),
        weight_decay=t_cfg.get("weight_decay", 0.01),
        max_grad_norm=t_cfg.get("max_grad_norm", 0.3),
        fp16=precision["fp16"],
        bf16=precision["bf16"],
        logging_steps=t_cfg.get("logging_steps", 10),
        logging_strategy="steps",
        eval_strategy=t_cfg.get("eval_strategy", "steps"),
        eval_steps=t_cfg.get("eval_steps", 50),
        save_strategy=t_cfg.get("save_strategy", "steps"),
        save_steps=t_cfg.get("save_steps", 50),
        save_total_limit=t_cfg.get("save_total_limit", 3),
        load_best_model_at_end=t_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=t_cfg.get("metric_for_best_model", "eval_loss"),
        seed=t_cfg.get("seed", 42),
        report_to=t_cfg.get("report_to", "tensorboard"),
        remove_unused_columns=False,
        dataset_text_field=ds_cfg.get("text_field", "text"),
        max_length=ds_cfg["max_seq_length"],
        packing=False,
    )


def main(args: argparse.Namespace):
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    ds_cfg = config["dataset"]
    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    lora_cfg = config["lora"]
    t_cfg = config["training"]

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is visible — training cannot proceed.\n"
            "  Colab: Runtime > Change runtime type > GPU, then rerun this cell.\n"
            "  Local/Windows: confirm `nvidia-smi` works and your driver is current.\n"
            "  Docker: confirm Docker Desktop uses the WSL2 backend and verify with:\n"
            "    docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi"
        )

    logger.info(f"Tracking via: {t_cfg.get('report_to', 'tensorboard')} "
                f"(logs under {t_cfg['output_dir']}/runs)")

    # --- Load datasets --------------------------------------------------------
    processed_dir = ds_cfg["processed_dir"]
    train_file = os.path.join(processed_dir, "train.jsonl")
    val_file = os.path.join(processed_dir, "validation.jsonl")
    if not (os.path.exists(train_file) and os.path.exists(val_file)):
        raise FileNotFoundError(
            f"Processed splits not found in {processed_dir}. "
            f"Run `python -m src.data_pipeline --config {args.config}` first."
        )

    logger.info("Loading train/validation splits...")
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    val_dataset = load_dataset("json", data_files=val_file, split="train")

    # data_pipeline.py saves "prompt" and "target" alongside "text" for
    # evaluate.py's later ROUGE scoring (which reloads test.jsonl on its
    # own). TRL 1.9.2's SFTTrainer decides the dataset format by checking
    # only whether a "prompt" key is present — not whether "completion" is
    # also present — so leaving that column in makes it wrongly assume a
    # prompt-completion dataset and then KeyError on the "completion" field
    # that was never there. Drop every column except the text field so the
    # trainer takes the plain text-only SFT path instead.
    text_field = ds_cfg.get("text_field", "text")
    train_dataset = train_dataset.remove_columns(
        [c for c in train_dataset.column_names if c != text_field]
    )
    val_dataset = val_dataset.remove_columns(
        [c for c in val_dataset.column_names if c != text_field]
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(
            f"Train or validation split is empty (train={len(train_dataset)}, "
            f"val={len(val_dataset)}). Re-run data_pipeline.py."
        )
    logger.info(f"Train: {len(train_dataset)} examples | Validation: {len(val_dataset)} examples")

    # --- Load model + tokenizer, apply LoRA -----------------------------------
    model, tokenizer = load_model_and_tokenizer(
        model_cfg["base_model_id"], quant_cfg, model_cfg.get("trust_remote_code", False)
    )
    peft_model = apply_lora_adapters(model, lora_cfg)

    # --- Training arguments ----------------------------------------------------
    training_args = build_training_arguments(config)

    # --- SFTTrainer --------------------------------------------------------------
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            "CUDA out of memory during training. Try, in order: (1) lower "
            "training.per_device_train_batch_size to 1 in the config and "
            "raise gradient_accumulation_steps to compensate, (2) lower "
            "dataset.max_seq_length, (3) lower lora.r. "
            f"Original error: {e}"
        ) from e

    final_dir = t_cfg["final_model_dir"]
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    adapter_weights_path_safetensors = os.path.join(final_dir, "adapter_model.safetensors")
    adapter_weights_path_bin = os.path.join(final_dir, "adapter_model.bin")
    if not (os.path.exists(adapter_weights_path_safetensors) or os.path.exists(adapter_weights_path_bin)):
        logger.warning(
            f"Expected adapter weights not found in {final_dir} after saving — "
            f"verify trainer.save_model() completed successfully."
        )
    else:
        logger.info(f"Final adapter + tokenizer saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with LoRA/QLoRA + SFTTrainer.")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None, dest="batch_size")
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--num_epochs", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    main(args)
