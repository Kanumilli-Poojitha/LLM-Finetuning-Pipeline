"""
data_pipeline.py

Loads a raw instruction-tuning dataset from the Hugging Face Hub, formats
every example using the base model's own chat template (falling back to a
generic ChatML template for base/non-instruct models with no chat template
defined), enforces a maximum sequence length, and produces a strict
80/10/10 train/val/test split saved to disk as JSONL files.

Each saved row contains three fields:
  - "text":   full formatted conversation (prompt + response) — what
              SFTTrainer trains on.
  - "prompt": formatted conversation up to (and including) the
              generation-prompt marker, but WITHOUT the response — used by
              evaluate.py to generate predictions.
  - "target": the raw ground-truth response text — used as the reference
              for ROUGE scoring.

Usage:
    python -m src.data_pipeline --config configs/training_config.yaml
"""

import argparse
import logging
import os
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _build_user_turn(example: dict) -> str:
    instruction = (example.get("instruction") or "").strip()
    context = (example.get("context") or "").strip()
    if context:
        return f"{instruction}\n\nContext:\n{context}"
    return instruction


def format_chatml(example: dict, tokenizer, system_prompt: str) -> dict:
    """
    Formats a single dataset example using the TOKENIZER'S OWN chat template
    (tokenizer.apply_chat_template). This is important: hand-writing generic
    "<|im_start|>"/"<|im_end|>" tags only works for models actually trained
    on that literal ChatML format (e.g. Mistral/Zephyr-style models). Most
    instruct models — including Phi-3 — use their own turn-delimiter tokens,
    and feeding them the wrong ones means those tokens get split into
    ordinary subwords instead of being treated as atomic turn boundaries.
    That silently degrades training and is a common cause of the model
    never learning to emit its real end-of-turn/EOS token (endless
    generation at inference time).

    If the tokenizer has no chat template configured (e.g. a raw base,
    non-instruct model), falls back to a generic ChatML template.

    Args:
        example (dict): raw example with 'instruction', 'context', 'response' keys.
        tokenizer: the model's tokenizer (already loaded).
        system_prompt (str): system instruction to prepend.

    Returns:
        dict: {"text": <full convoincl. response>,
               "prompt": <convo up to generation point, no response>,
               "target": <raw ground-truth response text>}
    """
    user_turn = _build_user_turn(example)
    response = (example.get("response") or example.get("output") or "").strip()

    if getattr(tokenizer, "chat_template", None):
        messages_no_response = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_turn},
        ]
        prompt = tokenizer.apply_chat_template(
            messages_no_response, tokenize=False, add_generation_prompt=True
        )
        full_messages = messages_no_response + [{"role": "assistant", "content": response}]
        text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
    else:
        # Fallback: generic ChatML for models with no defined chat template.
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_turn}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        text = f"{prompt}{response}<|im_end|>"

    return {"text": text, "prompt": prompt, "target": response}


def enforce_max_length(example: dict, tokenizer, max_seq_length: int) -> dict:
    """
    Tokenizes the formatted text solely to measure length and flag oversized
    sequences. We keep the *string* fields (not token ids) in the saved
    dataset — SFTTrainer tokenizes on the fly — but pre-filter here so
    oversized examples never reach the trainer and trigger OOM errors.
    """
    token_ids = tokenizer(example["text"], truncation=False)["input_ids"]
    example["num_tokens"] = len(token_ids)
    example["within_limit"] = len(token_ids) <= max_seq_length
    return example


def prepare_and_split_dataset(config: dict) -> DatasetDict:
    """
    Loads, formats, filters, and splits the dataset per the config.

    Steps:
      1. Load dataset using HF `datasets`.
      2. Load the tokenizer for the configured base model (so formatting
         matches that model's actual chat template).
      3. Apply `format_chatml` via `dataset.map()`.
      4. Tokenize-and-measure to drop examples over max_seq_length.
      5. Split into train/val/test using `dataset.train_test_split()`.
      6. Save each split to disk as JSONL under `processed_dir`.

    Returns:
        DatasetDict: the final {"train":..., "validation":..., "test":...} splits.

    Raises:
        RuntimeError: on dataset download failures, with an actionable message.
        ValueError: if the dataset (before or after filtering) is smaller
            than the configured minimum.
    """
    ds_cfg = config["dataset"]
    model_cfg = config["model"]

    logger.info(f"Loading dataset: {ds_cfg['name_or_path']}")
    raw_path = Path(ds_cfg["name_or_path"])
    try:
        if raw_path.exists() and raw_path.suffix in {".json", ".jsonl"}:
            raw_dataset = load_dataset("json", data_files=str(raw_path), split="train")
        else:
            raw_dataset = load_dataset(ds_cfg["name_or_path"], split="train")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset '{ds_cfg['name_or_path']}'. Check your "
            f"internet connection (the container needs outbound HTTPS access "
            f"to huggingface.co), that the dataset ID is spelled correctly, "
            f"and — if it's a gated dataset — that HUGGING_FACE_HUB_TOKEN is "
            f"set in .env. Original error: {e}"
        ) from e

    logger.info(f"Raw dataset size: {len(raw_dataset)} examples")
    if len(raw_dataset) < ds_cfg["min_examples"]:
        raise ValueError(
            f"Dataset has only {len(raw_dataset)} examples; "
            f"a minimum of {ds_cfg['min_examples']} is required."
        )

    logger.info("Loading tokenizer (for chat-template-aware formatting + length filtering)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["base_model_id"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
    except OSError as e:
        raise RuntimeError(
            f"Failed to load tokenizer for '{model_cfg['base_model_id']}'. If "
            f"this is a gated model, accept its license on the Hugging Face "
            f"Hub and set HUGGING_FACE_HUB_TOKEN in .env. Original error: {e}"
        ) from e
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Formatting examples using the model's chat template...")
    formatted = raw_dataset.map(
        lambda ex: format_chatml(ex, tokenizer, ds_cfg["system_prompt"]),
        desc="Formatting chat template",
    )

    measured = formatted.map(
        lambda ex: enforce_max_length(ex, tokenizer, ds_cfg["max_seq_length"]),
        desc="Measuring sequence length",
    )

    before = len(measured)
    measured = measured.filter(lambda ex: ex["within_limit"])
    after = len(measured)
    logger.info(
        f"Dropped {before - after} examples exceeding max_seq_length="
        f"{ds_cfg['max_seq_length']} ({after} remaining)."
    )
    if after < ds_cfg["min_examples"]:
        raise ValueError(
            f"After length filtering only {after} examples remain, below the "
            f"required minimum of {ds_cfg['min_examples']}. Consider raising "
            f"max_seq_length or choosing a different dataset."
        )

    # Drop helper columns before saving
    measured = measured.remove_columns(
        [c for c in ["num_tokens", "within_limit"] if c in measured.column_names]
    )

    logger.info("Splitting into train/val/test...")
    val_test_fraction = ds_cfg["val_split"] + ds_cfg["test_split"]
    split_1 = measured.train_test_split(test_size=val_test_fraction, seed=ds_cfg["seed"])
    train_split = split_1["train"]

    relative_test_fraction = ds_cfg["test_split"] / val_test_fraction
    split_2 = split_1["test"].train_test_split(
        test_size=relative_test_fraction, seed=ds_cfg["seed"]
    )
    val_split = split_2["train"]
    test_split = split_2["test"]

    dataset_dict = DatasetDict(
        {"train": train_split, "validation": val_split, "test": test_split}
    )

    logger.info(
        f"Split sizes -> train: {len(train_split)}, "
        f"validation: {len(val_split)}, test: {len(test_split)}"
    )

    output_dir = Path(ds_cfg["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_ds in dataset_dict.items():
        out_path = output_dir / f"{split_name}.jsonl"
        split_ds.to_json(str(out_path), orient="records", lines=True)
        logger.info(f"Saved {split_name} split ({len(split_ds)} rows) -> {out_path}")

    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="Prepare and split a dataset for SFT/LoRA fine-tuning.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to the YAML training configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(config["dataset"]["raw_dir"], exist_ok=True)
    prepare_and_split_dataset(config)


if __name__ == "__main__":
    main()
