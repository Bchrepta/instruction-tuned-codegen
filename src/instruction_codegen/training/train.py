"""LoRA SFT training for instruction-tuned code generation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)

from instruction_codegen.config import load_config, project_root
from instruction_codegen.data.packing import (
    build_packed_dataloader,
    estimate_naive_utilization,
)

logger = logging.getLogger(__name__)


def _dtype_from_cfg(precision: str) -> torch.dtype:
    precision = precision.lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _resolve_precision(cfg: dict) -> tuple[str, torch.dtype]:
    requested = str(cfg.get("precision", "bf16")).lower()
    wants_gpu = bool(cfg.get("load_in_4bit")) or requested in {"bf16", "fp16"}
    if not torch.cuda.is_available():
        if wants_gpu:
            raise RuntimeError(
                "This config expects a CUDA GPU, but torch.cuda.is_available() is False. "
                "You likely installed a CPU-only PyTorch build. In your venv run:\n"
                "  python -c \"import torch; print(torch.__version__, torch.version.cuda)\"\n"
                "Then install a CUDA build, e.g.:\n"
                "  pip uninstall -y torch\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
                "Confirm with nvidia-smi and torch.cuda.is_available() before training again."
            )
        logger.warning("CUDA unavailable; falling back to fp32")
        return "fp32", torch.float32
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        logger.warning("bf16 not supported on this GPU; falling back to fp16")
        return "fp16", torch.float16
    return requested, _dtype_from_cfg(requested)


def build_model_and_tokenizer(cfg: dict):
    model_name = cfg["model_name_or_path"]
    precision, dtype = _resolve_precision(cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_4bit = bool(cfg.get("load_in_4bit")) and torch.cuda.is_available()
    model_kwargs: dict = {
        "torch_dtype": dtype if precision != "fp32" else torch.float32,
    }
    if use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
        model_kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if torch.cuda.is_available() and not use_4bit:
        model = model.to("cuda")

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        )
    elif cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=list(lora_cfg["target_modules"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer, precision


def train(cfg: dict) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    root = project_root()
    dataset_path = Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, precision = build_model_and_tokenizer(cfg)
    max_seq = int(cfg["max_seq_length"])
    batch_size = int(cfg["per_device_train_batch_size"])
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))

    use_packing = bool(cfg.get("use_packing", True))
    if use_packing:
        train_loader, pack_stats = build_packed_dataloader(
            dataset_path,
            tokenizer,
            max_seq_length=max_seq,
            batch_size=batch_size,
            shuffle=True,
        )
        naive_util = estimate_naive_utilization(dataset_path, tokenizer, max_seq)
        logger.info(
            "Packing utilization=%.1f%% (naive pad=%.1f%%) sequences=%d",
            100 * pack_stats.utilization,
            100 * naive_util,
            pack_stats.num_packed_sequences,
        )
    else:
        train_loader, pack_stats = build_packed_dataloader(
            dataset_path,
            tokenizer,
            max_seq_length=max_seq,
            batch_size=batch_size,
            shuffle=True,
        )
        # Still pack for simplicity when flag off would need alternate path;
        # benchmark script compares packing vs naive.
        naive_util = estimate_naive_utilization(dataset_path, tokenizer, max_seq)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    epochs = int(cfg.get("num_train_epochs", 1))
    max_steps = cfg.get("max_steps")
    steps_per_epoch = max(1, len(train_loader))
    total_update_steps = epochs * ((steps_per_epoch + grad_accum - 1) // grad_accum)
    if max_steps is not None:
        total_update_steps = min(total_update_steps, int(max_steps))

    scheduler = get_scheduler(
        cfg.get("lr_scheduler_type", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=int(total_update_steps * float(cfg.get("warmup_ratio", 0.03))),
        num_training_steps=total_update_steps,
    )

    use_amp = precision in {"fp16", "bf16"} and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    if torch.cuda.is_available():
        scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and use_amp))
    else:
        class _NoScaler:
            def is_enabled(self):
                return False
            def scale(self, loss):
                return loss
            def unscale_(self, opt):
                return None
            def step(self, opt):
                opt.step()
            def update(self):
                return None
        scaler = _NoScaler()
    device = next(model.parameters()).device

    model.train()
    global_step = 0
    running_loss = 0.0
    t0 = time.perf_counter()
    log_every = int(cfg.get("logging_steps", 10))
    save_every = int(cfg.get("save_steps", 200))

    done = False
    for epoch in range(epochs):
        if done:
            break
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if use_amp:
                amp_ctx = torch.amp.autocast("cuda", dtype=amp_dtype)
            else:
                from contextlib import nullcontext

                amp_ctx = nullcontext()
            with amp_ctx:
                outputs = model(**batch)
                loss = outputs.loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    avg = running_loss / log_every
                    running_loss = 0.0
                    logger.info(
                        "epoch=%d step=%d loss=%.4f lr=%.2e",
                        epoch,
                        global_step,
                        avg,
                        scheduler.get_last_lr()[0],
                    )

                if save_every > 0 and global_step % save_every == 0:
                    ckpt = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)

                if max_steps is not None and global_step >= int(max_steps):
                    done = True
                    break

    elapsed = time.perf_counter() - t0
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "steps": global_step,
        "elapsed_sec": round(elapsed, 2),
        "precision": precision,
        "packing": pack_stats.as_dict() if pack_stats else None,
        "naive_pad_utilization": round(naive_util, 4),
        "model_name_or_path": cfg["model_name_or_path"],
        "output_dir": str(output_dir),
        "max_seq_length": max_seq,
        "lora": cfg["lora"],
    }
    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved adapter to %s", output_dir)
    logger.info("Metrics: %s", metrics)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="LoRA SFT for instruction-tuned codegen")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    # Ensure src/ is importable when run as script
    root = project_root()
    src = str(root / "src")
    if src not in os.sys.path:
        os.sys.path.insert(0, src)

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
