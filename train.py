from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import Dinov2Model


HERE = Path(__file__).resolve().parent
FIELDS = [
    "tongue_color", "tongue_shape_indentation", "tongue_shape_form", "tongue_shape_stab",
    "tongue_shape_crack", "tongue_shape_ecchymosis", "coating_nature_peel",
    "coating_nature_moistening", "coating_nature_decay", "coating_nature_thickness",
    "coating_colour", "tongue_color_tipside",
]
CLASS_CODES = {
    "tongue_color": [1, 2, 3, 4, 5], "tongue_shape_indentation": [1, 2],
    "tongue_shape_form": [1, 2, 3], "tongue_shape_stab": [1, 2],
    "tongue_shape_crack": [1, 2], "tongue_shape_ecchymosis": [1, 2],
    "coating_nature_peel": [1, 2, 3, 4], "coating_nature_moistening": [1, 2, 3],
    "coating_nature_decay": [1, 2, 3], "coating_nature_thickness": [1, 2],
    "coating_colour": [1, 2, 3], "tongue_color_tipside": [1, 2, 3, 4],
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def atomic_save(value: object, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_status(path: Path, **values: object) -> None:
    clean = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
             for key, value in values.items()}
    path.write_text(json.dumps({**clean, "updated_at_unix": time.time()}, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TongueDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, release_root: Path, size: int, train: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = release_root
        aug: list[nn.Module] = [transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC)]
        if train:
            aug.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(5, interpolation=InterpolationMode.BILINEAR, fill=(0, 0, 0)),
                transforms.ColorJitter(brightness=0.05, saturation=0.05, hue=0.01),
            ])
        aug.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        self.transform = transforms.Compose(aug)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        relative = Path(str(row["image_path"]).replace("\\", "/"))
        with Image.open(self.root / "seg" / relative) as image:
            pixels = self.transform(image.convert("RGB"))
        labels = torch.tensor([int(row[field]) - 1 for field in FIELDS], dtype=torch.long)
        return pixels, labels


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def add_lora(model: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacements = []
    targets = ("attention.attention.query", "attention.attention.key", "attention.attention.value", "mlp.fc1", "mlp.fc2")
    for full_name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and full_name.endswith(targets):
            parent_name, _, child_name = full_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            replacements.append((parent, child_name, module))
    if not replacements:
        raise RuntimeError("No DINOv2 linear layers matched the LoRA targets.")
    for parent, name, base in replacements:
        parent._modules[name] = LoRALinear(base, rank, alpha, dropout)
    return len(replacements)


class TongueClassifier(nn.Module):
    def __init__(self, base: Dinov2Model, rank: int, alpha: float, dropout: float,
                 trained_codes: dict[str, list[int]]) -> None:
        super().__init__()
        self.backbone = base
        self.num_lora_layers = add_lora(self.backbone, rank, alpha, dropout)
        hidden = int(base.config.hidden_size)
        self.projection = nn.Sequential(nn.Linear(hidden, 512), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.heads = nn.ModuleList([nn.Linear(512, len(trained_codes[field])) for field in FIELDS])

    def forward(self, pixels: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.backbone(pixel_values=pixels, interpolate_pos_encoding=True).last_hidden_state
        pooled = hidden[:, 1:, :].mean(dim=1)
        projected = self.projection(pooled)
        return [head(projected) for head in self.heads]


def build_model(model_dir: Path, device: torch.device, rank: int, alpha: float, dropout: float,
                trained_codes: dict[str, list[int]]):
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"DINOv2 weights not found: {model_dir}")
    model = TongueClassifier(Dinov2Model.from_pretrained(str(model_dir), local_files_only=True), rank, alpha, dropout,
                             trained_codes)
    return model.to(device)


def load_split(path: Path, release_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": str, "image_path": str})
    required = ["id", "image_path", *FIELDS]
    missing = [field for field in required if field not in frame]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    for field in FIELDS:
        values = pd.to_numeric(frame[field], errors="coerce")
        if values.isna().any() or ((values < 1) | (values > len(CLASS_CODES[field])) | (values % 1 != 0)).any():
            raise ValueError(f"{path}: invalid category code in {field}")
        frame[field] = values.astype("int64")
    missing_images = [p for p in frame.image_path if not (release_root / "seg" / str(p)).is_file()]
    if missing_images:
        raise FileNotFoundError(f"{path}: {len(missing_images)} missing images; first: {missing_images[0]}")
    return frame


def audit_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    for key in ("id", "image_path"):
        sets = [set(frame[key].astype(str)) for frame in (train, val, test)]
        for a, b, name in ((0, 1, "train/val"), (0, 2, "train/test"), (1, 2, "val/test")):
            overlap = sets[a] & sets[b]
            if overlap:
                raise ValueError(f"Split leakage in {key} ({name}): {len(overlap)} overlaps")


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool):
    model.eval()
    all_logits: list[list[np.ndarray]] = [[] for _ in FIELDS]
    labels: list[np.ndarray] = []
    for pixels, y in loader:
        pixels = pixels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(pixels)
        for i, value in enumerate(logits):
            all_logits[i].append(value.float().cpu().numpy())
        labels.append(y.numpy())
    return [np.concatenate(x) for x in all_logits], np.concatenate(labels)


def calculate_metrics(logits: list[np.ndarray], labels_zero_based: np.ndarray,
                      trained_codes: dict[str, list[int]]) -> dict:
    per_field = {}
    f1s, aps = [], []
    for i, field in enumerate(FIELDS):
        y = labels_zero_based[:, i]
        observed_probability = torch.softmax(torch.from_numpy(logits[i]), dim=1).numpy()
        probability = np.zeros((len(y), len(CLASS_CODES[field])), dtype=np.float32)
        for column, code in enumerate(trained_codes[field]):
            probability[:, code - 1] = observed_probability[:, column]
        pred = probability.argmax(axis=1)
        classes_present = sorted(set(y.tolist()))
        macro_f1 = float(f1_score(y, pred, labels=classes_present, average="macro", zero_division=0))
        class_ap = {}
        for class_id in range(probability.shape[1]):
            binary = (y == class_id).astype(np.uint8)
            if binary.sum() > 0 and binary.sum() < len(binary):
                class_ap[str(class_id + 1)] = float(average_precision_score(binary, probability[:, class_id]))
        field_ap = float(np.mean(list(class_ap.values()))) if class_ap else float("nan")
        f1s.append(macro_f1)
        if np.isfinite(field_ap):
            aps.append(field_ap)
        per_field[field] = {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1_present_classes": macro_f1,
            "macro_auprc_present_classes_ovr": field_ap,
            "support_by_code": {str(code + 1): int((y == code).sum()) for code in range(probability.shape[1])},
            "per_code_f1": {str(code + 1): float(f1_score(y == code, pred == code, zero_division=0))
                             for code in range(probability.shape[1])},
            "per_code_auprc": class_ap,
        }
    return {"macro_field_f1": float(np.mean(f1s)),
            "macro_field_auprc_ovr": float(np.mean(aps)) if aps else float("nan"),
            "per_field": per_field}


@torch.no_grad()
def update_ema(model: nn.Module, state: dict[str, torch.Tensor], decay: float) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            state[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)


class EMAWeights:
    def __init__(self, model: nn.Module, state: dict[str, torch.Tensor]) -> None:
        self.model, self.state, self.backup = model, state, {}

    def __enter__(self):
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    self.backup[name] = parameter.detach().clone()
                    parameter.copy_(self.state[name])
        return self.model

    def __exit__(self, *_):
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name in self.backup:
                    parameter.copy_(self.backup[name])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--model-dir", type=Path, default=HERE.parents[1] / "models" / "dinov2-small")
    parser.add_argument("--output-dir", type=Path, default=HERE.parents[1] / "runs" / "12fields")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--max-class-weight", type=float, default=20.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, help="Resume from a latest_training.pt checkpoint")
    args = parser.parse_args()

    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the training run requires the local RTX 4060 GPU.")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {torch.cuda.get_device_name(device)} | VRAM: {props.total_memory / 1024**3:.2f} GiB")

    data_dir = args.data_dir.resolve()
    train = load_split(data_dir / f"fold{args.fold}_train.csv", args.release_root)
    val = load_split(data_dir / f"fold{args.fold}_val.csv", args.release_root)
    test = load_split(data_dir / f"fold{args.fold}_test.csv", args.release_root)
    audit_splits(train, val, test)
    print(f"Split sizes | train={len(train)} val={len(val)} test={len(test)}")
    trained_codes: dict[str, list[int]] = {}
    for field in FIELDS:
        counts = train[field].value_counts().reindex(CLASS_CODES[field], fill_value=0)
        print(f"Train support {field}: {counts.to_dict()}")
        trained_codes[field] = [int(code) for code, count in counts.items() if count > 0]
        unseen = [int(code) for code, count in counts.items() if count == 0]
        if unseen:
            print(f"WARNING: {field} codes {unseen} are absent from training and cannot be learned in this fold.")

    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    size = int(cfg["model"]["input_size"])
    datasets = [TongueDataset(frame, args.release_root, size, train=(i == 0))
                for i, frame in enumerate((train, val, test))]
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=(i == 0), num_workers=args.workers,
                          pin_memory=True, persistent_workers=args.workers > 0)
               for i, ds in enumerate(datasets)]
    model = build_model(args.model_dir, device, int(cfg["model"]["lora_rank"]),
                        float(cfg["model"]["lora_alpha"]), float(cfg["model"]["lora_dropout"]), trained_codes)
    trainable = [p for p in model.parameters() if p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    print(f"LoRA modules={model.num_lora_layers}; trainable parameters={sum(p.numel() for p in trainable):,}")

    class_weights = []
    target_maps = []
    for field in FIELDS:
        counts = train[field].value_counts().reindex(trained_codes[field]).to_numpy(dtype=np.float32)
        weights = len(train) / (len(counts) * counts)
        class_weights.append(torch.tensor(np.minimum(weights, args.max_class_weight), device=device, dtype=torch.float32))
        code_to_index = {code: index for index, code in enumerate(trained_codes[field])}
        target_maps.append(torch.tensor([code_to_index.get(code, -1) for code in CLASS_CODES[field]],
                                         device=device, dtype=torch.long))
    ema = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([{"params": lora_params, "lr": args.lora_lr},
                                   {"params": head_params, "lr": args.head_lr}], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run_dir = args.output_dir / f"fold{args.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, stale = -float("inf"), 0, 0
    best_path = run_dir / "best_trainables.pt"
    latest_path = run_dir / "latest_training.pt"
    status_path = run_dir / "status.json"
    history: list[dict] = []
    first_epoch = 1

    if args.resume:
        previous = torch.load(args.resume, map_location=device, weights_only=False)
        if previous["fold"] != args.fold or previous["fields"] != FIELDS or previous["trained_class_codes"] != trained_codes:
            raise ValueError("Resume checkpoint does not match this fold and label mapping.")
        _, unexpected = model.load_state_dict(previous["trainable_state"], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected resume keys: {unexpected[:5]}")
        ema = {name: value.to(device) for name, value in previous["ema_state"].items()}
        optimizer.load_state_dict(previous["optimizer_state"])
        scheduler.load_state_dict(previous["scheduler_state"])
        scaler.load_state_dict(previous["scaler_state"])
        best_score, best_epoch, stale = previous["best_score"], previous["best_epoch"], previous["stale"]
        history = previous["history"]
        first_epoch = previous["epoch"] + 1
        random.setstate(previous["python_rng_state"])
        np.random.set_state(previous["numpy_rng_state"])
        torch.set_rng_state(previous["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in previous["cuda_rng_states"]])
        print(f"Resumed after epoch {previous['epoch']} from {args.resume}")

    write_status(status_path, state="running", epoch=first_epoch, step=0,
                 total_steps=len(loaders[0]), best_epoch=best_epoch, best_score=best_score)

    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, seen, started = 0.0, 0, time.time()
        progress = tqdm(loaders[0], desc=f"12fields fold {args.fold} epoch {epoch}/{args.epochs}",
                        leave=False, disable=not sys.stderr.isatty())
        for step, (pixels, labels) in enumerate(progress, start=1):
            pixels, labels = pixels.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(pixels)
                targets = torch.stack([target_maps[i][labels[:, i]] for i in range(len(FIELDS))], dim=1)
                raw_loss = sum(nn.functional.cross_entropy(logits[i].float(), targets[:, i], weight=class_weights[i])
                               for i in range(len(FIELDS))) / len(FIELDS)
                group_start = ((step - 1) // args.grad_accum) * args.grad_accum
                group_size = min(args.grad_accum, len(loaders[0]) - group_start)
                loss = raw_loss / group_size
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0 or step == len(loaders[0]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(model, ema, args.ema_decay)
            total_loss += float(raw_loss.detach()) * len(labels)
            seen += len(labels)
            progress.set_postfix(loss=f"{float(raw_loss.detach()):.4f}")
            if step % 50 == 0 or step == len(loaders[0]):
                write_status(status_path, state="running", epoch=epoch, step=step,
                             total_steps=len(loaders[0]), train_loss_so_far=total_loss / seen,
                             best_epoch=best_epoch, best_score=best_score)
        scheduler.step(epoch - 1)
        with EMAWeights(model, ema):
            val_logits, val_y = collect_logits(model, loaders[1], device, amp)
        val_metrics = calculate_metrics(val_logits, val_y, trained_codes)
        score = val_metrics["macro_field_auprc_ovr"]
        row = {"epoch": epoch, "train_loss": total_loss / max(seen, 1),
               "val_macro_field_f1": val_metrics["macro_field_f1"],
               "val_macro_field_auprc_ovr": val_metrics["macro_field_auprc_ovr"],
               "seconds": time.time() - started}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            atomic_save({
                "trainable_state": {name: value.detach().cpu() for name, value in ema.items()},
                "fields": FIELDS, "class_codes": CLASS_CODES, "trained_class_codes": trained_codes, "fold": args.fold,
                "best_epoch": epoch, "best_val_metrics": val_metrics,
                "config": cfg,
                "cli": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "train_class_counts": {field: train[field].value_counts().sort_index().to_dict() for field in FIELDS},
            }, best_path)
        else:
            stale += 1
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        atomic_save({
            "epoch": epoch, "fold": args.fold, "fields": FIELDS, "trained_class_codes": trained_codes,
            "trainable_state": {name: value.detach().cpu() for name, value in model.named_parameters()
                                if value.requires_grad},
            "ema_state": {name: value.detach().cpu() for name, value in ema.items()},
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(), "best_score": best_score, "best_epoch": best_epoch,
            "stale": stale, "history": history, "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(), "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
        }, latest_path)
        write_status(status_path, state="running", epoch=epoch, step=len(loaders[0]),
                     total_steps=len(loaders[0]), best_epoch=best_epoch, best_score=best_score,
                     val_macro_field_f1=row["val_macro_field_f1"],
                     val_macro_field_auprc_ovr=row["val_macro_field_auprc_ovr"])
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, val macro field AUPRC={best_score:.4f}")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    _, unexpected = model.load_state_dict(checkpoint["trainable_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:5]}")
    trained_codes = checkpoint["trained_class_codes"]
    test_logits, test_y = collect_logits(model, loaders[2], device, amp)
    result = {"fold": args.fold, "fields": FIELDS, "class_codes": CLASS_CODES,
              "best_epoch": best_epoch, "best_val_macro_field_auprc_ovr": best_score,
              "trained_class_codes": trained_codes,
              "validation": checkpoint["best_val_metrics"], "test": calculate_metrics(test_logits, test_y, trained_codes),
              "test_used_for_selection": False, "config": cfg, "cli": checkpoint["cli"]}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"fold": args.fold, "best_epoch": best_epoch,
                      "validation_macro_field_f1": result["validation"]["macro_field_f1"],
                      "validation_macro_field_auprc_ovr": best_score,
                      "test_macro_field_f1": result["test"]["macro_field_f1"],
                      "test_macro_field_auprc_ovr": result["test"]["macro_field_auprc_ovr"]}, indent=2))
    print(f"Saved run to {run_dir.resolve()}")
    write_status(status_path, state="complete", epoch=history[-1]["epoch"], best_epoch=best_epoch,
                 best_score=best_score, test_macro_field_f1=result["test"]["macro_field_f1"])


if __name__ == "__main__":
    main()
