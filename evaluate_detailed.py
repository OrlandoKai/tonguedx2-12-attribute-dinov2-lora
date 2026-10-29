from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

import train as pipeline


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RELEASE = HERE.parents[3]


def evaluate_split(model, frame, release_root: Path, batch_size: int, device,
                   trained_codes: dict[str, list[int]], input_size: int) -> dict:
    dataset = pipeline.TongueDataset(frame, release_root, input_size, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logits, labels = pipeline.collect_logits(model, loader, device, amp=True)
    details: dict[str, dict] = {}
    for column, field in enumerate(pipeline.FIELDS):
        actual = labels[:, column]
        observed_probabilities = torch.softmax(torch.from_numpy(logits[column]), dim=1).numpy()
        probabilities = np.zeros((len(frame), len(pipeline.CLASS_CODES[field])), dtype=np.float32)
        for trained_column, code in enumerate(trained_codes[field]):
            probabilities[:, code - 1] = observed_probabilities[:, trained_column]
        predicted = probabilities.argmax(axis=1)
        present = sorted(set(actual.tolist()))
        p, r, f1, _ = precision_recall_fscore_support(
            actual, predicted, labels=present, average="macro", zero_division=0)
        per_class: dict[str, dict] = {}
        aps = []
        for code in pipeline.CLASS_CODES[field]:
            class_index = code - 1
            binary_actual = actual == class_index
            binary_predicted = predicted == class_index
            class_p, class_r, class_f1, _ = precision_recall_fscore_support(
                binary_actual, binary_predicted, average="binary", zero_division=0)
            support = int(binary_actual.sum())
            ap = float(average_precision_score(binary_actual, probabilities[:, class_index])) \
                if 0 < support < len(frame) else None
            if ap is not None:
                aps.append(ap)
            per_class[str(code)] = {
                "support": support, "predicted_count": int(binary_predicted.sum()),
                "precision": float(class_p), "recall": float(class_r), "f1": float(class_f1),
                "auprc_ovr": ap, "trainable_in_this_fold": code in trained_codes[field],
            }
        details[field] = {
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_precision_present_classes": float(p),
            "macro_recall_present_classes": float(r),
            "macro_f1_present_classes": float(f1),
            "macro_auprc_present_classes_ovr": float(np.mean(aps)) if aps else None,
            "per_class": per_class,
        }
    return {
        "samples": len(frame),
        "macro_field_f1": float(np.mean([value["macro_f1_present_classes"] for value in details.values()])),
        "macro_field_auprc_ovr": float(np.mean([
            value["macro_auprc_present_classes_ovr"] for value in details.values()
            if value["macro_auprc_present_classes_ovr"] is not None
        ])),
        "per_field": details,
    }


def write_csvs(result: dict, output_dir: Path) -> None:
    attribute_fields = ["split", "attribute", "samples", "accuracy", "macro_precision_present_classes",
                        "macro_recall_present_classes", "macro_f1_present_classes",
                        "macro_auprc_present_classes_ovr"]
    class_fields = ["split", "attribute", "class_code", "support", "predicted_count",
                    "precision", "recall", "f1", "auprc_ovr", "trainable_in_this_fold"]
    with (output_dir / "per_attribute_metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=attribute_fields)
        writer.writeheader()
        for split, report in result["splits"].items():
            for field, values in report["per_field"].items():
                writer.writerow({"split": split, "attribute": field, "samples": report["samples"],
                                 **{key: values[key] for key in attribute_fields[3:]}})
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=class_fields)
        writer.writeheader()
        for split, report in result["splits"].items():
            for field, values in report["per_field"].items():
                for code, class_values in values["per_class"].items():
                    writer.writerow({"split": split, "attribute": field, "class_code": code, **class_values})


def write_report(result: dict, output_dir: Path) -> None:
    lines = [
        "# 12 属性 Fold 1 评估报告",
        "",
        f"最佳轮次：{result['best_epoch']}。模型仅按验证集 Macro-AUPRC 选取；测试集未参与选择。",
        "",
        "每个属性的 Macro 指标先对该数据划分中出现的类别求平均；总体 Macro 指标再对 12 个属性求平均。",
        "",
    ]
    for split, report in result["splits"].items():
        title = "验证集" if split == "validation" else "测试集"
        lines.extend([
            f"## {title}（{report['samples']} 张）", "",
            f"总体 Macro-F1：{report['macro_field_f1']:.4f}；总体 Macro-AUPRC：{report['macro_field_auprc_ovr']:.4f}。",
            "", "| 属性 | 准确率 | Macro-Precision | Macro-Recall | Macro-F1 | Macro-AUPRC |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for field, values in report["per_field"].items():
            lines.append(
                f"| `{field}` | {values['accuracy']:.4f} | "
                f"{values['macro_precision_present_classes']:.4f} | "
                f"{values['macro_recall_present_classes']:.4f} | "
                f"{values['macro_f1_present_classes']:.4f} | "
                f"{values['macro_auprc_present_classes_ovr']:.4f} |"
            )
        lines.append("")
    lines.append("各类别的样本数、Precision、Recall、F1 与 AUPRC 见 `per_class_metrics.csv`。")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed validation and held-out test metrics for 12 fields")
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--release-root", type=Path, default=RELEASE)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models" / "dinov2-small")
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs" / "12fields_relaunch" / "fold1")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    status = json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete" or not (args.run_dir / "metrics.json").is_file():
        raise RuntimeError("The training run has not finished successfully; final test evaluation is deferred.")
    exit_path = args.run_dir / "process_exit.json"
    for _ in range(24):
        if exit_path.is_file():
            try:
                exit_data = json.loads(exit_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
            else:
                if exit_data["exit_code"] != 0:
                    raise RuntimeError(f"Training process exited with code {exit_data['exit_code']}.")
                break
        time.sleep(5)
    else:
        raise RuntimeError("Training completion was reported, but its process did not exit within 2 minutes.")
    checkpoint_path = args.run_dir / "best_trainables.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["fold"] != args.fold or checkpoint["fields"] != pipeline.FIELDS:
        raise ValueError("Checkpoint fold or field order does not match the requested evaluation.")
    trained_codes = checkpoint["trained_class_codes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = checkpoint["config"]
    model = pipeline.build_model(args.model_dir, device, int(cfg["model"]["lora_rank"]),
                                 float(cfg["model"]["lora_alpha"]),
                                 float(cfg["model"]["lora_dropout"]), trained_codes)
    _, unexpected = model.load_state_dict(checkpoint["trainable_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameter names: {unexpected[:5]}")
    result = {
        "fold": args.fold, "best_epoch": checkpoint["best_epoch"],
        "checkpoint": str(checkpoint_path.resolve()),
        "metric_definition": "Macro metrics average codes present in each split, then average the 12 fields.",
        "test_used_for_selection": False,
        "trained_class_codes": trained_codes,
        "splits": {},
    }
    for split in ("validation", "test"):
        name = "val" if split == "validation" else "test"
        frame = pipeline.load_split(args.data_dir / f"fold{args.fold}_{name}.csv", args.release_root)
        result["splits"][split] = evaluate_split(
            model, frame, args.release_root, args.batch_size, device, trained_codes,
            int(cfg["model"]["input_size"]))
    expected = checkpoint["best_val_metrics"]
    observed = result["splits"]["validation"]
    for key in ("macro_field_f1", "macro_field_auprc_ovr"):
        if abs(observed[key] - expected[key]) > 1e-4:
            raise RuntimeError(f"Validation {key} differs from the selected checkpoint: "
                               f"{observed[key]} vs {expected[key]}")
    training_result = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    if training_result["best_epoch"] != result["best_epoch"]:
        raise RuntimeError("The final training report and selected checkpoint have different best epochs.")
    for key in ("macro_field_f1", "macro_field_auprc_ovr"):
        recorded = training_result["test"][key]
        measured = result["splits"]["test"][key]
        if abs(measured - recorded) > 1e-4:
            raise RuntimeError(f"Test {key} differs from the final training report: "
                               f"{measured} vs {recorded}")
    output = args.run_dir / "detailed_metrics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csvs(result, args.run_dir)
    write_report(result, args.run_dir)
    print(json.dumps({"best_epoch": result["best_epoch"],
                      "validation_macro_field_f1": observed["macro_field_f1"],
                      "test_macro_field_f1": result["splits"]["test"]["macro_field_f1"],
                      "test_macro_field_auprc_ovr": result["splits"]["test"]["macro_field_auprc_ovr"],
                      "outputs": [str(output), str(args.run_dir / "per_attribute_metrics.csv"),
                                  str(args.run_dir / "per_class_metrics.csv"),
                                  str(args.run_dir / "report.md")]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
