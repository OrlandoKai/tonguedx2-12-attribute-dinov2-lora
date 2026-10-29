# TongueDx2 12 属性舌象识别

本项目从舌象图片预测 12 项舌象属性。每项属性都有自己的类别编码，模型对每项属性分别输出一个类别。项目基于 PyTorch、DINOv2 ViT-S/14 和 LoRA，提供数据整理、模型训练、评估及逐属性结果导出代码。

## 识别属性

| 舌体属性 | 舌苔属性 |
| --- | --- |
| `tongue_color` | `coating_nature_peel` |
| `tongue_shape_indentation` | `coating_nature_moistening` |
| `tongue_shape_form` | `coating_nature_decay` |
| `tongue_shape_stab` | `coating_nature_thickness` |
| `tongue_shape_crack` | `coating_colour` |
| `tongue_shape_ecchymosis` | — |
| `tongue_color_tipside` | — |

各属性的类别编码范围见 `config.json`。这些字段是**多分类属性**：例如某项属性有 3 个类别时，模型会在这 3 个类别中选择一个，而不是对该属性输出 3 个独立的二值判断。

## 数据

标注来自 TongueDx2 release 的 `list/标注全量.xlsx`。训练、验证和测试样本由 `list/train_foldN.csv`、`list/val_foldN.csv`、`list/test.csv` 指定，图像存放在 `seg/`。`prepare_data.py` 按 `id` 和 `image_path` 将样本与 12 项标注对应，生成训练使用的 CSV；若有样本无法对应或标注格式不正确，会直接报错。

```text
release/
├── list/
│   ├── 标注全量.xlsx
│   ├── train_fold1.csv
│   ├── val_fold1.csv
│   └── test.csv
└── seg/
    └── ... 舌象图片 ...
```

数据集、预训练权重、训练检查点及生成的 CSV 不包含在本目录的源代码中。

## 模型与评估

- 图像输入尺寸为 336×336，DINOv2 ViT-S/14 提取图像特征，LoRA 用于参数高效训练。
- 模型对 12 项属性分别设置 softmax 分类头，并使用类别加权交叉熵训练。
- 训练最多 80 轮，验证指标连续 12 轮未提升时提前停止；按验证集各属性 Macro-AUPRC 的平均值选择最佳检查点。
- 测试集只在模型选择结束后评估。报告包含每项属性的准确率、Macro-Precision、Macro-Recall、Macro-F1、Macro-AUPRC，以及各类别的样本数和指标。

具体模型参数、数据增强及优化器设置见 `config.json`。

## 文件与目录

| 路径 | 作用 |
| --- | --- |
| `config.json` | 属性字段、类别编码、模型、训练和评估设置。 |
| `prepare_data.py` | 读取标注表并生成各数据划分的 12 属性 CSV。 |
| `train.py` | 训练、验证、保存检查点，并对选定模型进行测试。 |
| `evaluate_detailed.py` | 重新计算验证集和测试集指标，导出逐属性与逐类别明细。 |
| `run_background.py` | 运行 Fold 1 训练并记录日志和进程退出状态。 |
| `watch_completion.py` | 等待训练结束后启动详细评估。 |
| `data/` | `prepare_data.py` 生成的数据文件；不纳入版本管理。 |

预训练模型默认读取仓库根目录的 `models/dinov2-small/`；训练产物默认写入仓库根目录的 `runs/12fields/foldN/`。运行时可通过命令行参数指定模型、数据和输出目录。

训练完成后，输出目录中的 `metrics.json` 保存总体与逐属性测试结果；`report.md`、`per_attribute_metrics.csv` 和 `per_class_metrics.csv` 提供便于阅读和分析的详细结果。`history.csv` 记录各轮指标，`best_trainables.pt` 保存最佳模型的可训练参数。
