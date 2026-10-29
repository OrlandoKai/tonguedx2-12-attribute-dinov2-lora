from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


FIELDS = [
    "tongue_color",
    "tongue_shape_indentation",
    "tongue_shape_form",
    "tongue_shape_stab",
    "tongue_shape_crack",
    "tongue_shape_ecchymosis",
    "coating_nature_peel",
    "coating_nature_moistening",
    "coating_nature_decay",
    "coating_nature_thickness",
    "coating_colour",
    "tongue_color_tipside",
]
KEYS = ["id", "image_path"]
NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def read_xlsx(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships}
        sheet = workbook.find("m:sheets/m:sheet", NS)
        if sheet is None:
            raise ValueError(f"No worksheet found in {path}")
        target = targets[sheet.attrib[f"{{{NS['r']}}}id"]].lstrip("/")
        worksheet_path = target if target.startswith("xl/") else f"xl/{target}"
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.findall(".//m:t", NS))
                      for si in shared_root.findall("m:si", NS)]
        root = ET.fromstring(archive.read(worksheet_path))
        rows: list[dict[str, str]] = []
        for row in root.findall(".//m:sheetData/m:row", NS):
            values: dict[str, str] = {}
            for cell in row.findall("m:c", NS):
                column = re.match(r"[A-Z]+", cell.attrib["r"]).group(0)
                value = cell.find("m:v", NS)
                if cell.attrib.get("t") == "inlineStr":
                    text = "".join(node.text or "" for node in cell.findall(".//m:t", NS))
                elif value is not None:
                    text = value.text or ""
                    if cell.attrib.get("t") == "s":
                        text = shared[int(text)]
                else:
                    text = ""
                values[column] = text
            rows.append(values)
    if not rows:
        raise ValueError(f"Worksheet is empty: {path}")
    headers = [rows[0].get(column, "") for column in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N")]
    if headers[:2] != KEYS or headers[2:] != FIELDS:
        raise ValueError(f"Unexpected workbook columns: {headers}")
    return pd.DataFrame([[row.get(column, "") for column in "ABCDEFGHIJKLMN"]
                         for row in rows[1:]], columns=headers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Join TongueDx2 12-field labels onto the existing split CSVs.")
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()

    labels_path = args.release_root / "list" / "标注全量.xlsx"
    labels = read_xlsx(labels_path)
    if labels.duplicated(KEYS).any() or labels["image_path"].duplicated().any():
        raise ValueError("The workbook contains duplicate image keys; refusing ambiguous label joins.")
    for field in FIELDS:
        numeric = pd.to_numeric(labels[field], errors="coerce")
        if numeric.isna().any() or ((numeric < 1) | (numeric % 1 != 0)).any():
            raise ValueError(f"{field} must contain positive integer category codes without blanks.")
        labels[field] = numeric.astype("int64")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"workbook": str(labels_path), "workbook_rows": len(labels), "fold": args.fold}
    for split, filename in (("train", f"train_fold{args.fold}.csv"),
                            ("val", f"val_fold{args.fold}.csv"),
                            ("test", "test.csv")):
        original = pd.read_csv(args.release_root / "list" / filename, dtype={"id": str, "image_path": str})
        joined = original[KEYS].merge(labels[KEYS + FIELDS], on=KEYS, how="left", validate="one_to_one", indicator=True)
        unmatched = joined["_merge"] != "both"
        if unmatched.any():
            examples = joined.loc[unmatched, KEYS].head(5).to_dict("records")
            raise ValueError(f"{split}: {int(unmatched.sum())} rows do not match the workbook, e.g. {examples}")
        if len(joined) != len(original):
            raise ValueError(f"{split}: merged row count changed ({len(original)} -> {len(joined)}).")
        output = args.output_dir / f"fold{args.fold}_{split}.csv"
        joined.drop(columns="_merge").to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8-sig")
        summary[split] = {"rows": len(joined), "csv": str(output)}
        summary[f"{split}_class_counts"] = {
            field: {str(int(code)): int(count) for code, count in joined[field].value_counts().sort_index().items()}
            for field in FIELDS
        }
    print(pd.Series(summary).to_json(force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
