import argparse
import json
import os
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_full_data(data_dir: Path):
    full_path = data_dir / "full_data.json"
    if not full_path.exists():
        return {}
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for obj in data:
        name = obj.get("media_name") or obj.get("image")
        if name:
            mapping[name] = obj
    return mapping


def get_image_name(obj):
    return obj.get("image") or obj.get("media_name") or Path(obj.get("image_path", "")).name


def get_image_path(obj, data_dir: Path):
    if obj.get("image_path"):
        p = Path(obj["image_path"])
        if p.exists():
            return str(p)
    name = get_image_name(obj)
    candidates = [
        data_dir / "images" / name,
        data_dir / "images" / "images" / name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # fallback: slow search not used by default
    return str(data_dir / "images" / name)


def merge_labels(items, full_map):
    merged = []
    for obj in items:
        name = get_image_name(obj)
        if name in full_map:
            new = dict(full_map[name])
            new.update(obj)
            # Prefer original full labels if obj lacks them
            for k, v in full_map[name].items():
                if k not in new or new[k] in (None, "", []):
                    new[k] = v
            merged.append(new)
        else:
            merged.append(obj)
    return merged


def normalize_label_field(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        if not v.strip():
            return []
        # Some JSON exports use semicolon/comma separated labels
        if ";" in v:
            return [x.strip() for x in v.split(";") if x.strip()]
        return [v.strip()]
    return [str(v).strip()]


def choose_primary_label(obj, task):
    field = {
        "body_system": "Body_system_level",
        "organ": "Organ_level",
        "diagnosis": "Diagnosis",
    }[task]
    labs = normalize_label_field(obj.get(field))
    return labs[0] if labs else None


def filter_by_labels(items, task, class_names, data_dir):
    X_items, y = [], []
    class_set = set(class_names)
    for obj in items:
        lab = choose_primary_label(obj, task)
        if lab in class_set:
            img_path = get_image_path(obj, data_dir)
            if Path(img_path).exists():
                X_items.append((img_path, obj.get("caption", ""), lab))
                y.append(class_names.index(lab))
    return X_items, np.array(y, dtype=np.int64)


def top_classes(items, task, max_classes):
    c = Counter()
    for obj in items:
        lab = choose_primary_label(obj, task)
        if lab:
            c[lab] += 1
    if max_classes <= 0:
        return [k for k, _ in c.most_common()]
    return [k for k, _ in c.most_common(max_classes)]


@torch.no_grad()
def encode_images(model, preprocess, image_records, device, batch_size=64, num_workers=0):
    # image_records: list of (path, caption, label)
    feats = []
    paths = [r[0] for r in image_records]
    for i in tqdm(range(0, len(paths), batch_size), desc="Encoding images"):
        batch_paths = paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), color="black")
            imgs.append(preprocess(img))
        images = torch.stack(imgs).to(device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            f = model.encode_image(images)
        f = F.normalize(f, dim=-1).cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def encode_prompts(model, tokenizer, prompts, device, batch_size=64):
    feats = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Encoding prompts"):
        batch = prompts[i:i+batch_size]
        toks = tokenizer(batch).to(device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            f = model.encode_text(toks)
        f = F.normalize(f, dim=-1).cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def load_model(checkpoint, device):
    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
        print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model = model.to(device).eval()
    return model, preprocess_val, tokenizer


def save_report(out_dir, prefix, y_true, y_pred, class_names):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(out_dir / f"{prefix}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.savetxt(out_dir / f"{prefix}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    # Confusion plot for top classes. Keep readable.
    fig_w = max(7, min(14, 0.42 * len(class_names)))
    fig_h = max(6, min(14, 0.42 * len(class_names)))
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cm, interpolation="nearest")
    plt.title(prefix)
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    labels = [c[:18] for c in class_names]
    plt.xticks(ticks, labels, rotation=90, fontsize=7)
    plt.yticks(ticks, labels, fontsize=7)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=220)
    plt.close()

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "num_classes": int(len(class_names)),
        "num_test_samples": int(len(y_true)),
    }
    with open(out_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def zero_shot_eval(model, tokenizer, preprocess, train_items, test_items, task, max_classes, data_dir, out_dir, variant_name, batch_size):
    class_names = top_classes(train_items, task, max_classes)
    test_records, y_test = filter_by_labels(test_items, task, class_names, data_dir)
    print(f"[zero-shot] task={task}, classes={len(class_names)}, test samples={len(y_test)}")
    prompts = [f"an ultrasound image of {c}" for c in class_names]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_feats = encode_images(model, preprocess, test_records, device, batch_size=batch_size)
    text_feats = encode_prompts(model, tokenizer, prompts, device)
    sim = image_feats @ text_feats.T
    y_pred = sim.argmax(axis=1)
    prefix = f"{variant_name}_{task}_zero_shot"
    metrics = save_report(out_dir, prefix, y_test, y_pred, class_names)
    return prefix, metrics


def linear_probe_eval(model, preprocess, train_items, test_items, task, max_classes, data_dir, out_dir, variant_name, batch_size):
    class_names = top_classes(train_items, task, max_classes)
    train_records, y_train = filter_by_labels(train_items, task, class_names, data_dir)
    test_records, y_test = filter_by_labels(test_items, task, class_names, data_dir)
    print(f"[linear probe] task={task}, classes={len(class_names)}, train={len(y_train)}, test={len(y_test)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train = encode_images(model, preprocess, train_records, device, batch_size=batch_size)
    X_test = encode_images(model, preprocess, test_records, device, batch_size=batch_size)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced", solver="saga", verbose=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    prefix = f"{variant_name}_{task}_linear_probe"
    metrics = save_report(out_dir, prefix, y_test, y_pred, class_names)
    return prefix, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/workspace/US-365K")
    ap.add_argument("--train_json", default="/workspace/US-365K/prepared/train_50000.jsonl")
    ap.add_argument("--test_json", default="/workspace/US-365K/prepared/test_5000.jsonl")
    ap.add_argument("--checkpoint", default="/workspace/outputs/biomedclip_50k_lr1e5/best_model.pt")
    ap.add_argument("--output_dir", default="/workspace/outputs/extended_eval")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--organ_topk", type=int, default=20)
    ap.add_argument("--run_baseline", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_map = load_full_data(data_dir)
    train_items = merge_labels(read_jsonl(Path(args.train_json)), full_map)
    test_items = merge_labels(read_jsonl(Path(args.test_json)), full_map)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = []

    variants = []
    if args.run_baseline:
        variants.append(("baseline", None))
    variants.append(("ft50k", args.checkpoint))

    for variant_name, ckpt in variants:
        model, preprocess, tokenizer = load_model(ckpt, device)
        for task, max_classes in [("body_system", 0), ("organ", args.organ_topk)]:
            prefix, metrics = zero_shot_eval(
                model, tokenizer, preprocess, train_items, test_items, task, max_classes,
                data_dir, out_dir, variant_name, args.batch_size
            )
            all_results.append({"variant": variant_name, "task": task, "mode": "zero_shot", **metrics})

            prefix, metrics = linear_probe_eval(
                model, preprocess, train_items, test_items, task, max_classes,
                data_dir, out_dir, variant_name, args.batch_size
            )
            all_results.append({"variant": variant_name, "task": task, "mode": "linear_probe", **metrics})

        # Free GPU memory before next variant
        del model
        torch.cuda.empty_cache()

    summary = pd.DataFrame(all_results)
    summary.to_csv(out_dir / "extended_evaluation_summary.csv", index=False)
    with open(out_dir / "extended_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(summary)
    print("Saved:", out_dir / "extended_evaluation_summary.csv")


if __name__ == "__main__":
    main()
