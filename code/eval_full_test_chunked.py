import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm
import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_manifest(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            image_path = obj.get("image_path")
            caption = obj.get("caption", "")
            image_name = obj.get("image") or obj.get("media_name") or Path(image_path).name
            if image_path and caption:
                items.append({
                    "image_path": image_path,
                    "caption": caption,
                    "image": image_name,
                    "case_id": obj.get("case_id", ""),
                    "caption_id": obj.get("caption_id", ""),
                    "Body_system_level": obj.get("Body_system_level", []),
                    "Organ_level": obj.get("Organ_level", []),
                    "Diagnosis": obj.get("Diagnosis", []),
                })
    return items


def load_model(checkpoint_path, device):
    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    print("Loading BioMedCLIP:", model_name)

    model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    if checkpoint_path:
        print("Loading checkpoint:", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Missing keys:", len(missing))
        print("Unexpected keys:", len(unexpected))
        if isinstance(ckpt, dict):
            print("Checkpoint epoch:", ckpt.get("epoch"))
            print("Checkpoint valid_metrics:", ckpt.get("valid_metrics"))

    model = model.to(device).eval()
    return model, preprocess_val, tokenizer


@torch.no_grad()
def encode_dataset(model, preprocess, tokenizer, items, device, batch_size):
    image_features = []
    text_features = []
    kept_items = []
    bad_images = []

    for start in tqdm(range(0, len(items), batch_size), desc="Encoding"):
        batch = items[start:start + batch_size]

        imgs = []
        captions = []
        local_items = []

        for obj in batch:
            try:
                img = Image.open(obj["image_path"]).convert("RGB")
                imgs.append(preprocess(img))
                captions.append(obj["caption"])
                local_items.append(obj)
            except Exception as e:
                bad_images.append({
                    "image_path": obj["image_path"],
                    "error": str(e)
                })

        if not imgs:
            continue

        imgs = torch.stack(imgs).to(device, non_blocking=True)
        toks = tokenizer(captions).to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            im = model.encode_image(imgs)
            tx = model.encode_text(toks)

        im = F.normalize(im.float(), dim=-1).cpu()
        tx = F.normalize(tx.float(), dim=-1).cpu()

        image_features.append(im)
        text_features.append(tx)
        kept_items.extend(local_items)

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    return image_features, text_features, kept_items, bad_images


@torch.no_grad()
def chunked_recall(query_features, candidate_features, chunk_size, device):
    n = query_features.shape[0]
    max_k = 10

    candidate_features = candidate_features.to(device)

    hit1 = 0
    hit5 = 0
    hit10 = 0

    all_top10_indices = []
    all_top10_scores = []
    ranks = torch.empty(n, dtype=torch.long)

    for start in tqdm(range(0, n, chunk_size), desc="Retrieval chunks"):
        end = min(start + chunk_size, n)

        q = query_features[start:end].to(device)
        sim = q @ candidate_features.t()

        vals, inds = torch.topk(sim, k=max_k, dim=1)

        target = torch.arange(start, end, device=device).unsqueeze(1)
        match = inds.eq(target)

        hit1 += match[:, :1].any(dim=1).sum().item()
        hit5 += match[:, :5].any(dim=1).sum().item()
        hit10 += match[:, :10].any(dim=1).sum().item()

        target_score = sim.gather(1, target).squeeze(1)
        rank = (sim > target_score.unsqueeze(1)).sum(dim=1) + 1
        ranks[start:end] = rank.cpu()

        all_top10_indices.append(inds.cpu())
        all_top10_scores.append(vals.cpu())

        del q, sim, vals, inds, target, match, target_score, rank
        torch.cuda.empty_cache()

    metrics = {
        "R@1": hit1 / n,
        "R@5": hit5 / n,
        "R@10": hit10 / n,
        "median_rank": float(ranks.float().median().item())
    }

    return metrics, torch.cat(all_top10_indices), torch.cat(all_top10_scores), ranks


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_qualitative(out_dir, items, i2t_top10, t2i_top10, i2t_ranks, t2i_ranks):
    out_path = Path(out_dir) / "qualitative_retrieval_examples.csv"

    good = torch.where(i2t_ranks <= 10)[0][:5].tolist()
    bad = torch.where(i2t_ranks > 1000)[0][:5].tolist()
    chosen = good + bad

    if not chosen:
        chosen = list(range(min(10, len(items))))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "query_index",
            "image",
            "case_id",
            "caption_id",
            "body_system",
            "organ",
            "diagnosis",
            "ground_truth_caption",
            "i2t_rank",
            "i2t_top1_caption",
            "t2i_rank",
            "t2i_top1_image"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in chosen:
            i2t_top1 = int(i2t_top10[idx, 0])
            t2i_top1 = int(t2i_top10[idx, 0])
            obj = items[idx]

            writer.writerow({
                "query_index": idx,
                "image": obj.get("image", ""),
                "case_id": obj.get("case_id", ""),
                "caption_id": obj.get("caption_id", ""),
                "body_system": str(obj.get("Body_system_level", "")),
                "organ": str(obj.get("Organ_level", "")),
                "diagnosis": str(obj.get("Diagnosis", "")),
                "ground_truth_caption": obj["caption"],
                "i2t_rank": int(i2t_ranks[idx]),
                "i2t_top1_caption": items[i2t_top1]["caption"],
                "t2i_rank": int(t2i_ranks[idx]),
                "t2i_top1_image": items[t2i_top1].get("image", ""),
            })

    print("Saved qualitative examples:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--retrieval_chunk", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    out_dir = Path(args.output_dir) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    items = read_manifest(args.manifest)
    print("Manifest:", args.manifest)
    print("Total manifest items:", len(items))

    model, preprocess, tokenizer = load_model(args.checkpoint, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    image_features, text_features, kept_items, bad_images = encode_dataset(
        model, preprocess, tokenizer, items, device, args.batch_size
    )

    print("Usable pairs:", len(kept_items))
    print("Bad images:", len(bad_images))
    print("Image feature shape:", tuple(image_features.shape))
    print("Text feature shape:", tuple(text_features.shape))

    save_json(out_dir / "bad_images.json", bad_images)
    save_json(out_dir / "run_info.json", {
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "manifest_items": len(items),
        "usable_pairs": len(kept_items),
        "bad_images": len(bad_images),
        "total_params": total_params,
        "trainable_params_current_loaded_model": trainable_params,
        "batch_size": args.batch_size,
        "retrieval_chunk": args.retrieval_chunk,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    })

    print("\nRunning image-to-text retrieval")
    i2t_metrics, i2t_top10, i2t_scores, i2t_ranks = chunked_recall(
        image_features, text_features, args.retrieval_chunk, device
    )

    print("\nRunning text-to-image retrieval")
    t2i_metrics, t2i_top10, t2i_scores, t2i_ranks = chunked_recall(
        text_features, image_features, args.retrieval_chunk, device
    )

    metrics = {
        "variant": args.variant,
        "test_size": len(kept_items),
        "i2t_R@1": i2t_metrics["R@1"],
        "i2t_R@5": i2t_metrics["R@5"],
        "i2t_R@10": i2t_metrics["R@10"],
        "t2i_R@1": t2i_metrics["R@1"],
        "t2i_R@5": t2i_metrics["R@5"],
        "t2i_R@10": t2i_metrics["R@10"],
        "i2t_median_rank": i2t_metrics["median_rank"],
        "t2i_median_rank": t2i_metrics["median_rank"],
    }

    print("\nFULL TEST METRICS")
    print(json.dumps(metrics, indent=2))

    save_json(out_dir / "full_test_metrics.json", metrics)

    torch.save({
        "i2t_top10_indices": i2t_top10,
        "i2t_top10_scores": i2t_scores,
        "i2t_ranks": i2t_ranks,
        "t2i_top10_indices": t2i_top10,
        "t2i_top10_scores": t2i_scores,
        "t2i_ranks": t2i_ranks,
    }, out_dir / "retrieval_top10_and_ranks.pt")

    save_qualitative(out_dir, kept_items, i2t_top10, t2i_top10, i2t_ranks, t2i_ranks)

    summary_path = Path(args.output_dir) / "full_test_summary.csv"
    write_header = not summary_path.exists()

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        fields = [
            "variant", "test_size",
            "i2t_R@1", "i2t_R@5", "i2t_R@10",
            "t2i_R@1", "t2i_R@5", "t2i_R@10",
            "i2t_median_rank", "t2i_median_rank"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({k: metrics[k] for k in fields})

    print("Saved:", out_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
