import argparse, csv, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm
import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ManifestDataset(Dataset):
    def __init__(self, manifest, preprocess, limit=0):
        self.preprocess = preprocess
        self.items = []
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                if o.get("image_path") and o.get("caption"):
                    self.items.append({
                        "image_path": o["image_path"],
                        "caption": o["caption"],
                        "image": o.get("image") or o.get("media_name") or Path(o["image_path"]).name,
                        "case_id": o.get("case_id", ""),
                        "caption_id": o.get("caption_id", ""),
                        "Body_system_level": o.get("Body_system_level", []),
                        "Organ_level": o.get("Organ_level", []),
                        "Diagnosis": o.get("Diagnosis", []),
                    })
        if limit and limit > 0:
            self.items = self.items[:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        o = self.items[idx]
        try:
            img = Image.open(o["image_path"]).convert("RGB")
            img = self.preprocess(img)
            return idx, img, o["caption"], o
        except Exception as e:
            return idx, None, None, {"image_path": o["image_path"], "error": str(e)}

def collate_fn(batch):
    good, bad = [], []
    for idx, img, cap, obj in batch:
        if img is None:
            bad.append(obj)
        else:
            good.append((idx, img, cap, obj))
    if not good:
        return None
    idxs, imgs, caps, objs = zip(*good)
    return list(idxs), torch.stack(imgs), list(caps), list(objs), bad

def load_model(checkpoint, device):
    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    print("Loading BioMedCLIP:", model_name)
    model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    if checkpoint:
        print("Loading checkpoint:", checkpoint)
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
        if isinstance(ckpt, dict):
            print("Checkpoint epoch:", ckpt.get("epoch"))
            print("Checkpoint valid_metrics:", ckpt.get("valid_metrics"))

    model = model.to(device).eval()
    return model, preprocess_val, tokenizer

@torch.inference_mode()
def encode_dataset(model, preprocess, tokenizer, manifest, device, batch_size, num_workers, limit):
    ds = ManifestDataset(manifest, preprocess, limit=limit)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn
    )

    image_features, text_features, kept_items, bad_images = [], [], [], []

    for batch in tqdm(dl, desc="Encoding fast"):
        if batch is None:
            continue
        idxs, imgs, caps, objs, bad = batch
        bad_images.extend(bad)

        imgs = imgs.to(device, non_blocking=True)
        toks = tokenizer(caps).to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            im = model.encode_image(imgs)
            tx = model.encode_text(toks)

        image_features.append(F.normalize(im.float(), dim=-1).cpu())
        text_features.append(F.normalize(tx.float(), dim=-1).cpu())
        kept_items.extend(objs)

    return torch.cat(image_features), torch.cat(text_features), kept_items, bad_images

@torch.inference_mode()
def chunked_recall(query, cand, chunk_size, device):
    n = query.shape[0]
    cand = cand.to(device)
    hit1 = hit5 = hit10 = 0
    ranks = torch.empty(n, dtype=torch.long)
    top10_inds, top10_scores = [], []

    for s in tqdm(range(0, n, chunk_size), desc="Retrieval chunks"):
        e = min(s + chunk_size, n)
        q = query[s:e].to(device)
        sim = q @ cand.t()
        vals, inds = torch.topk(sim, k=10, dim=1)

        target = torch.arange(s, e, device=device).unsqueeze(1)
        match = inds.eq(target)

        hit1 += match[:, :1].any(dim=1).sum().item()
        hit5 += match[:, :5].any(dim=1).sum().item()
        hit10 += match[:, :10].any(dim=1).sum().item()

        target_score = sim.gather(1, target).squeeze(1)
        rank = (sim > target_score.unsqueeze(1)).sum(dim=1) + 1
        ranks[s:e] = rank.cpu()

        top10_inds.append(inds.cpu())
        top10_scores.append(vals.cpu())

        del q, sim, vals, inds, target, match, target_score, rank
        torch.cuda.empty_cache()

    return {
        "R@1": hit1 / n,
        "R@5": hit5 / n,
        "R@10": hit10 / n,
        "median_rank": float(ranks.float().median().item())
    }, torch.cat(top10_inds), torch.cat(top10_scores), ranks

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

    fields = [
        "query_index", "image", "case_id", "caption_id", "body_system",
        "organ", "diagnosis", "ground_truth_caption", "i2t_rank",
        "i2t_top1_caption", "t2i_rank", "t2i_top1_image"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for idx in chosen:
            i2t_top1 = int(i2t_top10[idx, 0])
            t2i_top1 = int(t2i_top10[idx, 0])
            o = items[idx]
            w.writerow({
                "query_index": idx,
                "image": o.get("image", ""),
                "case_id": o.get("case_id", ""),
                "caption_id": o.get("caption_id", ""),
                "body_system": str(o.get("Body_system_level", "")),
                "organ": str(o.get("Organ_level", "")),
                "diagnosis": str(o.get("Diagnosis", "")),
                "ground_truth_caption": o["caption"],
                "i2t_rank": int(i2t_ranks[idx]),
                "i2t_top1_caption": items[i2t_top1]["caption"],
                "t2i_rank": int(t2i_ranks[idx]),
                "t2i_top1_image": items[t2i_top1].get("image", ""),
            })
    print("Saved qualitative:", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--retrieval_chunk", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    out_dir = Path(args.output_dir) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    model, preprocess, tokenizer = load_model(args.checkpoint, device)

    image_feats, text_feats, items, bad = encode_dataset(
        model, preprocess, tokenizer, args.manifest, device,
        args.batch_size, args.num_workers, args.limit
    )

    print("Usable pairs:", len(items))
    print("Bad images:", len(bad))
    print("Image features:", tuple(image_feats.shape))
    print("Text features:", tuple(text_feats.shape))

    save_json(out_dir / "bad_images.json", bad)
    save_json(out_dir / "run_info.json", {
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "usable_pairs": len(items),
        "bad_images": len(bad),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "retrieval_chunk": args.retrieval_chunk,
        "limit": args.limit,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    })

    print("Image-to-text retrieval")
    i2t, i2t_top10, i2t_scores, i2t_ranks = chunked_recall(
        image_feats, text_feats, args.retrieval_chunk, device
    )

    print("Text-to-image retrieval")
    t2i, t2i_top10, t2i_scores, t2i_ranks = chunked_recall(
        text_feats, image_feats, args.retrieval_chunk, device
    )

    metrics = {
        "variant": args.variant,
        "test_size": len(items),
        "i2t_R@1": i2t["R@1"],
        "i2t_R@5": i2t["R@5"],
        "i2t_R@10": i2t["R@10"],
        "t2i_R@1": t2i["R@1"],
        "t2i_R@5": t2i["R@5"],
        "t2i_R@10": t2i["R@10"],
        "i2t_median_rank": i2t["median_rank"],
        "t2i_median_rank": t2i["median_rank"],
    }

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

    save_qualitative(out_dir, items, i2t_top10, t2i_top10, i2t_ranks, t2i_ranks)

    summary = Path(args.output_dir) / "full_test_summary.csv"
    write_header = not summary.exists()
    fields = [
        "variant", "test_size", "i2t_R@1", "i2t_R@5", "i2t_R@10",
        "t2i_R@1", "t2i_R@5", "t2i_R@10",
        "i2t_median_rank", "t2i_median_rank"
    ]
    with open(summary, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow({k: metrics[k] for k in fields})

    print("Saved outputs:", out_dir)
    print("Updated summary:", summary)

if __name__ == "__main__":
    main()
