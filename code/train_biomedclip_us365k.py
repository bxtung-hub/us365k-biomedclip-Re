import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm
import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True


class US365KDataset(Dataset):
    def __init__(self, jsonl_path, preprocess):
        self.jsonl_path = Path(jsonl_path)
        self.preprocess = preprocess
        self.items = []

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                caption = obj.get("caption", "")
                img_path = (
                    obj.get("image_path")
                    or obj.get("path")
                    or obj.get("filepath")
                    or obj.get("file_path")
                )

                if img_path is None:
                    # fallback nếu file chỉ có tên ảnh
                    img_name = obj.get("image") or obj.get("media_name")
                    img_path = f"/workspace/US-365K/images/{img_name}"

                self.items.append({
                    "image_path": img_path,
                    "caption": caption
                })

        print(f"Loaded {len(self.items)} samples from {self.jsonl_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = item["image_path"]
        caption = item["caption"]

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.preprocess(image)
            return image, caption
        except Exception:
            # Nếu có ảnh lỗi, lấy mẫu kế tiếp
            new_idx = (idx + 1) % len(self.items)
            return self.__getitem__(new_idx)


def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(captions)


def contrastive_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    labels = torch.arange(len(image_features), device=image_features.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


@torch.no_grad()
def encode_dataset(model, tokenizer, dataloader, device):
    model.eval()
    all_img = []
    all_txt = []

    for images, captions in tqdm(dataloader, desc="Encoding"):
        images = images.to(device)
        texts = tokenizer(captions).to(device)

        with torch.cuda.amp.autocast():
            img_feat = model.encode_image(images)
            txt_feat = model.encode_text(texts)

        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        all_img.append(img_feat.cpu())
        all_txt.append(txt_feat.cpu())

    return torch.cat(all_img, dim=0), torch.cat(all_txt, dim=0)


def retrieval_metrics(image_features, text_features):
    sim = image_features @ text_features.t()
    n = sim.shape[0]
    target = torch.arange(n)

    def recall_at_k(scores, k):
        topk = scores.topk(k, dim=1).indices
        correct = (topk == target.unsqueeze(1)).any(dim=1).float().mean().item()
        return correct

    metrics = {
        "i2t_R@1": recall_at_k(sim, 1),
        "i2t_R@5": recall_at_k(sim, 5),
        "i2t_R@10": recall_at_k(sim, 10),
        "t2i_R@1": recall_at_k(sim.t(), 1),
        "t2i_R@5": recall_at_k(sim.t(), 5),
        "t2i_R@10": recall_at_k(sim.t(), 10),
    }
    return metrics


def print_trainable_parameters(model):
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable ratio: {100 * trainable / total:.4f}%")


def set_projection_tuning(model):
    # Đóng băng toàn bộ model
    for p in model.parameters():
        p.requires_grad = False

    # Chỉ mở các lớp projection và logit_scale
    for name, p in model.named_parameters():
        lname = name.lower()
        if (
            "proj" in lname
            or "projection" in lname
            or "logit_scale" in lname
        ):
            p.requires_grad = True

    # Nếu vì cấu trúc model không bắt được projection, mở thêm một số lớp cuối
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_count < 1000:
        print("Warning: too few trainable parameters. Unfreezing logit_scale only may be weak.")


def evaluate(model, tokenizer, test_loader, device, out_dir, prefix):
    image_features, text_features = encode_dataset(model, tokenizer, test_loader, device)
    metrics = retrieval_metrics(image_features, text_features)

    print(f"\nRetrieval metrics - {prefix}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    out_path = Path(out_dir) / f"{prefix}_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics:", out_path)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--valid_json", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    print("Loading BioMedCLIP...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device)

    train_ds = US365KDataset(args.train_json, preprocess_train)
    valid_ds = US365KDataset(args.valid_json, preprocess_val)
    test_ds = US365KDataset(args.test_json, preprocess_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Baseline trước khi fine-tune
    evaluate(model, tokenizer, test_loader, device, args.output_dir, "baseline_biomedclip")

    if args.eval_only:
        return

    print("\nSetting projection tuning...")
    set_projection_tuning(model)
    print_trainable_parameters(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    scaler = torch.cuda.amp.GradScaler()

    best_valid_r10 = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, captions in pbar:
            images = images.to(device)
            texts = tokenizer(captions).to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                logit_scale = model.logit_scale.exp()
                loss = contrastive_loss(image_features, text_features, logit_scale)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=total_loss / steps)

        print(f"Epoch {epoch} train loss: {total_loss / max(steps,1):.4f}")

        valid_metrics = evaluate(model, tokenizer, valid_loader, device, args.output_dir, f"valid_epoch_{epoch}")
        valid_score = valid_metrics["i2t_R@10"] + valid_metrics["t2i_R@10"]

        if valid_score > best_valid_r10:
            best_valid_r10 = valid_score
            ckpt_path = Path(args.output_dir) / "best_model.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "valid_metrics": valid_metrics
            }, ckpt_path)
            print("Saved best model:", ckpt_path)

    # Test cuối cùng
    evaluate(model, tokenizer, test_loader, device, args.output_dir, "test_after_finetune")


if __name__ == "__main__":
    main()
