"""Frozen backbone embeddings for every view, cached as .npy.

Usage: uv run python embed.py [--backbones all] [--arms raw,oracle,u2net,birefnet-general]
                              [--latency]

Backbones (no fine-tuning):
  dinov2_s, dinov2_b   DINOv2 ViT-S/14 and ViT-B/14, CLS token (384 / 768-d). Loaded from timm
                       (vit_*_patch14_dinov2.lvd142m, the official weights) at 224 px instead of
                       torch.hub, so no downloaded hub code runs.
  clip_b32             OpenAI CLIP ViT-B/32 image embedding via open_clip (512-d)
  siglip_b16           SigLIP ViT-B/16 (webli) image embedding via open_clip (768-d)
  mobilenet_v3         torchvision MobileNetV3-Large ImageNet weights, pooled features (960-d)

Arms decide what the backbone sees:
  raw       the view as is
  oracle    the view composited onto white with its ground-truth matte (synthetic views only;
            real photos have no matte, so their rows are NaN)
  <model>   the view composited onto white with that model's predicted mask (data/masks/<model>)

Writes data/emb/<backbone>/<arm>.npy, float32 (n_views, dim), L2-normalised, rows in
views.csv order. --latency also times single-image inference (GPU and CPU) into
data/emb/latency.csv.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

BACKBONES = ["dinov2_s", "dinov2_b", "clip_b32", "siglip_b16", "mobilenet_v3"]
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load(name, device):
    """Return (model, transform(PIL)->tensor, embed(batch)->features)."""
    import torchvision.transforms as T

    if name.startswith("dinov2"):
        import timm

        arch = {"dinov2_s": "vit_small_patch14_dinov2.lvd142m", "dinov2_b": "vit_base_patch14_dinov2.lvd142m"}[name]
        model = timm.create_model(arch, pretrained=True, num_classes=0, img_size=224)  # pooled = CLS token
        tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(*IMAGENET)])
        fn = model
    elif name in ("clip_b32", "siglip_b16"):
        import open_clip

        arch, tag = {"clip_b32": ("ViT-B-32", "openai"), "siglip_b16": ("ViT-B-16-SigLIP", "webli")}[name]
        model, _, tf = open_clip.create_model_and_transforms(arch, pretrained=tag)
        fn = model.encode_image
    elif name == "mobilenet_v3":
        import torchvision

        model = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2")
        tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(*IMAGENET)])

        def fn(x):
            return model.avgpool(model.features(x)).flatten(1)
    else:
        raise ValueError(name)
    model.to(device).eval()
    return model, tf, fn


def arm_image(views, row, arm):
    img = Image.open(views / row.path).convert("RGB")
    if arm == "raw":
        return img
    if arm == "oracle":
        if not isinstance(row.alpha, str) or not row.alpha:
            return None
        mask = Image.open(views / row.alpha)
    else:
        mask = Image.open(Path("data/masks") / arm / f"{row.view}.png")
    m = np.asarray(mask.convert("L"), np.float32)[..., None] / 255
    out = np.asarray(img, np.float32) * m + 255 * (1 - m)
    return Image.fromarray(np.round(out).astype(np.uint8))


@torch.no_grad()
def embed_arm(fn, tf, views, df, arm, device, bs=64):
    feats, idx, batch = [], [], []

    def flush():
        x = torch.stack(batch).to(device)
        feats.append(F.normalize(fn(x).float(), dim=-1).cpu().numpy())
        batch.clear()

    for i, row in enumerate(df.itertuples()):
        img = arm_image(views, row, arm)
        if img is None:
            continue
        batch.append(tf(img))
        idx.append(i)
        if len(batch) == bs:
            flush()
    if batch:
        flush()
    f = np.concatenate(feats)
    out = np.full((len(df), f.shape[1]), np.nan, np.float32)
    out[idx] = f
    return out


@torch.no_grad()
def latency(name, views, df, n=50):
    rows = []
    imgs = [Image.open(views / p).convert("RGB") for p in df.path[:n]]
    for device in ("cuda", "cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        model, tf, fn = load(name, device)
        for img in imgs[:5]:  # warm-up
            fn(tf(img)[None].to(device))
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for img in imgs:
            fn(tf(img)[None].to(device)).cpu()
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / len(imgs) * 1000
        params = sum(p.numel() for p in model.parameters()) / 1e6
        rows.append(dict(backbone=name, device=device, ms_per_image=ms, params_m=params, threads=torch.get_num_threads()))
        del model
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="data/views")
    ap.add_argument("--out", default="data/emb")
    ap.add_argument("--backbones", default="all")
    ap.add_argument("--arms", default="raw,oracle,u2net,birefnet-general")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    views, out = Path(args.views), Path(args.out)
    df = pd.read_csv(views / "views.csv")
    names = BACKBONES if args.backbones == "all" else args.backbones.split(",")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lat = []
    for name in names:
        model, tf, fn = load(name, device)
        (out / name).mkdir(parents=True, exist_ok=True)
        for arm in args.arms.split(","):
            dst = out / name / f"{arm}.npy"
            if dst.exists() and not args.force:
                continue
            t = time.time()
            e = embed_arm(fn, tf, views, df, arm, device)
            np.save(dst, e)
            print(f"{name:13s} {arm:17s} {e.shape} {time.time() - t:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
        if args.latency:
            lat += latency(name, views, df)
    if lat:
        pd.DataFrame(lat).to_csv(out / "latency.csv", index=False)
        print(pd.DataFrame(lat).to_string(index=False))


if __name__ == "__main__":
    main()
