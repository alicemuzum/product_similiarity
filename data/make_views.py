"""Build multi-view product images from the background-removal benchmark.

Usage: uv run python data/make_views.py [--bm ../bgrm_bm/data] [--out data/views] [--seed 0]

Reads the 200 benchmark products (white-background ABO photos + hand-checked alpha mattes)
from the sibling repo ../bgrm_bm (override with --bm or $BGRM_DATA) and writes, per product:

  ref_0..ref_4      catalogue-style references: ref_0 is the original photo, ref_1..4 are the
                    product cut out and re-placed on white or a plain light colour with small
                    geometric changes and mild lighting jitter.
  q_<bg>_j<l>_<r>   synthetic queries: bg in {clean, texture}, jitter level l in {0,1,2},
                    two repeats r. Stronger geometric changes than the references.
  real_<n>          real alternate photos of the same ABO listing (different angle, scene, crop),
                    found through the listings metadata. No matte exists for these.

Every image is a 256x256 RGB JPEG. Synthetic views also get the transformed matte, which the
"oracle" background-removal arm uses. Products are split 160 known / 40 unknown, and each
group into two folds (cross-fitting for the open-set threshold), all by product.
Writes <out>/images/, <out>/alpha/ and <out>/views.csv.
"""
import argparse
import collections
import csv
import gzip
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

SIZE = 256
N_REF = 5
N_KNOWN = 160
MIN_ALPHA = 0.05
# (brightness, contrast, saturation, hue, per-channel gain) half-ranges per jitter level
JITTER = {0: (0, 0, 0, 0, 0), 1: (0.2, 0.2, 0.2, 4, 0.05), 2: (0.45, 0.45, 0.5, 12, 0.15)}


def decontaminate(bgr, alpha):
    """Recover foreground colour from pixels blended with white (same as bgrm_bm make_composites)."""
    a = alpha[..., None]
    fg = (bgr.astype(np.float32) - (1 - a) * 255.0) / np.maximum(a, MIN_ALPHA)
    return np.clip(fg, 0, 255)


def pad_square(img, fill=255):
    h, w = img.shape[:2]
    s = max(h, w)
    if img.ndim == 3:
        out = np.full((s, s, 3), fill, img.dtype)
    else:
        out = np.zeros((s, s), img.dtype)
    y, x = (s - h) // 2, (s - w) // 2
    out[y:y + h, x:x + w] = img
    return cv2.resize(out, (SIZE, SIZE), interpolation=cv2.INTER_AREA if s > SIZE else cv2.INTER_CUBIC)


def place(fg, alpha, rng, scale_range, max_rot):
    """Crop the product to its bbox, scale, rotate and paste it at a random spot on a SIZE canvas.

    Returns float32 foreground (SIZE,SIZE,3) and alpha (SIZE,SIZE).
    """
    ys, xs = np.where(alpha > 0.02)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    fg, alpha = fg[y0:y1, x0:x1], alpha[y0:y1, x0:x1]
    h, w = alpha.shape
    # rotate on a padded canvas so corners are not clipped
    d = int(np.ceil(np.hypot(h, w)))
    py, px = (d - h) // 2, (d - w) // 2
    fg = cv2.copyMakeBorder(fg, py, d - h - py, px, d - w - px, cv2.BORDER_CONSTANT, value=0)
    alpha = cv2.copyMakeBorder(alpha, py, d - h - py, px, d - w - px, cv2.BORDER_CONSTANT, value=0)
    M = cv2.getRotationMatrix2D((d / 2, d / 2), rng.uniform(-max_rot, max_rot), 1.0)
    fg = cv2.warpAffine(fg, M, (d, d), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpAffine(alpha, M, (d, d), flags=cv2.INTER_LINEAR)
    ys, xs = np.where(alpha > 0.02)
    fg, alpha = fg[ys.min():ys.max() + 1, xs.min():xs.max() + 1], alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = alpha.shape
    s = rng.uniform(*scale_range) * SIZE / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    fg = cv2.resize(fg, (nw, nh), interpolation=interp)
    alpha = np.clip(cv2.resize(alpha, (nw, nh), interpolation=interp), 0, 1)
    out_fg = np.zeros((SIZE, SIZE, 3), np.float32)
    out_a = np.zeros((SIZE, SIZE), np.float32)
    y, x = rng.randint(0, SIZE - nh), rng.randint(0, SIZE - nw)
    out_fg[y:y + nh, x:x + nw] = fg
    out_a[y:y + nh, x:x + nw] = alpha
    return out_fg, out_a


def random_crop_resize(img, alpha, rng, min_scale):
    """Random square crop covering min_scale..1 of the side, resized back to SIZE."""
    c = round(SIZE * rng.uniform(min_scale, 1.0))
    y, x = rng.randint(0, SIZE - c), rng.randint(0, SIZE - c)
    img = cv2.resize(img[y:y + c, x:x + c], (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    alpha = cv2.resize(alpha[y:y + c, x:x + c], (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return img, alpha


def jitter(bgr, level, rng):
    """Brightness / contrast / saturation / hue shift and a colour cast, uniform in +-range."""
    b, c, s, hue, g = JITTER[level]
    if level == 0:
        return bgr
    img = bgr.astype(np.float32)
    img = img * (1 + rng.uniform(-b, b))
    mean = img.mean()
    img = (img - mean) * (1 + rng.uniform(-c, c)) + mean
    img = img * np.array([1 + rng.uniform(-g, g) for _ in range(3)], np.float32)
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue, hue)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + rng.uniform(-s, s)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def background(kind, rng, textures):
    if kind == "white":
        return np.full((SIZE, SIZE, 3), 255, np.float32)
    if kind == "plain":  # light pastel, like a catalogue sweep
        return np.full((SIZE, SIZE, 3), [rng.randint(180, 255) for _ in range(3)], np.float32)
    tex = cv2.imread(str(rng.choice(textures)), cv2.IMREAD_COLOR)
    th, tw = tex.shape[:2]
    c = min(th, tw, rng.randint(SIZE, 2 * SIZE))
    y, x = rng.randint(0, th - c), rng.randint(0, tw - c)
    return cv2.resize(tex[y:y + c, x:x + c], (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)


def synth(fg, alpha, rng, textures, bg, level, query):
    if query:
        pfg, pa = place(fg, alpha, rng, (0.45, 0.95), 20)
    else:
        pfg, pa = place(fg, alpha, rng, (0.7, 0.95), 5)
    canvas = background(bg, rng, textures)
    img = np.round(pfg * pa[..., None] + canvas * (1 - pa[..., None])).astype(np.uint8)
    if query:
        img, pa = random_crop_resize(img, pa, rng, 0.8)
    return jitter(img, level, rng), pa


def real_views(bm, bench_ids):
    """Map bench id -> list of other ABO image paths from any listing that uses the bench image."""
    f2iid = {}
    with open(bm / "image_categories.csv", newline="") as f:
        for r in csv.DictReader(f):
            stem = r["file"].rsplit(".", 1)[0]
            if stem in bench_ids:
                f2iid[stem] = r["image_id"]
    iid2f = {v: k for k, v in f2iid.items()}
    alts = collections.defaultdict(set)
    for f in sorted((bm / "listings/metadata").glob("listings_*.json.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                it = json.loads(line)
                ids = [x for x in [it.get("main_image_id")] + it.get("other_image_id", []) if x]
                for b in {iid2f[i] for i in ids if i in iid2f}:
                    alts[b].update(i for i in ids if i != f2iid[b])
    # an image listed under two bench products has an ambiguous label: drop it, and never
    # use another bench product's main image as a real view
    owners = collections.Counter(i for s in alts.values() for i in s)
    paths = {}
    with gzip.open(bm / "images/metadata/images.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            paths[r["image_id"]] = bm / "images/small" / r["path"]
    out, dropped = {}, 0
    for b, ids in alts.items():
        keep = [i for i in sorted(ids) if owners[i] == 1 and i not in iid2f and paths[i].exists()]
        dropped += len(ids) - len(keep)
        out[b] = [paths[i] for i in keep]
    return out, dropped


def thumb(img):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm", default=os.environ.get("BGRM_DATA", "../bgrm_bm/data"))
    ap.add_argument("--out", default="data/views")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    bm, out = Path(args.bm), Path(args.out)

    with open(bm / "bench/manifest.csv", newline="") as f:
        items = list(csv.DictReader(f))
    textures = sorted((bm / "dtd/images").glob("*/*.jpg"))
    rng = random.Random(args.seed)

    ids = sorted(it["id"] for it in items)
    rng.shuffle(ids)
    split = {p: ("known" if n < N_KNOWN else "unknown") for n, p in enumerate(ids)}
    fold = {p: n % 2 for n, p in enumerate(ids)}  # alternating within each group: 80/80 and 20/20

    reals, dropped = real_views(bm, set(ids))
    print(f"real views: {sum(map(len, reals.values()))} kept, {dropped} dropped as ambiguous")

    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "alpha").mkdir()
    rows, near_dupes = [], 0
    for it in sorted(items, key=lambda r: r["id"]):
        pid = it["id"]
        bgr = cv2.imread(next(str(p) for p in (bm / "bench/images").glob(f"{pid}.*")), cv2.IMREAD_COLOR)
        alpha = cv2.imread(str(bm / "bench/alpha" / f"{pid}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        fg = decontaminate(bgr, alpha)
        base = dict(product=pid, product_type=it["product_type"], split=split[pid], fold=fold[pid])

        def save(vid, img, a, **kw):
            cv2.imwrite(str(out / "images" / f"{vid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            ap_ = ""
            if a is not None:
                ap_ = f"alpha/{vid}.png"
                cv2.imwrite(str(out / ap_), np.round(np.clip(a, 0, 1) * 255).astype(np.uint8))
            rows.append(dict(view=vid, **base, path=f"images/{vid}.jpg", alpha=ap_, **kw))

        # references
        save(f"{pid}_ref_0", pad_square(bgr), pad_square(alpha), kind="ref", ref_idx=0, bg="white", jitter=0)
        for r in range(1, N_REF):
            bg = "white" if r % 2 else "plain"
            img, a = synth(fg, alpha, rng, textures, bg, 1 if r >= 3 else 0, query=False)
            save(f"{pid}_ref_{r}", img, a, kind="ref", ref_idx=r, bg=bg, jitter=1 if r >= 3 else 0)
        # synthetic queries
        for bgk in ("clean", "texture"):
            for level in (0, 1, 2):
                for rep in range(2):
                    bg = rng.choice(["white", "plain"]) if bgk == "clean" else "texture"
                    img, a = synth(fg, alpha, rng, textures, bg, level, query=True)
                    save(f"{pid}_q_{bgk}_j{level}_{rep}", img, a, kind="query", ref_idx=-1, bg=bgk, jitter=level)
        # real alternate photos (skip near-copies of the reference photo)
        t0 = thumb(pad_square(bgr))
        n = 0
        for p in reals.get(pid, []):
            img = pad_square(cv2.imread(str(p), cv2.IMREAD_COLOR))
            if np.abs(thumb(img) - t0).mean() < 4:
                near_dupes += 1
                continue
            save(f"{pid}_real_{n}", img, None, kind="real", ref_idx=-1, bg="real", jitter=-1)
            n += 1

    with open(out / "views.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    kinds = collections.Counter(r["kind"] for r in rows)
    print(f"{len(rows)} views {dict(kinds)}, {near_dupes} real near-duplicates of ref_0 skipped -> {out}")


if __name__ == "__main__":
    main()
