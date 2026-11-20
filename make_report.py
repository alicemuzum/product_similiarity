"""Build report.ipynb from the cells below and execute it in place.

Usage: uv run python make_report.py

The notebook only reads data/results/*.csv, data/emb/ and data/views/, so it is cheap to
re-run after eval.py. Keeping the source here (instead of hand-editing JSON) keeps diffs readable.
"""
import subprocess

import nbformat as nbf

CELLS = [
    ("md", """# Product image similarity: results

Frozen backbones, a FAISS gallery of 160 known products (1, 3 or 5 references each), and an
"unknown" decision for the 40 products that are never in the gallery. Built on the 200 products
of the bgrm_bm background-removal benchmark. Every number comes from `eval.py`
(`data/results/`). Confidence intervals are 95% intervals from resampling *products*
(1000 resamples), since views of one product are not independent.

Two query sets, always reported separately:
* **synthetic**: 12 views per product made from the main photo (new background, crop,
  rotation, scale, colour jitter). Easy, because the gallery holds the same photo.
* **real**: the product's other ABO listing photos (other angles, packaging, close-ups).
  969 photos. This is the honest test."""),
    ("code", """import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
pd.set_option("display.width", 200); pd.set_option("display.precision", 3)
R = Path("data/results"); V = Path("data/views")
S = pd.read_csv(R / "summary.csv"); B = pd.read_csv(R / "breakdown.csv")
L = pd.read_csv(R / "latency.csv"); views = pd.read_csv(V / "views.csv")
NAMES = {"dinov2_s": "DINOv2 ViT-S/14", "dinov2_b": "DINOv2 ViT-B/14", "clip_b32": "CLIP ViT-B/32",
         "siglip_b16": "SigLIP ViT-B/16", "mobilenet_v3": "MobileNetV3-L"}
ORDER = list(NAMES)
def ci(df, m): return df.apply(lambda r: f"{r[m]:.3f} [{r[m+'_lo']:.3f}, {r[m+'_hi']:.3f}]", axis=1)"""),
    ("md", "## 1. Headline: raw images, prototype classifier, 5 references"),
    ("code", """h = S[(S.arm == "raw") & (S.k == 5) & (S.method == "proto")].copy()
h["backbone"] = pd.Categorical(h.backbone, ORDER)
t = h.sort_values(["queries", "backbone"]).assign(**{
    "top-1": lambda d: ci(d, "top1"), "top-5": lambda d: d.top5.round(3), "mAP": lambda d: d["map"].round(3),
    "AUROC (margin)": lambda d: ci(d, "auroc_margin"), "acc @5% FAR (margin)": lambda d: ci(d, "acc_far5_margin")})
t[["queries", "backbone", "top-1", "top-5", "mAP", "AUROC (margin)", "acc @5% FAR (margin)"]]"""),
    ("md", """## 2. Backbone x background-removal arm

`raw` is the image as is; `u2net` and `birefnet-general` composite it onto white with that
model's predicted mask; `oracle` uses the ground-truth matte (synthetic only)."""),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
arms = ["raw", "u2net", "birefnet-general", "oracle"]
for ax, q in zip(axes, ["synthetic", "real"]):
    d = S[(S.k == 5) & (S.method == "proto") & (S.queries == q)]
    for i, a in enumerate(arms):
        e = d[d.arm == a].set_index("backbone").reindex(ORDER)
        x = np.arange(len(ORDER)) + (i - 1.5) * 0.2
        ax.bar(x, e.top1, 0.2, yerr=[e.top1 - e.top1_lo, e.top1_hi - e.top1], label=a, capsize=2)
    ax.set_xticks(range(len(ORDER)), [NAMES[b] for b in ORDER], rotation=20)
    ax.set_title(f"{q} queries: top-1 (k=5, prototype)"); ax.set_ylim(0.3, 1)
axes[0].legend(); plt.tight_layout()"""),
    ("code", """p = S[(S.k == 5) & (S.method == "proto")].pivot_table(index="backbone", columns=["queries", "arm"], values="top1")
p.reindex(ORDER).round(3)"""),
    ("md", "## 3. References per product and classifier (k = 1, 3, 5)"),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=False)
for ax, q in zip(axes, ["synthetic", "real"]):
    for b in ORDER:
        for m, ls in [("proto", "-"), ("knn", "--")]:
            d = S[(S.arm == "raw") & (S.queries == q) & (S.backbone == b) & (S.method == m)].sort_values("k")
            ax.plot(d.k, d.top1, ls, marker="o", label=f"{NAMES[b]} {m}" if q == "real" else None,
                    color=f"C{ORDER.index(b)}")
    ax.set_xticks([1, 3, 5]); ax.set_xlabel("references per product"); ax.set_title(f"{q}: top-1, raw")
axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left"); plt.tight_layout()"""),
    ("md", """References 1-4 are re-placements of the main photo (`ref_0`) on white or plain colours, so
they add little that `ref_0` doesn't already carry: the curves are nearly flat. Real extra
references (other photos) would be needed for k to matter. With k = 1 the prototype and kNN
classifiers are identical by construction."""),
    ("md", "## 4. Robustness: background and colour jitter (synthetic queries)"),
    ("code", """b = B[(B.k == 5) & (B.method == "proto") & B.arm.isin(["raw", "birefnet-general"])]
p = b.pivot_table(index=["backbone", "arm"], columns=["bg", "jitter"], values="top1").reindex(ORDER, level=0)
p.round(3)"""),
    ("code", """fig, ax = plt.subplots(figsize=(8, 3.8))
for b_ in ["dinov2_b", "siglip_b16", "mobilenet_v3"]:
    for a, ls in [("raw", "-"), ("birefnet-general", "--")]:
        for bg, mk in [("clean", "o"), ("texture", "s")]:
            d = B[(B.k == 5) & (B.method == "proto") & (B.backbone == b_) & (B.arm == a) & (B.bg == bg)].sort_values("jitter")
            ax.plot(d.jitter, d.top1, ls, marker=mk, color=f"C{ORDER.index(b_)}",
                    label=f"{NAMES[b_]}, {a}, {bg}")
ax.set_xticks([0, 1, 2], ["none", "mild", "strong"]); ax.set_xlabel("colour jitter"); ax.set_ylabel("top-1")
ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left"); plt.tight_layout()"""),
    ("md", """## 5. Open-set rejection

Score = top-1 similarity, or the margin between the best and second-best product. The
threshold accepts 5% of unknown queries; it is fitted on one half of the products and applied to
the other half (cross-fitting), so no threshold ever sees its own test queries. `far` is the
false-accept rate actually realised on test."""),
    ("code", """o = S[(S.k == 5) & (S.method == "proto") & S.arm.isin(["raw", "birefnet-general"])].copy()
o["backbone"] = pd.Categorical(o.backbone, ORDER)
o.sort_values(["queries", "arm", "backbone"])[["queries", "arm", "backbone", "auroc_top1", "auroc_margin",
    "acc_far5_top1", "far_top1", "acc_far5_margin", "far_margin"]].set_index(["queries", "arm", "backbone"])"""),
    ("code", """pq = pd.read_csv(R / "per_query.csv.gz")
sel = lambda b, a: pq[(pq.backbone == b) & (pq.arm == a) & (pq.k == 5) & (pq.method == "proto")]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
for i, (b_, a) in enumerate([("siglip_b16", "raw"), ("dinov2_b", "birefnet-general")]):
    d = sel(b_, a)
    for j, kind in enumerate(["query", "real"]):
        ax = axes[2 * i + j]; e = d[d.kind == kind]
        for split, c in [("known", "C0"), ("unknown", "C3")]:
            ax.hist(e[e.split == split].margin, bins=40, alpha=.6, density=True, color=c, label=split)
        ax.set_title(f"{NAMES[b_]}, {a}\\n{'synthetic' if kind == 'query' else 'real'}: margin", fontsize=9)
axes[0].legend(); plt.tight_layout()"""),
    ("md", """## 6. Adding a product needs no retraining

Take a product the gallery has never seen, show it is rejected, then add its references with
`Gallery.add` and query again. Nothing is fitted; the new vectors are searchable at once."""),
    ("code", """from index import Gallery, prototype_classify
from openset import cross_fit
emb = np.load("data/emb/siglip_b16/raw.npy")
known = views.split.eq("known") & views.kind.eq("ref")
g = Gallery(emb.shape[1]).add(emb[known], views["product"][known])
new = views[views.split.eq("unknown")]["product"].iloc[0]
qm = (views["product"] == new) & views.kind.isin(["query", "real"])
th = sel("siglip_b16", "raw").query("kind == 'query'")
_, t = cross_fit(th.margin.to_numpy(), th.split.eq("known").to_numpy(), th.fold.to_numpy(), 0.05)
thr = min(t.values())
def show(label):
    r = prototype_classify(g, emb[qm])
    print(f"{label}: {g.index.ntotal} refs, {len(g.classes)} products; "
          f"accepted {np.mean(r.margin >= thr):.0%} of {qm.sum()} queries, "
          f"predicted '{new}' for {np.mean(r.pred == new):.0%}")
show("before")
import time; t0 = time.perf_counter()
refs = (views["product"] == new) & views.kind.eq("ref")
g.add(emb[refs], new)
print(f"add took {(time.perf_counter() - t0) * 1e3:.2f} ms")
show("after ")"""),
    ("md", "## 7. Latency and embedding size"),
    ("code", """acc = S[(S.arm == "raw") & (S.k == 5) & (S.method == "proto") & (S.queries == "real")].set_index("backbone").top1
lt = L.pivot_table(index="backbone", columns="device", values="ms_per_image").join(L.groupby("backbone")[["params_m", "dim"]].first())
lt["real top-1"] = acc
lt["bytes/vector (fp32)"] = lt.dim * 4
lt = lt.reindex(ORDER)
lt.columns.name = None
lt.round(2)"""),
    ("md", """Batch-1 latency per image on an RTX 3060 Laptop GPU and its CPU (6 threads), including
preprocessing tensors on the device but not image decoding. `params_m` for CLIP and SigLIP
counts the whole model including the unused text tower; the image towers are about 88 M
(CLIP B/32) and 93 M (SigLIP B/16). A gallery of 1 million 768-d float32 vectors is about 3 GB,
which `IndexFlatIP` searches exhaustively; at that scale IVF-PQ or HNSW would replace it."""),
    ("md", "## 8. Failures: most-confused pairs and nearest neighbours"),
    ("code", """pd.read_csv(R / "confusions.csv").head(12)"""),
    ("code", """def ref0(p): return Image.open(V / views[(views["product"] == p) & (views.ref_idx == 0)].path.iloc[0])
def grid(df, title, n=8):
    df = df.head(n)
    fig, axes = plt.subplots(len(df), 3, figsize=(6.5, 2.2 * len(df)))
    for r, (_, x) in zip(axes, df.iterrows()):
        q = Image.open(V / views.set_index("view").loc[x.view].path)
        for ax, im, t in zip(r, [q, ref0(x["product"]), ref0(x.pred)],
                             ["query", f"true {x['product'][:10]}", f"pred (margin {x.margin:.2f})"]):
            ax.imshow(im); ax.set_title(t, fontsize=8); ax.axis("off")
    fig.suptitle(title); plt.tight_layout()
d = sel("siglip_b16", "raw")
fails = d[(d.split == "known") & (d.kind == "real") & ~d.correct1].sort_values("margin", ascending=False)
grid(fails, "SigLIP raw, real queries: confident mistakes")"""),
    ("code", """fails = d[(d.split == "known") & (d.kind == "query") & ~d.correct1].sort_values("margin", ascending=False)
grid(fails, "SigLIP raw, synthetic queries: confident mistakes")"""),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, "report.ipynb")
    subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "report.ipynb"], check=True)


if __name__ == "__main__":
    main()
