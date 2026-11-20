"""Evaluate every backbone x arm x k x classifier on synthetic and real queries.

Usage: uv run python eval.py [--emb data/emb] [--out data/results] [--boot 1000]

For each config the gallery holds the first k references (ref_0..ref_{k-1}) of the 160 known
products. Queries are the synthetic views ("synthetic") and the real ABO alternate photos
("real") of all 200 products; the 40 unknown products only ever appear as queries.

Writes to <out>:
  per_query.csv.gz  one row per config x query: prediction, ranks, scores
  summary.csv       per config x query set: top-1/top-5 accuracy, Recall@1/5 and mAP over
                    reference images (known queries), open-set AUROC (top-1 similarity and
                    margin scores) and known-class accuracy at 5% false-accept rate, each with a
                    product-level bootstrap 95% CI
  breakdown.csv     synthetic queries split by background (clean/texture) and jitter level
  confusions.csv    most-confused product pairs for --confusion-config
  latency.csv       copied from data/emb/latency.csv with embedding sizes
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from index import Gallery, knn_classify, prototype_classify
from openset import auroc, cross_fit

KS = (1, 3, 5)
FAR = 0.05


def average_precision(rel):
    """AP for each row of a boolean (n_q, n_gallery) relevance matrix ranked best first."""
    hits = np.cumsum(rel, axis=1)
    prec = hits / np.arange(1, rel.shape[1] + 1)
    return (prec * rel).sum(1) / np.maximum(rel.sum(1), 1)


def run_config(emb, views, k, method):
    known = views.split.eq("known").to_numpy()
    ref = (views.kind.eq("ref") & (views.ref_idx < k)).to_numpy() & known
    qmask = views.kind.isin(["query", "real"]).to_numpy() & ~np.isnan(emb[:, 0])
    g = Gallery(emb.shape[1]).add(emb[ref], views["product"].to_numpy()[ref])
    q = emb[qmask]
    rank = prototype_classify(g, q) if method == "proto" else knn_classify(g, q, k)

    sims, ids = g.search(q)
    truth = views["product"].to_numpy()[qmask]
    rel = g.labels[ids] == truth[:, None]
    out = views.loc[qmask, ["view", "product", "product_type", "split", "fold", "kind", "bg", "jitter"]].copy()
    out["pred"] = rank.pred
    out["pred2"] = rank.classes[:, 1]
    out["correct1"] = rank.pred == truth
    out["correct5"] = (rank.classes[:, :5] == truth[:, None]).any(1)
    out["r1"] = rel[:, 0]
    out["r5"] = rel[:, :5].any(1)
    out["ap"] = average_precision(rel)
    out["top1"] = rank.top1
    out["margin"] = rank.margin
    return out


def boot_draws(products, B, rng):
    """(B, n_products) multiplicity of each product in each bootstrap resample."""
    n = len(products)
    idx = rng.integers(0, n, size=(B, n))
    return np.stack([np.bincount(r, minlength=n) for r in idx])


def mean_ci(df, col, B, rng):
    """Mean over queries with a 95% CI from resampling products."""
    g = df.groupby("product")[col].agg(["sum", "count"])
    W = boot_draws(g.index, B, rng)
    stats = (W @ g["sum"].to_numpy(float)) / (W @ g["count"].to_numpy(float))
    return df[col].mean(), *np.percentile(stats, [2.5, 97.5])


def auroc_ci(df, col, B, rng):
    kn, un = df[df.split == "known"], df[df.split == "unknown"]
    point = auroc(kn[col], un[col])
    kp, up = kn["product"].unique(), un["product"].unique()
    Wk, Wu = boot_draws(kp, B, rng), boot_draws(up, B, rng)
    ki = pd.Index(kp).get_indexer(kn["product"])
    ui = pd.Index(up).get_indexer(un["product"])
    stats = [auroc(kn[col], un[col], Wk[b, ki], Wu[b, ui]) for b in range(B)]
    return point, *np.percentile(stats, [2.5, 97.5])


def summarise(df, B, rng):
    row = {}
    kn = df[df.split == "known"]
    for col, name in [("correct1", "top1"), ("correct5", "top5"), ("r1", "recall1"), ("r5", "recall5"), ("ap", "map")]:
        row[name], row[f"{name}_lo"], row[f"{name}_hi"] = mean_ci(kn, col, B, rng)
    for score in ("top1", "margin"):
        row[f"auroc_{score}"], row[f"auroc_{score}_lo"], row[f"auroc_{score}_hi"] = auroc_ci(df, score, B, rng)
        acc, _ = cross_fit(df[score].to_numpy(), df.split.eq("known").to_numpy(), df.fold.to_numpy(), FAR)
        d = df.assign(_ok=acc & df.correct1.to_numpy(), _acc=acc)
        kd = d[d.split == "known"]
        name = f"acc_far5_{score}"
        row[name], row[f"{name}_lo"], row[f"{name}_hi"] = mean_ci(kd.assign(_ok=kd._ok.astype(float)), "_ok", B, rng)
        row[f"far_{score}"] = d[d.split == "unknown"]._acc.mean()  # realised test false-accept rate
    row["n_known"], row["n_unknown"] = len(kn), int((df.split == "unknown").sum())
    return row


def confusions(pq, views):
    err = pq[(pq.split == "known") & ~pq.correct1]
    pair = err.apply(lambda r: tuple(sorted((r["product"], r["pred"]))), axis=1)
    c = pair.value_counts().rename("errors").reset_index()
    c[["a", "b"]] = pd.DataFrame(c["index"].tolist(), index=c.index)
    ptype = views.drop_duplicates("product").set_index("product").product_type
    c["type_a"], c["type_b"] = ptype[c.a].to_numpy(), ptype[c.b].to_numpy()
    return c[["a", "b", "type_a", "type_b", "errors"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="data/views")
    ap.add_argument("--emb", default="data/emb")
    ap.add_argument("--out", default="data/results")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confusion-config", default="dinov2_b,raw,5,proto")
    args = ap.parse_args()

    views = pd.read_csv(Path(args.views) / "views.csv")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    per_query, summary, breakdown = [], [], []
    for f in sorted(Path(args.emb).glob("*/*.npy")):
        backbone, arm = f.parent.name, f.stem
        emb = np.load(f)
        for k in KS:
            for method in ("proto", "knn"):
                pq = run_config(emb, views, k, method).assign(backbone=backbone, arm=arm, k=k, method=method)
                per_query.append(pq)
                cfg = dict(backbone=backbone, arm=arm, k=k, method=method)
                for qset, kind in (("synthetic", "query"), ("real", "real")):
                    d = pq[pq.kind == kind]
                    if len(d):
                        summary.append({**cfg, "queries": qset, **summarise(d, args.boot, rng)})
                syn = pq[pq.kind == "query"]
                for (bg, jit), d in syn.groupby(["bg", "jitter"]):
                    kd = d[d.split == "known"]
                    acc, lo, hi = mean_ci(kd, "correct1", args.boot, rng)
                    breakdown.append({**cfg, "bg": bg, "jitter": jit, "top1": acc, "top1_lo": lo, "top1_hi": hi,
                                      "auroc_top1": auroc(kd.top1, d[d.split == "unknown"].top1)})
        print(f"{backbone:13s} {arm:17s} done", flush=True)

    pq = pd.concat(per_query, ignore_index=True)
    pq.to_csv(out / "per_query.csv.gz", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(breakdown).to_csv(out / "breakdown.csv", index=False)

    b, a, k, m = args.confusion_config.split(",")
    sel = pq[(pq.backbone == b) & (pq.arm == a) & (pq.k == int(k)) & (pq.method == m)]
    confusions(sel, views).to_csv(out / "confusions.csv", index=False)

    lat = Path(args.emb) / "latency.csv"
    if lat.exists():
        dims = {f.parent.name: np.load(f, mmap_mode="r").shape[1] for f in Path(args.emb).glob("*/raw.npy")}
        ldf = pd.read_csv(lat)
        ldf["dim"] = ldf.backbone.map(dims)
        ldf.to_csv(out / "latency.csv", index=False)
    print(f"{len(summary)} summary rows -> {out}")


if __name__ == "__main__":
    main()
