"""Toy-data checks for the retrieval, open-set and evaluation logic. Run: uv run pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import average_precision, confusions, mean_ci, run_config  # noqa: E402
from index import Gallery, knn_classify, prototype_classify  # noqa: E402
from openset import auroc, cross_fit, threshold_at_far  # noqa: E402


def unit(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture
def gallery():
    # class a: two refs near e0; class b: one ref near e1
    g = Gallery(3)
    g.add(unit([[1, 0.1, 0], [1, -0.1, 0]]), "a")
    g.add(unit([[0, 1, 0]]), "b")
    return g


def test_prototype_and_knn(gallery):
    q = unit([[1, 0, 0.05], [0.1, 1, 0]])
    p = prototype_classify(gallery, q)
    assert list(p.pred) == ["a", "b"]
    assert np.all(p.margin > 0) and np.all(p.top1 <= 1 + 1e-6)
    k = knn_classify(gallery, q, 1)
    assert list(k.pred) == ["a", "b"]


def test_knn_vote_beats_single_closest():
    # one "b" ref is the single closest, but two "a" refs out-vote it at k=3
    g = Gallery(2)
    g.add(unit([[1, 0.30], [1, -0.30]]), "a")
    g.add(unit([[1, 0.25], [0, 1]]), "b")
    q = unit([[1, 0.2]])
    assert knn_classify(g, q, 1).pred[0] == "b"
    assert knn_classify(g, q, 3).pred[0] == "a"


def test_add_product_without_retraining(gallery):
    q = unit([[0, 0, 1]])
    assert prototype_classify(gallery, q).pred[0] != "c"
    gallery.add(unit([[0, 0.1, 1]]), "c")
    assert prototype_classify(gallery, q).pred[0] == "c"
    assert knn_classify(gallery, q, 1).pred[0] == "c"


def test_class_max_sim_is_best_ref(gallery):
    q = unit([[1, 0.1, 0]])
    cls, best, _, _ = gallery.class_max_sim(q)
    ref = gallery.index.reconstruct_n(0, gallery.index.ntotal)
    for j, c in enumerate(cls):
        assert best[0, j] == pytest.approx((ref[gallery.labels == c] @ q[0]).max(), abs=1e-6)


def test_auroc():
    assert auroc([2, 3], [0, 1]) == 1.0
    assert auroc([0, 1], [2, 3]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 200), rng.normal(0, 1, 300)
    brute = np.mean((pos[:, None] > neg) + 0.5 * (pos[:, None] == neg))
    assert auroc(pos, neg) == pytest.approx(brute)
    # integer weights equal duplicating samples
    wp, wn = rng.integers(0, 3, 200), rng.integers(0, 3, 300)
    assert auroc(pos, neg, wp, wn) == pytest.approx(auroc(np.repeat(pos, wp), np.repeat(neg, wn)))


def test_threshold_at_far():
    s = np.arange(100, dtype=float)
    t = threshold_at_far(s, 0.05)
    assert (s >= t).sum() == 5
    assert (s >= threshold_at_far(s, 0.0)).sum() == 0


def test_cross_fit_uses_other_fold():
    # fold 0 unknowns score high, fold 1 unknowns low: thresholds must differ by fold
    score = np.array([10, 11, 0.5, 0.6, 5, 5, 5, 5], float)
    known = np.array([0, 0, 0, 0, 1, 1, 1, 1], bool)
    fold = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    acc, th = cross_fit(score, known, fold, 0.0)
    assert th[0] > 0.6 and th[1] > 11
    assert list(acc[known]) == [True, True, False, False]


def test_average_precision():
    rel = np.array([[1, 0, 1, 0], [0, 1, 0, 0]], bool)
    assert average_precision(rel) == pytest.approx([(1 + 2 / 3) / 2, 0.5])


def test_mean_ci_brackets_mean():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"product": np.repeat(list("abcdefghij"), 5), "x": rng.random(50) > 0.3})
    m, lo, hi = mean_ci(df, "x", 500, rng)
    assert lo <= m <= hi


def toy_views():
    rows, emb = [], []
    rng = np.random.default_rng(0)
    centres = unit(rng.normal(size=(6, 16)))
    for i, c in enumerate(centres):
        pid, split = f"p{i}", "known" if i < 4 else "unknown"
        for r in range(5):
            rows.append(dict(view=f"{pid}_ref_{r}", product=pid, kind="ref", ref_idx=r, bg="white", jitter=0))
            emb.append(c + 0.05 * rng.normal(size=16))
        for r in range(4):
            rows.append(dict(view=f"{pid}_q_{r}", product=pid, kind="query", ref_idx=np.nan, bg="clean", jitter=r % 2))
            emb.append(c + 0.1 * rng.normal(size=16))
        for row in rows:
            row.setdefault("split", split)
            row.setdefault("fold", i % 2)
    df = pd.DataFrame(rows).assign(product_type="toy")
    return df, unit(np.array(emb))


@pytest.mark.parametrize("method", ["proto", "knn"])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_run_config_separable(method, k):
    views, emb = toy_views()
    pq = run_config(emb, views, k, method)
    assert len(pq) == 6 * 4
    kn = pq[pq.split == "known"]
    assert kn.correct1.all() and kn.r1.all()
    # unknown products can never be predicted correctly and should score lower
    assert auroc(kn.top1, pq[pq.split == "unknown"].top1) > 0.9


def test_confusions():
    views, emb = toy_views()
    pq = run_config(emb, views, 1, "proto")
    pq.loc[pq.index[0], "correct1"] = False
    pq.loc[pq.index[0], "pred"] = "p1"
    c = confusions(pq, views)
    assert list(c.columns) == ["a", "b", "type_a", "type_b", "errors"]
    assert c.iloc[0][["a", "b", "errors"]].tolist() == ["p0", "p1", 1]
