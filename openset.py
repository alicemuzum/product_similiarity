"""Open-set rejection: is the query a product in the gallery at all?

Each query gets a score (top-1 similarity, or the top-1 minus top-2 margin). Known products
should score higher than unknown ones.

  auroc            how well the score separates known from unknown queries (1 = perfect,
                   0.5 = chance). Threshold-free. Supports per-sample weights, which the
                   product-level bootstrap uses.
  threshold_at_far the score cut-off that accepts `far` of the unknown queries it is fitted on.
  cross_fit        fit the threshold on one fold of products and apply it to the other, both
                   ways, so no query is ever judged by a threshold tuned on its own fold.
"""
import numpy as np


def auroc(pos, neg, w_pos=None, w_neg=None):
    """P(score of a known query > score of an unknown one), ties count half. Weighted Mann-Whitney."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    w_pos = np.ones(len(pos)) if w_pos is None else np.asarray(w_pos, float)
    w_neg = np.ones(len(neg)) if w_neg is None else np.asarray(w_neg, float)
    order = np.argsort(neg)
    neg, w_neg = neg[order], w_neg[order]
    cw = np.concatenate([[0.0], np.cumsum(w_neg)])
    below = cw[np.searchsorted(neg, pos, side="left")]
    ties = cw[np.searchsorted(neg, pos, side="right")] - below
    return float(np.sum(w_pos * (below + 0.5 * ties)) / (w_pos.sum() * w_neg.sum()))


def threshold_at_far(unknown_scores, far):
    """Smallest threshold t such that at most `far` of unknown scores are >= t (accepted)."""
    s = np.sort(np.asarray(unknown_scores, float))[::-1]
    n_accept = int(np.floor(far * len(s)))
    if n_accept >= len(s):
        return -np.inf
    # accept the top n_accept unknowns: threshold just above the next one
    return float(np.nextafter(s[n_accept], np.inf))


def cross_fit(score, is_known, fold, far):
    """Accept decision per query, with the threshold fitted on the other fold's unknown queries.

    Returns (accepted bool array, {fold: threshold applied to that fold}).
    """
    score, is_known, fold = map(np.asarray, (score, is_known, fold))
    accepted = np.zeros(len(score), bool)
    thresholds = {}
    for f in np.unique(fold):
        val_unknown = score[(fold != f) & ~is_known]
        t = threshold_at_far(val_unknown, far)
        thresholds[int(f)] = t
        m = fold == f
        accepted[m] = score[m] >= t
    return accepted, thresholds
