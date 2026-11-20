"""FAISS gallery with two few-shot classifiers that need no training.

Embeddings are L2-normalised, so inner product = cosine similarity and IndexFlatIP is exact
cosine search. At 160 products x 5 references an approximate index would buy nothing; FAISS is
used so the same code scales to a catalogue where IVF/HNSW would be swapped in.

  Prototype  average each product's k reference embeddings, re-normalise, nearest prototype wins.
  kNN        search all reference embeddings; the top-k neighbours (k = references per product)
             vote, ties broken by the highest similarity. With k = 1 this is plain 1-NN.

Adding a product is `Gallery.add(embeddings, label)`: nothing is retrained, the new vectors
are searchable immediately.
"""
from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class Ranking:
    """Per-query output of a classifier.

    classes  (n_q, n_classes) product labels, best first
    sim      (n_q, n_classes) cosine similarity behind each ranked class (prototype: to the
             prototype; kNN: best reference of that class)
    """

    classes: np.ndarray
    sim: np.ndarray

    @property
    def pred(self):
        return self.classes[:, 0]

    @property
    def top1(self):
        """Open-set score: similarity of the predicted class."""
        return self.sim[:, 0]

    @property
    def margin(self):
        """Open-set score: similarity gap between the predicted class and the best other class."""
        return self.sim[:, 0] - self.sim[:, 1]


class Gallery:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.labels = np.empty(0, dtype=object)

    def add(self, emb, labels):
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        labels = np.broadcast_to(np.asarray(labels, dtype=object), (len(emb),))
        self.index.add(emb)
        self.labels = np.concatenate([self.labels, labels])
        return self

    def search(self, q, k=None):
        """Image-level retrieval: (sims, gallery row ids), best first."""
        k = self.index.ntotal if k is None else k
        return self.index.search(np.ascontiguousarray(q, dtype=np.float32), k)

    @property
    def classes(self):
        return np.array(sorted(set(self.labels)), dtype=object)

    def prototypes(self):
        cls = self.classes
        protos = np.stack([self.index.reconstruct_n(0, self.index.ntotal)[self.labels == c].mean(0) for c in cls])
        protos /= np.linalg.norm(protos, axis=1, keepdims=True)
        return cls, protos.astype(np.float32)

    def class_max_sim(self, q):
        """(classes, (n_q, n_classes) best similarity to any reference of each class)."""
        cls = self.classes
        col = {c: j for j, c in enumerate(cls)}
        sims, ids = self.search(q)
        out = np.full((len(q), len(cls)), -np.inf, np.float32)
        lab_col = np.array([col[c] for c in self.labels])
        cols = lab_col[ids]
        # unbuffered max: plain fancy assignment with repeated indices has no defined write order
        rows = np.repeat(np.arange(len(q)), ids.shape[1])
        np.maximum.at(out, (rows, cols.ravel()), sims.ravel())
        return cls, out, sims, cols


def prototype_classify(gallery, q):
    cls, protos = gallery.prototypes()
    idx = faiss.IndexFlatIP(protos.shape[1])
    idx.add(protos)
    sims, ids = idx.search(np.ascontiguousarray(q, dtype=np.float32), len(cls))
    return Ranking(cls[ids], sims)


def knn_classify(gallery, q, k):
    """Majority vote over the top-k references; ties by best similarity; rest ranked by best similarity."""
    cls, best, sims, cols = gallery.class_max_sim(q)
    votes = np.zeros_like(best)
    rows = np.repeat(np.arange(len(q)), k)
    np.add.at(votes, (rows, cols[:, :k].ravel()), 1.0)
    # similarities lie in [-1, 1], so a vote count always outranks any similarity difference
    order = np.argsort(-(votes * 4 + best), axis=1, kind="stable")
    return Ranking(cls[order], np.take_along_axis(best, order, axis=1))
