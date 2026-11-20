# Product image similarity search

Can a frozen, off-the-shelf vision model recognise a product from a handful of catalogue photos,
and say "I don't know this one" when it isn't in the catalogue? No training and no fine-tuning:
embed the reference photos once, put them in a vector index, and match each query photo against
them.

This repo tests five backbones on 200 products and measures:

- **Closed-set accuracy.** Is the best match the right product? Reported as top-1/top-5,
  Recall@k and mAP.
- **Open-set rejection.** Can the system refuse products it has never seen?
- **Robustness.** Busy backgrounds, colour jitter, and whether removing the background first
  helps.
- **Cost.** Latency on GPU and CPU, and embedding size.

It builds on the sibling repo [`bgrm_bm`](../bgrm_bm), a background-removal benchmark. Its 200
product photos and hand-checked masks are the raw material here. Its models (u2net, BiRefNet)
are the background-removal arms.

## The setup

**Products.** There are 200 products from
[Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), the
same ones bgrm_bm uses. 160 are *known*: they are in the gallery. 40 are *unknown*: they only
ever appear as queries, and the system should reject them. All splits are by product.

**Views** (`data/make_views.py`, 256 x 256 px, 4,369 images in total):

| View | Count | What it is |
|---|---|---|
| `ref_0..ref_4` | 5 per product | Catalogue references. `ref_0` is the original white-background photo. `ref_1..4` are the product cut out with its true mask and re-placed on white or a light colour, with small shifts and mild lighting changes. |
| synthetic queries | 12 per product (2,400) | The product re-placed on a **clean** or **textured** (DTD) background, at colour-jitter levels 0, 1 and 2, with stronger crops, rotations and scaling than the references. |
| real queries | 969 | Other ABO photos of the same listing: different angle, packaging side, close-up or lifestyle scene. |

Synthetic queries are augmentations of the reference photo, so they are easy: a frozen
backbone matches them almost by pixel identity. Real queries are the honest test. Every result is reported for both
sets separately.

**Backbones** (`embed.py`). All are frozen, use the image embedding only, and are L2-normalised.

| Backbone | Embedding | Source |
|---|---|---|
| DINOv2 ViT-S/14 | 384-d CLS token | timm `vit_small_patch14_dinov2.lvd142m` |
| DINOv2 ViT-B/14 | 768-d CLS token | timm `vit_base_patch14_dinov2.lvd142m` |
| CLIP ViT-B/32 | 512-d | open_clip, OpenAI weights |
| SigLIP ViT-B/16 | 768-d | open_clip, `webli` weights |
| MobileNetV3-Large | 960-d pooled | torchvision, ImageNet weights |

**Retrieval** (`index.py`). A FAISS `IndexFlatIP` holds the first *k* references of each known
product, with k ∈ {1, 3, 5}. Because the vectors are normalised, inner product is cosine
similarity and the search is exact. Two classifiers, neither trained:

- **Prototype.** Average each product's references into one vector; the nearest prototype wins.
- **kNN.** The top-*k* reference images vote; ties go to the highest similarity.

Adding a product is `gallery.add(embeddings, label)`, and it is searchable immediately.

**Open-set** (`openset.py`). Each query gets a confidence score:

- the **top-1** similarity, or
- the **margin**: best product minus second-best.

The tests:

- **AUROC.** How well the score separates known from unknown queries. 1 is perfect; 0.5 is
  chance.
- **Accuracy at 5% FAR.** The threshold is set so that 5% of unknown queries get through. The
  number reported is the share of known queries that are accepted *and* identified correctly.

The threshold is never tuned on the queries it judges. Products are split into two folds; the
threshold is fitted on one fold's unknowns and applied to the other fold, both ways.

**Background removal arms.** These decide what the backbone sees. Each is applied to the
references and the queries alike.

- `raw`: the image as is.
- `u2net` / `birefnet-general`: the masks the bgrm_bm models predict, with the product
  composited onto white.
- `oracle`: the true mask composited onto white. This is the upper bound, available for
  synthetic queries only.

**Statistics.** Each 95% CI comes from a 1,000-draw bootstrap that resamples *products*, not
queries, because one product's queries are correlated.

## Results

These results use 5 references per product, the prototype classifier and the `raw` arm, unless
noted. Brackets show 95% CIs. The full grid is in `data/results/summary.csv`: 5 backbones × 4
arms × k ∈ {1,3,5} × 2 classifiers × 2 query sets. Plots and failure examples are in
[report.ipynb](report.ipynb).

**Real photos** (775 known-product queries, 194 unknown)

| Backbone | Top-1 ↑ | Top-5 ↑ | mAP ↑ | Open-set AUROC (margin) ↑ | Acc @ 5% FAR ↑ |
|---|---|---|---|---|---|
| **SigLIP ViT-B/16** | **0.770** [0.721–0.814] | **0.890** | **0.729** | **0.783** [0.713–0.840] | **0.274** |
| DINOv2 ViT-B/14 | 0.600 [0.547–0.654] | 0.794 | 0.629 | 0.635 [0.572–0.692] | 0.164 |
| DINOv2 ViT-S/14 | 0.563 [0.511–0.614] | 0.733 | 0.592 | 0.659 [0.590–0.718] | 0.115 |
| CLIP ViT-B/32 | 0.533 [0.485–0.585] | 0.723 | 0.501 | 0.658 [0.596–0.709] | 0.174 |
| MobileNetV3-L | 0.477 [0.428–0.526] | 0.644 | 0.496 | 0.650 [0.594–0.700] | 0.191 |

**Synthetic queries** (1,920 known, 480 unknown)

| Backbone | Top-1 ↑ | Top-5 ↑ | mAP ↑ | Open-set AUROC (margin) ↑ | Acc @ 5% FAR ↑ |
|---|---|---|---|---|---|
| **SigLIP ViT-B/16** | **0.924** [0.906–0.941] | **0.968** | **0.870** | **0.891** [0.846–0.929] | **0.607** |
| MobileNetV3-L | 0.864 [0.844–0.884] | 0.938 | 0.857 | 0.826 [0.792–0.859] | 0.540 |
| DINOv2 ViT-S/14 | 0.856 [0.835–0.875] | 0.929 | 0.860 | 0.818 [0.770–0.856] | 0.497 |
| DINOv2 ViT-B/14 | 0.853 [0.831–0.873] | 0.930 | 0.857 | 0.809 [0.761–0.851] | 0.465 |
| CLIP ViT-B/32 | 0.781 [0.751–0.810] | 0.881 | 0.706 | 0.798 [0.760–0.831] | 0.469 |

**Background removal, top-1**

| Backbone | Synthetic: raw | u2net | BiRefNet | oracle | Real: raw | u2net | BiRefNet |
|---|---|---|---|---|---|---|---|
| SigLIP ViT-B/16 | 0.924 | 0.917 | 0.962 | 0.979 | **0.770** | 0.674 | 0.717 |
| DINOv2 ViT-B/14 | 0.853 | 0.942 | 0.975 | 0.990 | **0.600** | 0.573 | 0.582 |
| DINOv2 ViT-S/14 | 0.856 | 0.936 | 0.974 | 0.989 | **0.563** | 0.539 | 0.557 |
| CLIP ViT-B/32 | 0.781 | 0.846 | 0.911 | 0.921 | 0.533 | 0.508 | **0.539** |
| MobileNetV3-L | 0.864 | 0.911 | 0.959 | 0.974 | **0.477** | 0.445 | 0.471 |

**Synthetic top-1 by background**

| Backbone | Clean: raw | Clean: BiRefNet | Textured: raw | Textured: BiRefNet |
|---|---|---|---|---|
| SigLIP ViT-B/16 | 0.982 | 0.975 | 0.866 | 0.950 |
| DINOv2 ViT-B/14 | 0.987 | 0.983 | 0.719 | 0.967 |
| DINOv2 ViT-S/14 | 0.995 | 0.984 | 0.717 | 0.965 |
| CLIP ViT-B/32 | 0.915 | 0.925 | 0.648 | 0.898 |
| MobileNetV3-L | 0.973 | 0.970 | 0.755 | 0.948 |

**Cost** (single image, batch 1; GPU is an RTX 3060 Laptop with 6 GB, CPU uses 6 threads)

| Backbone | GPU ms | CPU ms | Dim | Params (M) |
|---|---|---|---|---|
| MobileNetV3-L | 4.3 | 9.9 | 960 | 5.5 |
| CLIP ViT-B/32 | 5.5 | 44 | 512 | 87.8 (image tower) |
| DINOv2 ViT-S/14 | 5.6 | 50 | 384 | 21.6 |
| SigLIP ViT-B/16 | 11.9 | 124 | 768 | 92.9 (image tower) |
| DINOv2 ViT-B/14 | 13.9 | 140 | 768 | 85.7 |

Background removal adds 33 ms per image for u2net and 744 ms for BiRefNet (fp16, GPU). The
search itself takes microseconds at this scale.

### What the results say

- **SigLIP is clearly the best backbone**, on every query set and metric. On real photos it gets
  0.770 top-1 against 0.600 for the next best. The two CIs do not overlap. SigLIP was trained
  on web image–text pairs, which include a lot of product listings, and that seems to transfer.
- **Synthetic queries overstate accuracy by 15–33 points.** Every backbone scores
  0.78–0.92 on augmented copies of the reference photo, but 0.48–0.77 on other photos of the
  same product. MobileNetV3 shows the gap most: it ranks second on synthetic queries and last
  on real ones, so it matches pixels well but not products.
- **Background removal fixes busy backgrounds and hurts real photos.** On textured
  backgrounds, BiRefNet lifts DINOv2-B from 0.719 to 0.967. On clean backgrounds it changes
  nothing. On real photos it costs accuracy: SigLIP drops from 0.770 to 0.717 with BiRefNet
  and to 0.674 with u2net. Real alternate photos are mostly studio shots already. Masking
  removes context, such as the rest of a multipack or a scene, and a wrong mask removes part
  of the product. Better masks hurt less: BiRefNet beats u2net everywhere, matching bgrm_bm's
  ranking.
- **Open-set rejection is the weak spot.** Even for SigLIP, keeping false accepts at 5% means
  turning away most known products. On real photos only 27% of known queries are accepted and
  correct with the margin score, and 33% with the top-1 score. Margin gives the higher AUROC in
  most configs (SigLIP real: 0.783 vs 0.733). At the strict 5% operating point, though, top-1
  similarity often accepts more, so the choice of score depends on where the threshold sits.
  The achieved false-accept rate on held-out folds was 3–11%, close to the 5% target.
- **More references help SigLIP, not DINOv2.** From k=1 to 5, SigLIP's top-1 goes from 0.838
  to 0.924 on synthetic queries and from 0.741 to 0.770 on real ones. DINOv2-B stays within
  one point. References 1–4 are re-placements of the same photo, so they add little new
  information. Real multi-angle references would likely help more. Prototype and kNN agree to
  within a point everywhere.
- **Errors are near-duplicates.** The most confused pairs are two laundry detergents, two jars
  of dried herbs from the same brand, phone cases, and a chair vs. its matching ottoman. Many
  real-photo failures are nutrition labels or back panels, which share no appearance with the
  front-facing reference. See the failure grids in the report.

### Adding a product needs no retraining

The report's demo uses SigLIP with an unknown product (`0dee972c`):

- **Before adding it**, none of its 12 queries passes the open-set threshold.
- **The add** puts its 5 references in the index and takes 1.25 ms.
- **After adding it**, the gallery has 805 references across 161 products. 67% of the
  product's queries are accepted, and every accepted query is identified correctly.

## Running it

Everything uses [uv](https://docs.astral.sh/uv/). bgrm_bm must sit next to this repo
(`../bgrm_bm`), with its data and `~/.rembg/models` downloaded.

```bash
uv sync

# 1. views: references, synthetic and real queries, splits (reads ../bgrm_bm/data)
uv run python data/make_views.py

# 2. background-removal masks on the GPU (separate env: onnxruntime-gpu clashes with rembg[cpu])
GPU="uv run --no-project --python 3.13 --with onnxruntime-gpu[cuda,cudnn] --with pandas \
     --with pillow --with onnx --with onnxconverter-common"
$GPU python bgremove.py --model u2net
$GPU python bgremove.py --model birefnet-general --fp16      # ~45 min on a 3060

# 3. embeddings for every backbone x arm, plus latency
uv run python embed.py --latency

# 4. evaluation with bootstrap CIs (--boot 50 for a quick run)
uv run python eval.py

# 5. figures and tables -> report.ipynb (built and executed)
uv run python make_report.py

# tests
CUDA_VISIBLE_DEVICES= uv run pytest -q tests
```

Steps 2–4 run for a long time. Start them in `tmux` so they survive a closed terminal.
`bgremove.py` and `embed.py` skip outputs that already exist, so an interrupted run can be
resumed (`embed.py --force` recomputes).

## Files

| File | What it does |
|---|---|
| `data/make_views.py` | Builds reference, synthetic and real views and the known/unknown and fold splits |
| `bgremove.py` | Runs the bgrm_bm background-removal models on the GPU, with an fp16 option for BiRefNet |
| `embed.py` | Frozen backbone embeddings per arm, and the latency benchmark |
| `index.py` | FAISS gallery, prototype and kNN classifiers, `add` |
| `openset.py` | AUROC, threshold at a target FAR, cross-fitting |
| `eval.py` | Full grid of configs, bootstrap CIs, per-background breakdown, confusions |
| `make_report.py` → `report.ipynb` | Plots, tables, add-product demo, failure grids |
| `tests/test_core.py` | Toy-data tests for the retrieval, open-set and metric code |

## Limitations

- **The scale is small.** 160 known products is far from a real catalogue. `IndexFlatIP` is
  exact and fast here; at millions of images you would swap in IVF or HNSW, which `index.py`
  is written to allow.
- **The references come from one photo.** That is why the k sweep says little. A real
  catalogue with several angles per product would be a fairer gallery.
- **The parameter counts for CLIP and SigLIP** cover the image tower only; the text tower is not
  used. Using the text tower to re-rank matches against product titles is a natural next step
  and was not tried.
- **Some real queries cannot be matched from appearance alone**, such as a nutrition label vs. a
  front-of-pack reference. They are kept rather than removed by hand, so the
  real-photo numbers are a conservative estimate.
