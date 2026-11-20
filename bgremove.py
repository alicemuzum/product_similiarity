"""Background-removal masks for every view, using the benchmark's models on the GPU.

Usage (onnxruntime-gpu lives in a throwaway env so it never clashes with rembg's CPU build):
  uv run --no-project --python 3.13 --with "onnxruntime-gpu[cuda,cudnn]" --with pandas \
      --with pillow --with onnx --with onnxconverter-common python bgremove.py \
      --model birefnet-general --fp16

Runs the same ONNX files rembg and the bgrm_bm benchmark use (~/.rembg/models) with
onnxruntime's CUDA provider (CPU if no GPU). Pre- and post-processing copy rembg's sessions
exactly: LANCZOS resize, divide by image max, ImageNet normalisation, min-max scaling, and a
sigmoid for BiRefNet. --check N re-runs the model on N bench images and compares with the
benchmark's saved predictions in ../bgrm_bm/data/preds/bench/<model>/, which proves the outputs
match what was scored there.
Writes data/masks/<model>/<view>.png for every view in views.csv.

--fp16 runs a float16 copy of the model (converted once into data/models/, I/O kept float32).
BiRefNet at 1024 px needs it on a 6 GB card: in float32 its deformable-conv decoder asks for
822 MB buffers and runs out of memory whatever the arena settings. The float16 masks differ
from the benchmark's float32 ones by a mean absolute error of about 0.0001 (--check).
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

MODELS = {"u2net": dict(size=320, sigmoid=False), "birefnet-general": dict(size=1024, sigmoid=True)}
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def fp16_model(src, dst):
    if not dst.exists():
        import onnx
        from onnxconverter_common import float16

        dst.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True), str(dst))
    return dst


def preprocess(img, size):
    im = np.array(img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)).astype(np.float64)
    im = im / max(im.max(), 1e-6)
    return ((im - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def postprocess(out, img_size, sigmoid):
    pred = out[0, 0].astype(np.float64)
    if sigmoid:
        pred = 1 / (1 + np.exp(-pred))
    pred = np.clip((pred - pred.min()) / (pred.max() - pred.min()), 0, 1)
    mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
    return mask.resize(img_size, Image.Resampling.LANCZOS)


class Remover:
    def __init__(self, name, model_dir="~/.rembg/models", fp16=False):
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # CUDA/cuDNN from the nvidia-* wheels
        self.name, self.cfg = name, MODELS[name]
        path = Path(model_dir).expanduser() / name / f"{name}.onnx"
        if fp16:
            path = fp16_model(path, Path("data/models") / f"{name}-fp16.onnx")
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        if "CUDAExecutionProvider" in providers:
            # BiRefNet at 1024 px does not fit a 6 GB card with the default arena (it doubles
            # on every extension) and exhaustive cuDNN search (huge workspaces).
            providers[0] = ("CUDAExecutionProvider", {
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
                "cudnn_conv_use_max_workspace": "0",
            })
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False  # same setting the benchmark needed for BiRefNet's RAM
        self.sess = ort.InferenceSession(str(path), opts, providers=providers)
        self.input = self.sess.get_inputs()[0].name

    def __call__(self, img):
        out = self.sess.run(None, {self.input: preprocess(img, self.cfg["size"])})[0]
        return postprocess(out, img.size, self.cfg["sigmoid"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--views", default="data/views")
    ap.add_argument("--out", default="data/masks")
    ap.add_argument("--bm", default=os.environ.get("BGRM_DATA", "../bgrm_bm/data"))
    ap.add_argument("--fp16", action="store_true", help="run a float16 copy of the model (fits BiRefNet in 6 GB)")
    ap.add_argument("--check", type=int, default=5, help="compare N bench images with the benchmark's preds")
    args = ap.parse_args()

    views, bm = Path(args.views), Path(args.bm)
    df = pd.read_csv(views / "views.csv")
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    rm = Remover(args.model, fp16=args.fp16)
    print(f"{args.model}{' fp16' if args.fp16 else ''} on {rm.sess.get_providers()[0]}")

    if args.check:
        diffs = []
        for p in sorted((bm / "bench/images").iterdir())[: args.check]:
            a = np.asarray(rm(Image.open(p)), np.float32) / 255
            b = np.asarray(Image.open(bm / "preds/bench" / args.model / f"{p.stem}.png").convert("L"), np.float32) / 255
            diffs.append(np.abs(a - b).mean())
        print(f"mask MAE vs benchmark preds over {args.check} images: mean {np.mean(diffs):.4f}, max {np.max(diffs):.4f}")

    t0, n = time.time(), 0
    for v, p in zip(df.view, df.path):
        dst = out / f"{v}.png"
        if dst.exists():
            continue
        rm(Image.open(views / p)).save(dst)
        n += 1
    dt = time.time() - t0
    print(f"{args.model}: {n} masks in {dt:.0f}s ({dt / max(n, 1) * 1000:.0f} ms/img) -> {out}")


if __name__ == "__main__":
    main()
