"""Ultralytics detectors (YOLO11, RT-DETR): training with a fixed time budget, inference and evaluation
with the shared COCO evaluator (common.coco_evaluate), so the numbers are comparable with torchvision models.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from common import CHECKPOINTS, PROC, PROJECT, ROOT, SEED, coco_evaluate, save_results

RUNS = PROJECT / "runs"          # ultralytics run folders (weights, results.csv, plots); not committed
DATA_YAML = PROC / "s10" / "data.yaml"


def _model(weights_or_cfg, kind="yolo"):
    from ultralytics import RTDETR, YOLO
    return (RTDETR if kind == "rtdetr" else YOLO)(str(weights_or_cfg))


def train_ultra(name, model_cfg, pretrained=None, kind="yolo", hours=1.25, imgsz=1280, batch=4, **overrides):
    """Train an ultralytics model for a fixed wall-clock budget (`hours`; ultralytics then sets the number
    of epochs and the lr schedule to fit it). Interrupted runs are resumed, finished runs are skipped.

    model_cfg:  model yaml or .pt (e.g. "yolo11m.pt", "configs/yolo11m-p2.yaml", "rtdetr-l.pt")
    pretrained: weights to transfer into a yaml-defined model (e.g. "yolo11m.pt" for the P2 variant)
    overrides:  any ultralytics train argument (lr0, box, cls, mosaic, ...)
    Returns the path to best.pt.
    """
    run = RUNS / name
    best, last = run / "weights" / "best.pt", run / "weights" / "last.pt"
    done = run / "DONE"
    if done.exists() and best.exists():
        print(f"{name}: already trained, using {best}")
        return best
    if last.exists():
        print(f"{name}: resuming from {last}")
        _model(last, kind).train(resume=True)
    else:
        model = _model(model_cfg, kind)
        if pretrained:
            model.load(pretrained)
        args = dict(data=str(DATA_YAML), imgsz=imgsz, batch=batch, time=hours, epochs=100, seed=SEED,
                    deterministic=True, project=str(RUNS), name=name, exist_ok=True, workers=4,
                    patience=1000, plots=True, cache=False)
        args.update(overrides)
        model.train(**args)
    done.touch()
    return best


def history(name):
    """Per-epoch log of an ultralytics run (losses and ultralytics' own val metrics)."""
    h = pd.read_csv(RUNS / name / "results.csv")
    h.columns = [c.strip() for c in h.columns]
    return h


def predict_ultra(model, items, imgsz=1280, conf=0.001, iou=0.6, max_det=500, batch=4, augment=False):
    """items: list of (key, path to the ORIGINAL image). Returns dict key -> N x 6 [x1, y1, x2, y2, score, cls]
    in original pixels, plus the mean inference + postprocess (NMS) time per image in ms."""
    out, ms = {}, []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        res = model.predict([str(p) for _, p in chunk], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                            augment=augment, verbose=False)
        for (key, _), r in zip(chunk, res):
            b = r.boxes
            out[key] = np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()].astype(np.float64)
            ms.append(r.speed["inference"] + r.speed["postprocess"])
    return out, float(np.mean(ms))


def val_items(gt_path):
    gt = json.load(open(gt_path))
    return [(im["id"], ROOT / im["file_name"]) for im in gt["images"]]


def test_items(test_df):
    return [(r.image_id, ROOT / r.file_name) for r in test_df.itertuples()]


def evaluate_ultra(name, weights, val_gt, config, kind="yolo", imgsz=1280, **predict_kw):
    """Evaluate a trained run with the shared COCO evaluator and store results/<name>.json.
    Speed = model inference + NMS per image at batch 1 (measured by ultralytics, decoding excluded)."""
    model = _model(weights, kind)
    items = val_items(val_gt)
    preds, _ = predict_ultra(model, items, imgsz=imgsz, **predict_kw)
    CHECKPOINTS.mkdir(exist_ok=True)
    pickle.dump(preds, open(CHECKPOINTS / f"{name}_val_preds.pkl", "wb"))
    metrics = coco_evaluate(val_gt, preds)
    timing_kw = {k: v for k, v in predict_kw.items() if k != "batch"}
    _, ms = predict_ultra(model, items[:50], imgsz=imgsz, batch=1, **timing_kw)
    h = history(name)
    res = save_results(name, metrics, ms_per_image=round(ms, 1), epochs_done=len(h),
                       train_hours=round(float(h["time"].iloc[-1]) / 3600, 2) if "time" in h else None,
                       imgsz=imgsz, **config)
    return res, preds


def predict_ultra_hflip(model, items, imgsz=1280, conf=0.001, max_det=500, batch=4):
    """Predictions on horizontally flipped images, mapped back to the original coordinates (flip TTA).
    Combined with the normal predictions via common.wbf_merge."""
    import cv2
    out = {}
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        imgs = [cv2.imread(str(p))[:, ::-1].copy() for _, p in chunk]
        res = model.predict(imgs, imgsz=imgsz, conf=conf, max_det=max_det, verbose=False)
        for (key, _), r, im in zip(chunk, res, imgs):
            w = im.shape[1]
            b = r.boxes.xyxy.cpu().numpy().copy()
            b[:, [0, 2]] = w - b[:, [2, 0]]
            out[key] = np.c_[b, r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()].astype(np.float64)
    return out


def predict_ultra_tiles(model, items, imgsz=1280, grid=(2, 2), overlap=0.2, conf=0.001, max_det=500,
                        edge_px=3, batch=4):
    """Sliced inference (idea of SAHI, Akyon et al. 2022): the frame is cut into grid[0] x grid[1] overlapping
    tiles, every tile is resized to `imgsz` by the model (so small objects become 1.5-3x larger), and the boxes are
    mapped back to the frame. Boxes that touch an inner tile border are dropped: they are usually cut objects,
    which the neighbouring tile (thanks to the overlap) or the full-frame pass sees whole.
    Returns dict key -> N x 6 [x1, y1, x2, y2, score, cls] in original pixels (tiles only; merge with the
    full-frame predictions afterwards)."""
    import cv2
    nx, ny = grid
    out = {}
    for key, p in items:
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        tw, th = int(np.ceil(W / (nx - (nx - 1) * overlap))), int(np.ceil(H / (ny - (ny - 1) * overlap)))
        xs = [round(i * (W - tw) / max(nx - 1, 1)) for i in range(nx)]
        ys = [round(j * (H - th) / max(ny - 1, 1)) for j in range(ny)]
        tiles = [(x, y) for y in ys for x in xs]
        crops = [img[y:y + th, x:x + tw] for x, y in tiles]
        dets = []
        for i in range(0, len(crops), batch):
            res = model.predict(crops[i:i + batch], imgsz=imgsz, conf=conf, max_det=max_det, verbose=False)
            for (x, y), r in zip(tiles[i:i + batch], res):
                b = r.boxes.xyxy.cpu().numpy()
                d = np.c_[b, r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()]
                # drop boxes touching a tile border that lies inside the frame
                inner = np.zeros(len(d), bool)
                if x > 0:          inner |= d[:, 0] <= edge_px
                if y > 0:          inner |= d[:, 1] <= edge_px
                if x + tw < W:     inner |= d[:, 2] >= tw - edge_px
                if y + th < H:     inner |= d[:, 3] >= th - edge_px
                d = d[~inner]
                d[:, [0, 2]] += x
                d[:, [1, 3]] += y
                d[:, [0, 2]] = d[:, [0, 2]].clip(0, W)
                d[:, [1, 3]] = d[:, [1, 3]].clip(0, H)
                dets.append(d)
        out[key] = np.concatenate(dets).astype(np.float64) if dets else np.zeros((0, 6))
    return out
