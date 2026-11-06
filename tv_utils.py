"""Torchvision detectors (Faster R-CNN, RetinaNet): dataset, model builders, training loop and inference.

Shared by 02_faster_rcnn.ipynb and 03_retinanet.ipynb, so both are trained under the same protocol:
same frames, same input size (long side 1280), same augmentation, optimiser, schedule and evaluation.
"""
import json
import math
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import roi_heads as tv_roi_heads
from torchvision.models.detection._utils import BoxCoder
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import generalized_box_iou_loss

from common import CHECKPOINTS, CLASSES, PROC, ROOT, SEED, coco_evaluate

LONG_SIDE = 1280          # input size used for all model comparisons (see 01_data, section 1.2)
NUM_CLASSES = len(CLASSES) + 1   # torchvision convention: label 0 is background, our classes are 1..8


# ----------------------------------------------------------------------------- data


def resize_long(img, long_side=LONG_SIDE):
    """Resize so that the long side equals `long_side`; returns the image and the scale factor."""
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s != 1:
        img = cv2.resize(img, (round(w * s), round(h * s)),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return img, s


def to_tensor(img_bgr):
    return torch.from_numpy(np.ascontiguousarray(img_bgr[..., ::-1])).permute(2, 0, 1).float().div_(255)


class YoloDetDataset(Dataset):
    """Reads data/proc/<ds>/images/<part>/*.jpg with YOLO labels and returns torchvision targets.

    Augmentation (train only): horizontal flip with p = 0.5. Kept simple and identical for both
    torchvision detectors; stronger augmentation is explored later in the final pipeline.
    """

    def __init__(self, ds="s10", part="train", train=True, long_side=LONG_SIDE):
        self.img_dir = PROC / ds / "images" / part
        self.lbl_dir = PROC / ds / "labels" / part
        self.stems = sorted(p.stem for p in self.img_dir.glob("*.jpg"))
        self.train, self.long_side = train, long_side

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img, _ = resize_long(cv2.imread(str(self.img_dir / f"{stem}.jpg")), self.long_side)
        h, w = img.shape[:2]
        lab = np.loadtxt(self.lbl_dir / f"{stem}.txt", ndmin=2).reshape(-1, 5)
        cx, cy, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
        boxes = np.c_[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2].astype(np.float32)
        if self.train and np.random.rand() < 0.5:
            img = img[:, ::-1]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        target = {"boxes": torch.from_numpy(boxes).reshape(-1, 4),
                  "labels": torch.from_numpy(lab[:, 0].astype(np.int64) + 1)}
        return to_tensor(img), target


def collate(batch):
    return tuple(zip(*batch))


def make_loader(ds, batch_size, shuffle, workers=4):
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                      generator=g, worker_init_fn=lambda k: np.random.seed(SEED + k), persistent_workers=workers > 0)


# ----------------------------------------------------------------------------- models


def _common_transform_kwargs():
    # images arrive already resized to long side 1280; min/max size keep them as they are (16:9 -> 1280 x 720)
    return dict(min_size=720, max_size=LONG_SIDE)


def build_faster_rcnn(anchor_sizes=None, pretrained=True, **kwargs):
    """Faster R-CNN ResNet50-FPN v2 (COCO weights) with a new 9-way box predictor.

    anchor_sizes: one size per FPN level (P2..P6); default torchvision uses (32, 64, 128, 256, 512).
    kwargs go to torchvision FasterRCNN, e.g. rpn_post_nms_top_n_test, box_batch_size_per_image.
    """
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    kw = dict(box_detections_per_img=500, box_score_thresh=0.001, **_common_transform_kwargs())
    kw.update(kwargs)
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
                                       weights_backbone=None, **kw)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    if anchor_sizes is not None:
        # 3 aspect ratios per location as before, so the pretrained RPN head still fits
        model.rpn.anchor_generator = AnchorGenerator(tuple((s,) for s in anchor_sizes),
                                                     ((0.5, 1.0, 2.0),) * len(anchor_sizes))
    return model


# --- box regression loss of the second stage: Smooth L1 (default) or GIoU


def fastrcnn_loss_giou(class_logits, box_regression, labels, regression_targets):
    """Same as torchvision fastrcnn_loss, but the box term is GIoU instead of Smooth L1.

    Predicted and target deltas are decoded relative to the same unit reference box. IoU and GIoU do not
    change under translation and per-axis scaling, so this equals the GIoU of the boxes in image space.
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    cls_loss = F.cross_entropy(class_logits, labels)
    pos = torch.where(labels > 0)[0]
    n = class_logits.shape[0]
    box_regression = box_regression.reshape(n, box_regression.size(-1) // 4, 4)
    pred, tgt = box_regression[pos, labels[pos]].float(), regression_targets[pos].float()
    ref = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=pred.device).expand(len(pos), 4)
    coder = BoxCoder((10.0, 10.0, 5.0, 5.0))   # torchvision RoIHeads weights
    box_loss = generalized_box_iou_loss(coder.decode_single(pred, ref), coder.decode_single(tgt, ref),
                                        reduction="sum") / labels.numel()
    return cls_loss, box_loss


_FASTRCNN_LOSSES = {"smooth_l1": tv_roi_heads.fastrcnn_loss, "giou": fastrcnn_loss_giou}


def set_roi_box_loss(name):
    """Switch the RoI head box loss globally ('smooth_l1' or 'giou')."""
    tv_roi_heads.fastrcnn_loss = _FASTRCNN_LOSSES[name]


# ----------------------------------------------------------------------------- inference


@torch.no_grad()
def predict(model, items, device="cuda", batch_size=4, long_side=LONG_SIDE):
    """items: list of (key, path to the ORIGINAL image). Returns dict key -> N x 6 [x1, y1, x2, y2, score, cls]
    in original pixels (cls 0-based)."""
    model.eval()
    out = {}
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        imgs, scales = [], []
        for _, p in chunk:
            img, s = resize_long(cv2.imread(str(p)), long_side)
            imgs.append(to_tensor(img).to(device)); scales.append(s)
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            res = model(imgs)
        for (key, _), r, s in zip(chunk, res, scales):
            b = r["boxes"].float().cpu().numpy() / s
            out[key] = np.c_[b, r["scores"].float().cpu().numpy(), r["labels"].cpu().numpy() - 1].astype(np.float64)
    return out


def val_items(gt_path):
    """(image_id, path) pairs of the validation frames listed in the COCO GT file."""
    gt = json.load(open(gt_path))
    return [(im["id"], ROOT / im["file_name"]) for im in gt["images"]]


def test_items(test_df):
    return [(r.image_id, ROOT / r.file_name) for r in test_df.itertuples()]


# ----------------------------------------------------------------------------- training


def train_detector(model, name, train_ds, val_gt, epochs=12, batch_size=4, lr=0.01, momentum=0.9,
                   weight_decay=1e-4, milestones=(8, 11), warmup_iters=300, device="cuda", eval_every=1,
                   resume=False):
    """SGD + linear warmup + step decay (x0.1 at `milestones`), mixed precision.

    After every `eval_every` epochs the model is evaluated on the validation frames with the COCO evaluator.
    Checkpoints: checkpoints/<name>_last.pt (for resuming) and checkpoints/<name>_best.pt (best val mAP).
    Returns the history table (loss and val metrics per epoch).
    """
    CHECKPOINTS.mkdir(exist_ok=True)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    loader = make_loader(train_ds, batch_size, shuffle=True)
    iters = len(loader)

    def lr_at(it):
        warm = min(1.0, (it + 1) / warmup_iters)
        decay = 0.1 ** sum(it >= m * iters for m in milestones)
        return lr * (0.001 + 0.999 * warm) * decay

    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    history, start, best = [], 0, -1.0
    last_ckpt = CHECKPOINTS / f"{name}_last.pt"
    if resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        history, start, best = ck["history"], ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {start}")

    vitems = val_items(val_gt)
    for ep in range(start, epochs):
        model.train()
        t0, run = time.time(), {}
        for k, (imgs, targets) in enumerate(loader):
            it = ep * iters + k
            for g in opt.param_groups:
                g["lr"] = lr_at(it)
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            targets = [{kk: v.to(device) for kk, v in t.items()} for t in targets]
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                losses = model(imgs, targets)
                loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"loss is {loss.item()} at iter {it}: {losses}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update()
            for kk, v in losses.items():
                run[kk] = run.get(kk, 0.0) + v.item()
        row = {"epoch": ep + 1, "lr": lr_at(it), "train_min": round((time.time() - t0) / 60, 1),
               **{kk: round(v / iters, 4) for kk, v in run.items()}}
        if (ep + 1) % eval_every == 0 or ep + 1 == epochs:
            m = coco_evaluate(val_gt, predict(model, vitems, device))
            row.update({kk: round(m[kk], 4) for kk in ["mAP", "AP50", "AP_small"]})
            if m["mAP"] > best:
                best = m["mAP"]
                torch.save(model.state_dict(), CHECKPOINTS / f"{name}_best.pt")
        history.append(row)
        print({kk: v for kk, v in row.items() if kk != "lr"})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "history": history, "epoch": ep, "best": best}, last_ckpt)
    return pd.DataFrame(history)


# ----------------------------------------------------------------------------- final evaluation of a run


@torch.no_grad()
def model_ms_per_image(model, items, device="cuda", n=50, warmup=5, long_side=LONG_SIDE):
    """Pure model time per image (batch 1, mixed precision); image decoding is excluded because the
    validation set contains 4K frames whose JPEG decoding would dominate the measurement."""
    model.eval()
    imgs = [to_tensor(resize_long(cv2.imread(str(p)), long_side)[0]).to(device) for _, p in items[:n]]
    with torch.autocast(device_type="cuda", enabled=device == "cuda"):
        for im in imgs[:warmup]:
            model([im])
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for im in imgs:
            model([im])
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / len(imgs) * 1000


def evaluate_run(model, name, val_gt, config, device="cuda"):
    """Load the best checkpoint of a run, evaluate it on the validation frames, measure speed and
    store everything in results/<name>.json. Validation predictions are cached for later analysis/ensembling."""
    import pickle
    from common import save_results
    model.load_state_dict(torch.load(CHECKPOINTS / f"{name}_best.pt", map_location=device))
    model.to(device)
    items = val_items(val_gt)
    preds = predict(model, items, device)
    pickle.dump(preds, open(CHECKPOINTS / f"{name}_val_preds.pkl", "wb"))
    metrics = coco_evaluate(val_gt, preds)
    ms = model_ms_per_image(model, items, device)
    res = save_results(name, metrics, ms_per_image=round(ms, 1), **config)
    return res, preds


# ----------------------------------------------------------------------------- Faster R-CNN analysis helpers


def set_rpn_proposals(model, n_post, n_pre=None):
    """Number of RPN proposals kept at TEST time (after NMS); pre-NMS top-k is per FPN level."""
    model.rpn._post_nms_top_n["testing"] = n_post
    model.rpn._pre_nms_top_n["testing"] = n_pre or max(1000, n_post)


@torch.no_grad()
def rpn_recall(model, gt_path, device="cuda", top_ns=(100, 300, 1000, 2000), iou_thr=0.5):
    """Class-agnostic recall of the first stage: share of GT boxes covered (IoU >= iou_thr) by at least one
    of the top-N RPN proposals, by object size. Shows how much the second stage could find at most."""
    from common import _iou
    old_post, old_pre = dict(model.rpn._post_nms_top_n), dict(model.rpn._pre_nms_top_n)
    set_rpn_proposals(model, max(top_ns), max(2000, max(top_ns)))
    model.eval()
    gt = json.load(open(gt_path))
    G = {}
    for a in gt["annotations"]:
        if not a["iscrowd"]:
            x, y, w, h = a["bbox"]
            G.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
    rows = []
    for im in gt["images"]:
        g = np.array(G.get(im["id"], []), float).reshape(-1, 4)
        if not len(g):
            continue
        img, s = resize_long(cv2.imread(str(ROOT / im["file_name"])))
        x = to_tensor(img).to(device)
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            images, _ = model.transform([x])
            props, _ = model.rpn(images, model.backbone(images.tensors))
        r = images.image_sizes[0][0] / x.shape[1]          # extra resize done by the model transform
        p = props[0].float().cpu().numpy() / r / s           # proposals in original pixels, sorted by objectness
        area = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        size = np.where(area < 32 ** 2, "small", np.where(area < 96 ** 2, "medium", "large"))
        for n in top_ns:
            best = _iou(g, p[:n])[0].max(1) if len(p) else np.zeros(len(g))
            rows += [{"top_n": n, "size": sz, "recalled": b >= iou_thr} for sz, b in zip(size, best)]
    model.rpn._post_nms_top_n.update(old_post); model.rpn._pre_nms_top_n.update(old_pre)
    df = pd.DataFrame(rows)
    out = df.pivot_table(index="top_n", columns="size", values="recalled", aggfunc="mean")
    out["all"] = df.groupby("top_n").recalled.mean()
    return out.reindex(columns=["small", "medium", "large", "all"]).round(3)