# Run instructions: Multiple Object Video Detection (HW2, UCU AI 2026)

Object detection on drone video (VisDrone2019-VID, 8 classes) for the
[Kaggle competition](https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026).

**Report:** see [report.md](report.md) (published with GitHub Pages).

Team: Yuliia Sova, Oksana Dziuba.

## Final solution

RT-DETR-L (Ultralytics, COCO-pretrained), input 1280, trained for 1.25 GPU-hours on frames of all 63 labelled clips
(train + val, every 10th frame, ignored regions masked). At test time the full frame is combined with 2x2 overlapping
tiles (sliced inference) by class-wise NMS.

| Selected submission | File | Validation mAP* | Public LB |
|---|---|---|---|
| G: RT-DETR + tiles | `submissions/final_G_scale.csv` | 0.306 | 0.294 |
| A: RT-DETR | `submissions/final_A_rtdetr.csv` | 0.297 | 0.295 |

\* COCO mAP@[.5:.95] on 420 frames of the 7 held-out validation clips, measured with the same recipe and the model
trained on train clips only (the final model also saw the validation clips).

## Repository structure

| Path | Content |
|---|---|
| `01_data.ipynb` | download, EDA, preprocessing, validation ground truth |
| `02_faster_rcnn.ipynb` | two-stage anchor-based: Faster R-CNN R50-FPN v2 (torchvision) |
| `03_retinanet.ipynb` | one-stage anchor-based: RetinaNet R50-FPN v2 (torchvision) |
| `04_yolo.ipynb` | one-stage anchor-free: YOLO11m (Ultralytics) |
| `05_rtdetr.ipynb` | query-based: RT-DETR-L (Ultralytics) |
| `06_final.ipynb` | final pipeline: inference and data experiments, temporal smoothing, final models, submissions, summary |
| `common.py` | paths, data loading, frame selection, dataset building, COCO evaluation with ignored regions, WBF/NMS merging, temporal rescoring, submission writer |
| `tv_utils.py` | torchvision dataset, model builders, training loop, inference, RPN / anchor analysis |
| `yolo_utils.py` | Ultralytics training with a time budget, inference (incl. flip TTA and tiles), evaluation |
| `configs/` | custom model configs: YOLO11m with a P2 head, RT-DETR-L with 500 queries / without denoising |
| `results/` | metrics of every run (`<run>.json`) and per-epoch logs of the torchvision runs |

Not in the repository (created by the notebooks, see `.gitignore`): `data/` (raw and processed data),
`runs/` (Ultralytics runs and weights), `checkpoints/` (torchvision weights, cached predictions), `submissions/`.

## Setup

Tested on Ubuntu 24.04, Python 3.12.3, NVIDIA RTX 5070 Laptop GPU (8 GB), CUDA 13.

```bash
git clone <repo url> && cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Kaggle API access (needed to download the data and to submit): create a token at kaggle.com → Settings → API and
either run `kaggle auth login` or save the token to `~/.kaggle/access_token`. Join the competition first.
About 25 GB of free disk space are needed while the archive is being extracted (the archive is deleted afterwards).

## How to reproduce

Run the notebooks in order with "Run All". Every notebook can be re-run: finished trainings are detected and
skipped (`tv_utils.train_detector` loads the best checkpoint, `yolo_utils.train_ultra` checks `runs/<name>/DONE`),
interrupted ones resume from the last epoch, and predictions are cached in `checkpoints/`.

| Notebook | What it produces | GPU time (RTX 5070 Laptop) |
|---|---|---|
| `01_data` | `data/raw`, `data/proc/s10` (2446 train / 420 val frames), `data/proc/val_gt.json` | none (~30 min download) |
| `02_faster_rcnn` | 3 runs: baseline, anchors 16–256, GIoU loss | ~5 h |
| `03_retinanet` | 3 runs: baseline, anchors 16–256, focal gamma 1.5 | ~4.5 h |
| `04_yolo` | 3 runs: 1280, 1536, P2 head (1.25 h budget each) | ~4 h |
| `05_rtdetr` | 3 runs: baseline, 500 queries, no denoising (1.25 h budget each) | ~4 h |
| `06_final` | data experiments, final models, test predictions, submissions | ~6 h |

**Final submissions only** (needs `01_data` first): run `06_final.ipynb`. Section 1 needs `rtdetr_base` and
`yolo_1536` from `05`/`04`; sections 2–5 train the remaining models; section 7 writes `final_G_scale.csv`.
Set `SUBMIT = False` in the setup cell to write the CSV files without sending them to Kaggle.

### Settings and seeds

* Seed 42 everywhere (`common.seed_everything`, DataLoader generator, Ultralytics `seed=42, deterministic=True`).
  Some CUDA kernels (deformable attention in RT-DETR) have no deterministic implementation, so repeated runs can
  differ slightly.
* Validation: the organisers' `val/` split (7 clips), 60 evenly spaced frames per clip; train frames: every 10th
  frame. Splits are by clip, never by frame.
* torchvision models: input long side 1280, batch 2, SGD (lr 0.005 Faster R-CNN, 0.0025 RetinaNet), 12 epochs,
  lr x0.1 after epochs 8 and 11, mixed precision.
* Ultralytics models: fixed budget of 1.25 GPU-hours (`time=1.25`), default augmentation, physical batch 1–2
  (nominal batch 64 through gradient accumulation).

## Sources

Code and ideas we used (details and discussion in the report):

* Ultralytics YOLO11 and RT-DETR: https://github.com/ultralytics/ultralytics (AGPL-3.0)
* torchvision detection models and reference scripts: https://github.com/pytorch/vision
* pycocotools (COCO evaluation): https://github.com/cocodataset/cocoapi
* ensemble-boxes (Weighted Boxes Fusion): https://github.com/ZFTurbo/Weighted-Boxes-Fusion
* Papers: Faster R-CNN (Ren et al., 2015), FPN (Lin et al., 2017), RetinaNet / focal loss (Lin et al., 2017),
  GIoU (Rezatofighi et al., 2019), YOLOv2 anchor k-means (Redmon & Farhadi, 2017), TOOD (Feng et al., 2021),
  DETR (Carion et al., 2020), DN-DETR (Li et al., 2022), RT-DETR (Zhao et al., 2023), WBF (Solovyev et al., 2019),
  SAHI (Akyon et al., 2022), Seq-NMS (Han et al., 2016), TIDE (Bolya et al., 2020), VisDrone (Zhu et al., 2020).
* The P2 head config follows Ultralytics `yolov8-p2.yaml`.
* Parts of the code were written with the help of an AI assistant (Claude, Anthropic); all experiments, results
  and the report are our own.
