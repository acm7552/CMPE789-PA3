#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path

import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# === import your dataset ===
# If your class is MOT16 and returns (image, target, meta)
from MOT16 import MOT16 as MOT16Dataset   # <-- change module name
# simple collate for detection-style batches
def detection_collate(batch):
    images, targets, metas = zip(*batch)
    return images, targets, metas
# --- tiny viz helper (optional) ---
from PIL import ImageDraw
def draw_boxes(img_tensor, boxes, scores=None, score_thresh=0.5):
    pil = transforms.ToPILImage()(img_tensor.cpu())
    draw = ImageDraw.Draw(pil)
    for i, b in enumerate(boxes):
        s = float(scores[i]) if scores is not None else 1.0
        if s < score_thresh: 
            continue
        x1, y1, x2, y2 = [float(v) for v in b.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        draw.text((x1+3, y1+3), f"{s:.2f}", fill="lime")
    return pil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="MOT16 split dir (e.g., MOT16/test or MOT16/train)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--score_thresh", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--save_vis", action="store_true", help="save a few visualization images")
    ap.add_argument("--vis_limit", type=int, default=10)
    ap.add_argument("--out", default="runs/detections/mot16_detections.json")
    args = ap.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)

    get_gt = "train" in os.path.abspath(args.root)  # GT exists in train split
    tf = transforms.ToTensor()                      # detector expects tensors

    ds = MOT16Dataset(root=args.root, transform=tf, getGroundTruth=get_gt)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=detection_collate, num_workers=2)

    # --- pretrained detector ---
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")  # COCO-pretrained
    model.eval().to(args.device)

    detections = []
    vis_count = 0
    frames_done = 0

    with torch.inference_mode():
        for images, targets, metas in dl:
            # move images to device
            imgs_dev = [img.to(args.device) for img in images]
            preds = model(imgs_dev)  # list of dicts: boxes, labels, scores

            for img_t, pred, meta in zip(images, preds, metas):
                boxes  = pred["boxes"].cpu()
                scores = pred["scores"].cpu()
                labels = pred["labels"].cpu()

                # keep only confident detections
                keep = scores >= args.score_thresh
                boxes  = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                # record one JSON entry per frame
                detections.append({
                    "sequence": meta["sequence"],
                    "frame_id": int(meta["frameID"]),
                    "boxes": boxes.tolist(),      # xyxy
                    "scores": scores.tolist(),
                    "labels": labels.tolist(),    # COCO class ids (person=1)
                })

                # optional visualization
                if args.save_vis and vis_count < args.vis_limit:
                    pil = draw_boxes(img_t, boxes, scores, args.score_thresh)
                    vis_dir = Path("runs/vis")
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    pil.save(vis_dir / f"{meta['sequence']}_{int(meta['frameID'])}_det.jpg")
                    vis_count += 1

                frames_done += 1
                if args.max_frames and frames_done >= args.max_frames:
                    break

            if args.max_frames and frames_done >= args.max_frames:
                break

    # sort by (sequence, frame_id) for reproducibility
    detections.sort(key=lambda d: (d["sequence"], d["frame_id"]))

    with open(args.out, "w") as f:
        json.dump(detections, f)
    print(f"[OK] wrote detections → {args.out}  (frames: {len(detections)})")

if __name__ == "__main__":
    main()
