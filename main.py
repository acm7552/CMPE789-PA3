# test_mot16.py
import argparse
import os
from PIL import ImageDraw
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# import your class
from MOT16 import MOT16, detection_collate   # or: from your_file import MOT16 as MOT16Dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to MOT16 split folder, e.g. MOT16/train or MOT16/test")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--num", type=int, default=1, help="number of batches to inspect")
    ap.add_argument("--show", action="store_true", help="draw boxes on first image and save to disk")
    args = ap.parse_args()

    # minimal transforms (detector expects tensors; for a quick visual check we keep PIL for drawing later if needed)
    tf = transforms.ToTensor()

    # get_ground_truth=True for train, False for test
    get_gt = "train" in os.path.abspath(args.root)
    ds = MOT16(root=args.root, transform=tf, getGroundTruth=get_gt)

    print(f"[INFO] dataset size: {len(ds)} frames | split has GT: {get_gt}")

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=detection_collate)

    batches_inspected = 0
    for images, targets, metas in loader:
        print("\n=== Batch ===")
        for i in range(len(images)):
            img_t = images[i]                     # CxHxW tensor
            tgt   = targets[i]                    # dict with boxes/labels/image_id
            meta  = metas[i]                      # sequence/frameID/object_ids

            H, W = img_t.shape[-2], img_t.shape[-1]
            n_boxes = tgt["boxes"].shape[0]

            print(f"- {meta['sequence']} frame {int(meta['frameID'])}: "
                  f"{n_boxes} boxes | image_id={int(tgt['image_id'])} | size={W}x{H}")

            if n_boxes > 0:
                print(f"  first box (xyxy): {tgt['boxes'][0].tolist()}")
                print(f"  labels dtype: {tgt['labels'].dtype}")

            # optional: save a quick visualization for the first sample
            if args.show and batches_inspected == 0 and i == 0:
                # convert back to PIL for drawing
                pil = transforms.ToPILImage()(img_t)
                draw = ImageDraw.Draw(pil)
                for box in tgt["boxes"]:
                    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                out_path = "mot16_sample_vis.jpg"
                pil.save(out_path)
                print(f"[INFO] saved visualization: {out_path}")

        batches_inspected += 1
        if batches_inspected >= args.num:
            break

    print("[INFO] done.")

if __name__ == "__main__":
    main()
