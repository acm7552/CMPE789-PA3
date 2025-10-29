# viz_gt_ids.py
import os, random
from PIL import ImageDraw
import torch
from torchvision import transforms

# import your dataset
from MOT16 import MOT16   # <-- change to your dataset module

def _to_pil(img_tensor):
    return transforms.ToPILImage()(img_tensor.cpu())

def _color_from_id(i):
    # stable pseudo-random color per ID
    rnd = random.Random(int(i))
    return (rnd.randint(64,255), rnd.randint(64,255), rnd.randint(64,255))

def draw_boxes_with_ids(img_tensor, boxes_xyxy, ids=None, width=2):
    pil = _to_pil(img_tensor)
    draw = ImageDraw.Draw(pil)

    # normalize ids to a simple python list (or None)
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids is not None and not isinstance(ids, (list, tuple)):
        ids = list(ids)

    N = boxes_xyxy.shape[0]
    for k in range(N):
        x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[k].tolist()]

        label = None
        color = "red"  # fallback color

        if ids is not None and k < len(ids):
            idk = ids[k]
            if idk is not None:
                try:
                    id_int = int(idk)
                except (TypeError, ValueError):
                    id_int = None

                # In MOT, IDs <= 0 are often "ignored/no-ID"
                if id_int is not None and id_int > 0:
                    label = f"ID {id_int}"
                    color = _color_from_id(id_int)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if label:
            # small text offset inside the box
            draw.text((x1 + 3, y1 + 3), label, fill=color)

    return pil

if __name__ == "__main__":
    # Load one random TRAIN frame (has GT)
    ds = MOT16(root="MOT16/train", transform=transforms.ToTensor(), getGroundTruth=True)

    #idx = random.randrange(len(ds))
    idx = 10
    img, target, meta = ds[idx]  # assumes your dataset returns (image, target, meta)

    # Try both key styles, depending on how your dataset named it
    ids = meta.get("object_ids", None)
    if ids is None:
        ids = meta.get("object IDs", None)
    # if it's a torch tensor, make sure it's on CPU and list-like
    if isinstance(ids, torch.Tensor):
        ids = ids.cpu().tolist()

    print(f"Visualizing {meta['sequence']} frame {meta['frameID']} with {target['boxes'].shape[0]} boxes")

    pil = draw_boxes_with_ids(img, target["boxes"], ids=ids)
    os.makedirs("viz_out", exist_ok=True)
    out = os.path.join("viz_out", f"{meta['sequence']}_{int(meta['frameID'])}_gt.jpg")
    pil.save(out)
    print(f"Saved: {out}")
