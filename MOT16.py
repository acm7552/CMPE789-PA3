import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def _is_dir(p): 
    try: return os.path.isdir(p)
    except: return False

def _natural_sort(names):
    def key(s):
        stem = os.path.splitext(s)[0]
        try: return int(stem)
        except: return stem
    return sorted(names, key=key)

def _xywh_to_xyxy(xywh):
    xyxy = xywh.clone()
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2]
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3]
    return xyxy

def _clip_xyxy(boxes, w, h):
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h - 1)
    return boxes

def detection_collate(batch):
    images, targets, metas = zip(*batch)
    return images, targets, metas


class MOT16(Dataset):
    def __init__(self, root, transform=None, getGroundTruth=True,
                 keep_classes=(1,), require_flag=True, min_visibility=0.0):
        self.root = root
        self.transform = transform
        self.getGroundTruth = getGroundTruth
        self.keep_classes = set(keep_classes) if keep_classes else None
        self.require_flag = require_flag
        self.min_visibility = float(min_visibility)

        self.instances = []  # (img_path, seq_name, frame_id)
        self.labels = {}     # seq_name -> DataFrame or None

        sequences = sorted([d for d in os.listdir(self.root) if _is_dir(os.path.join(self.root, d))])

        for name in sequences:
            seq_path = os.path.join(self.root, name)
            images_folder = os.path.join(seq_path, "img1")
            if not _is_dir(images_folder):
                continue

            imageFiles = _natural_sort([f for f in os.listdir(images_folder) if f.lower().endswith((".jpg",".png",".jpeg"))])
            for imageFile in imageFiles:
                frame_number = int(os.path.splitext(imageFile)[0])
                self.instances.append((os.path.join(images_folder, imageFile), name, frame_number))

            gt_txt = os.path.join(seq_path, "gt", "gt.txt")
            if getGroundTruth and os.path.exists(gt_txt):
                df = pd.read_csv(gt_txt, header=None)
                df.columns = ["frame","id","bb_left","bb_top","bb_width","bb_height","flag","class","visibility"]
                self.labels[name] = df
            else:
                self.labels[name] = None

    def __len__(self):
        return len(self.instances)

    def _filter_rows(self, df, frame_id):
        rows = df[df["frame"] == frame_id]
        if self.require_flag and "flag" in rows:
            rows = rows[rows["flag"] == 1]
        if self.keep_classes is not None and "class" in rows:
            rows = rows[rows["class"].isin(self.keep_classes)]
        if "visibility" in rows:
            rows = rows[rows["visibility"] >= self.min_visibility]
        return rows

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_path, name, frame_number = self.instances[index]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        df_seq = self.labels[name]
        if df_seq is not None:
            rows = self._filter_rows(df_seq, frame_number)
            if len(rows) > 0:
                xywh = torch.tensor(rows[["bb_left","bb_top","bb_width","bb_height"]].values, dtype=torch.float32)
                boxes = _xywh_to_xyxy(xywh)
                boxes = _clip_xyxy(boxes, w, h)
                labels = torch.ones((boxes.size(0),), dtype=torch.int64)  # single class = 1
                obj_ids = torch.tensor(rows["id"].values, dtype=torch.int64)
            else:
                boxes = torch.zeros((0,4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                obj_ids = torch.zeros((0,), dtype=torch.int64)
            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([frame_number], dtype=torch.int64),
            }
            meta = {"sequence": name, "frameID": frame_number, "object_ids": obj_ids}
        else:
            target = {
                "boxes": torch.zeros((0,4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([frame_number], dtype=torch.int64),
            }
            meta = {"sequence": name, "frameID": frame_number, "object_ids": None}

        if self.transform is not None:
            img = self.transform(img)

        return img, target, meta

# Use this with your DataLoader:
# DataLoader(ds, batch_size=2, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))
