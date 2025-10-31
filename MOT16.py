import os
import torch
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


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
    def __init__(self, root, transform=None, getGroundTruth=True, useDetections=False,
                 keep_classes=(1,), require_flag=True, min_visibility=0.0):
        self.root = root
        self.transform = transform
        self.getGroundTruth = getGroundTruth
        self.useDetections = useDetections
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

            if getGroundTruth:
                txt_path = os.path.join(seq_path, "gt", "gt.txt")
            elif useDetections:
                txt_path = os.path.join(seq_path, "det", "det.txt")
            else:
                txt_path = None

            if getGroundTruth and os.path.exists(txt_path):
                df = pd.read_csv(txt_path, header=None)
                df.columns = ["frame","id","bb_left","bb_top","bb_width","bb_height","flag","class","visibility"]
                self.labels[name] = df
            else:
                self.labels[name] = None

    def __len__(self):
        return len(self.instances)

    def _filter_rows(self, df, frame_id, detection=False, score_thresh=0.0):
        rows = df[df["frame"] == frame_id]
        if not detection:  # ground truth
            if self.require_flag and "flag" in rows:
                rows = rows[rows["flag"] == 1]
            if self.keep_classes is not None and "class" in rows:
                rows = rows[rows["class"].isin(self.keep_classes)]
            if "visibility" in rows:
                rows = rows[rows["visibility"] >= self.min_visibility]
        else:  # detection
            if "score" in rows:
                rows = rows[rows["score"] >= score_thresh]
        return rows

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_path, name, frame_number = self.instances[index]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        df_seq = self.labels[name]
        if df_seq is not None:
            rows = self._filter_rows(df_seq, frame_number, detection=self.useDetections, score_thresh=0.3)
            if len(rows) > 0:
                xywh = torch.tensor(rows[["bb_left","bb_top","bb_width","bb_height"]].values, dtype=torch.float32)
                boxes = _xywh_to_xyxy(xywh)
                boxes = _clip_xyxy(boxes, w, h)
                labels = torch.ones((boxes.size(0),), dtype=torch.int64)  # single class = 1
                obj_ids = torch.tensor(rows["id"].values, dtype=torch.int64)

                #filtering invalid boxes because this code seems to create some
                valid   = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes   = boxes[valid]
                labels  = labels[valid]
                obj_ids = obj_ids[valid]

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

        # resizing the 1920x1080 to 640x360. same width as other smaller files, 
        # uniform learning maybe and speeds up learning?
        # basically making them 9 times smaller while hopefully maintaing same amount of learning
        # if w == 1920 and h == 1080:
        #     #print("resizing")
        #     img = img.resize((640, 360), Image.BILINEAR)

        #     # scale bounding boxes
        #     scale = 1/3
        #     if "boxes" in target and target["boxes"].numel() > 0:
        #         target["boxes"] = target["boxes"] * torch.tensor([scale, scale, scale, scale])


        if self.transform is not None:
            img = self.transform(img)

        if self.transform is None or isinstance(self.transform, torch.nn.Identity):
            pass  # no other transforms
        # if random.random() < 0.5 and target["boxes"].numel() > 0:
        #     img = TF.hflip(img)
        #     boxes = target["boxes"]
        #     boxes[:, [0,2]] = w - boxes[:, [2,0]]  # flip x coordinates
        #     target["boxes"] = boxes

        # check if bounding boxes are invalid because it will crash
        boxes = target["boxes"]
        if (boxes[:, 2] - boxes[:, 0] <= 0).any() or (boxes[:, 3] - boxes[:, 1] <= 0).any():
            print(f"Invalid box in image: {img_path}, sequence: {name}, frame: {frame_number}")
            print(f"Boxes: {boxes}")

        return img, target, meta

# Use this with your DataLoader:
# DataLoader(ds, batch_size=2, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))
