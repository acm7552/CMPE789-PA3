# triplet_dataset.py
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.ops import box_convert

# expects your existing MOT16 dataset class:
# from MOT16 import MOT16

class TripletDataset(Dataset):
    """
    Random triplet sampler for Re-ID:
      - anchor (a) and positive (p): same identity, different samples
      - negative (n): different identity (same sequence by default)
    Produces 3 cropped tensors (CxHxW) suitable for triplet loss.

    Notes:
    - Uses MOT16 TRAIN split (GT required).
    - Applies basic GT filters via the underlying MOT16 (flag/class/visibility).
    - Crops are padded a bit, resized to (H, W), converted to tensor.
    """

    def __init__(
        self,
        mot_ds,                       # an instance of MOT16(getGroundTruth=True)
        crop_size: Tuple[int, int]=(128, 64),  # (H, W)
        min_per_id: int=2,            # at least 2 samples per ID to form (a,p)
        pad_ratio: float=0.10,        # 10% padding around bbox
        same_seq_negative: bool=True, # pick negative from same sequence (harder)
        to_gray: bool=False,          # if True → Grayscale
        aug: bool=True,               # simple augmentations on a/p (not on n)
        epoch_len: int=20000          # “virtual” length for one epoch
    ):
        super().__init__()
        self.mot = mot_ds
        self.H, self.W = crop_size
        self.min_per_id = min_per_id
        self.pad_ratio = pad_ratio
        self.same_seq_negative = same_seq_negative
        self.to_gray = to_gray
        self.epoch_len = int(epoch_len)

        # --- build index: seq -> id -> list[(img_path, xyxy)]
        self.index: Dict[str, Dict[int, List[Tuple[str, List[float]]]]] = {}
        for img_path, seq, frame_id in self.mot.instances:
            df = self.mot.labels.get(seq, None)
            if df is None:  # no GT in test split
                continue
            rows = df[df["frame"] == frame_id]
            # apply same filters as MOT16 dataset
            if self.mot.require_flag and "flag" in rows.columns:
                rows = rows[rows["flag"] == 1]
            if self.mot.keep_classes is not None and "class" in rows.columns:
                rows = rows[rows["class"].isin(self.mot.keep_classes)]
            if "visibility" in rows.columns:
                rows = rows[rows["visibility"] >= self.mot.min_visibility]
            if len(rows) == 0:
                continue

            xywh = torch.as_tensor(
                rows[["bb_left","bb_top","bb_width","bb_height"]].values,
                dtype=torch.float32
            )
            xyxy = box_convert(xywh, in_fmt="xywh", out_fmt="xyxy").tolist()
            ids  = rows["id"].tolist()
            for k, iid in enumerate(ids):
                self.index.setdefault(seq, {}).setdefault(int(iid), []).append(
                    (img_path, xyxy[k])
                )

        # filter out identities with < min_per_id
        for seq in list(self.index.keys()):
            self.index[seq] = {
                iid: lst for iid, lst in self.index[seq].items()
                if len(lst) >= self.min_per_id
            }
            if len(self.index[seq]) == 0:
                del self.index[seq]

        if not self.index:
            raise RuntimeError("No valid identities found for triplets. "
                               "Check GT filters / split path.")

        # convenience lists
        self.seqs = list(self.index.keys())
        self.ids_per_seq = {s: list(self.index[s].keys()) for s in self.seqs}

        # --- transforms
        base = []
        if self.to_gray:
            base.append(transforms.Grayscale())
        base += [transforms.Resize((self.H, self.W))]
        self.to_tensor = transforms.Compose(base + [transforms.ToTensor()])

        if aug:
            self.aug_tf = transforms.Compose([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
            ])
        else:
            self.aug_tf = None

    def __len__(self):
        # “virtual” length → you control epochs by DataLoader steps
        return self.epoch_len

    @staticmethod
    def _pad_crop(img: Image.Image, xyxy, pad_ratio: float):
        w, h = img.size
        x1, y1, x2, y2 = xyxy
        pw = pad_ratio * (x2 - x1)
        ph = pad_ratio * (y2 - y1)
        x1 = max(0, x1 - pw); y1 = max(0, y1 - ph)
        x2 = min(w - 1, x2 + pw); y2 = min(h - 1, y2 + ph)
        return img.crop((x1, y1, x2, y2))

    def _sample_positive(self, seq: str, iid: int):
        a_path, a_xyxy = random.choice(self.index[seq][iid])
        # ensure positive is not the exact same sample
        for _ in range(4):
            p_path, p_xyxy = random.choice(self.index[seq][iid])
            if p_path != a_path or p_xyxy != a_xyxy:
                break
        else:
            # fallback (rare): accept same source; training still works
            p_path, p_xyxy = random.choice(self.index[seq][iid])
        return (a_path, a_xyxy), (p_path, p_xyxy)

    def _sample_negative(self, seq: str, iid_anchor: int):
        if self.same_seq_negative and len(self.ids_per_seq[seq]) >= 2:
            iid_n = random.choice([x for x in self.ids_per_seq[seq] if x != iid_anchor])
            n_path, n_xyxy = random.choice(self.index[seq][iid_n])
            return n_path, n_xyxy
        # cross-sequence negative (easier)
        seq_n = random.choice(self.seqs)
        iid_n = random.choice(self.ids_per_seq[seq_n])
        n_path, n_xyxy = random.choice(self.index[seq_n][iid_n])
        return n_path, n_xyxy

    def __getitem__(self, _):
        # pick a sequence and an identity with >=2 samples
        seq = random.choice(self.seqs)
        iid = random.choice(self.ids_per_seq[seq])

        # (a, p)
        (a_path, a_xyxy), (p_path, p_xyxy) = self._sample_positive(seq, iid)
        # n
        n_path, n_xyxy = self._sample_negative(seq, iid)

        # load & crop
        a_img = Image.open(a_path).convert("RGB")
        p_img = Image.open(p_path).convert("RGB")
        n_img = Image.open(n_path).convert("RGB")

        a_img = self._pad_crop(a_img, a_xyxy, self.pad_ratio)
        p_img = self._pad_crop(p_img, p_xyxy, self.pad_ratio)
        n_img = self._pad_crop(n_img, n_xyxy, self.pad_ratio)

        # (optional) light aug on a/p only to improve invariance
        if self.aug_tf is not None:
            a_img = self.aug_tf(a_img)
            p_img = self.aug_tf(p_img)

        # resize + tensor (+optional grayscale)
        a = self.to_tensor(a_img)
        p = self.to_tensor(p_img)
        n = self.to_tensor(n_img)

        # return triplet tensors + optional debug info
        return a, p, n, {"seq": seq, "id": iid}
