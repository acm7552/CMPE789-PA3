#!/usr/bin/env python3
import os, argparse, time, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms

# ---- your deps ----
from MOT16 import MOT16
from MOT16_triplet import TripletDataset

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def log(s):
    print(s, flush=True)

# -----------------------------
# Embedding model
# -----------------------------
class EmbeddingNet(nn.Module):
    """
    Simple CNN -> GAP -> FC -> L2-normalized embedding (default 256-D).
    in_ch=3 for color crops; set to 1 if you pass grayscale images.
    """
    def __init__(self, emb_dim=256, in_ch=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.gap   = nn.AdaptiveAvgPool2d((1,1))
        self.fc    = nn.Linear(128, emb_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # /2
        x = self.pool(F.relu(self.conv2(x)))  # /2
        x = F.relu(self.conv3(x))
        x = self.gap(x).flatten(1)            # B x 128
        x = self.fc(x)                        # B x D
        x = F.normalize(x, p=2, dim=1)        # L2-normalize (cosine-friendly)
        return x

# -----------------------------
# Triplet loss (standard)
# -----------------------------
class TripletLoss(nn.Module):
    """
    Standard triplet margin loss:
      L = max(0, ||za - zp||^2 - ||za - zn||^2 + margin)
    Assumes embeddings are L2-normalized.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, za, zp, zn):
        d_ap = torch.sum((za - zp) ** 2, dim=1)  # squared L2
        d_an = torch.sum((za - zn) ** 2, dim=1)
        loss = F.relu(d_ap - d_an + self.margin).mean()
        return loss

# -----------------------------
# Train loop
# -----------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    ensure_dir(args.out_dir)

    # 1) Base MOT16 (GT needed, with light GT filtering)
    mot = MOT16(root=args.train_root, transform=None, getGroundTruth=True,
                require_flag=True, min_visibility=args.min_visibility)

    # 2) Triplet dataset (crops & resize inside)
    triplets = TripletDataset(
        mot_ds=mot,
        crop_size=(args.crop_h, args.crop_w),
        min_per_id=2,
        pad_ratio=args.pad_ratio,
        same_seq_negative=True,
        to_gray=args.gray,
        aug=True,
        epoch_len=args.epoch_len
    )

    # 3) DataLoader (Windows-safe default workers=0)
    loader = DataLoader(triplets, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=(args.device=='cuda'),
                        persistent_workers=False)

    # 4) Model / loss / optim
    in_ch = 1 if args.gray else 3
    model = EmbeddingNet(emb_dim=args.emb_dim, in_ch=in_ch).to(device)
    criterion = TripletLoss(margin=args.margin)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    log(f"[reid] device={device}  batch={args.batch}  epochs={args.epochs}  "
        f"emb_dim={args.emb_dim}  crop={args.crop_h}x{args.crop_w}  margin={args.margin}")

    model.train()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running = 0.0
        steps = 0

        for a, p, n, meta in loader:
            a = a.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            n = n.to(device, non_blocking=True)

            # forward
            za = model(a)
            zp = model(p)
            zn = model(n)

            loss = criterion(za, zp, zn)

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            running += float(loss)
            steps += 1

            if steps % args.log_every == 0:
                avg = running / args.log_every
                log(f"[reid][e{epoch}/{args.epochs} s{steps}] loss={avg:.4f}")
                running = 0.0

            if args.max_steps and steps >= args.max_steps:
                break

        dt = time.time() - t0
        log(f"[reid][e{epoch}] epoch_time={dt:.1f}s")

    # 5) Save checkpoint
    ckpt = os.path.join(args.out_dir, "siamese_triplet.pt")
    torch.save(model.state_dict(), ckpt)
    log(f"[reid] saved → {ckpt}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="MOT16/train")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=0)  # safest on Windows
    ap.add_argument("--out_dir", default="runs/reid")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = no limit")
    ap.add_argument("--epoch_len", type=int, default=20000)

    # data / cropping
    ap.add_argument("--crop_h", type=int, default=128)
    ap.add_argument("--crop_w", type=int, default=64)
    ap.add_argument("--pad_ratio", type=float, default=0.10)
    ap.add_argument("--min_visibility", type=float, default=0.2)
    ap.add_argument("--gray", action="store_true", help="use grayscale crops (in_ch=1)")

    # model / train
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    train(args)

if __name__ == "__main__":
    main()
