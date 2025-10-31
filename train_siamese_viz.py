#!/usr/bin/env python3
import os, argparse, time, random, math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torchvision import transforms

# ---- your deps ----
from MOT16 import MOT16
from MOT16_triplet import TripletDataset

# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def to_cpu_np(t): return t.detach().to("cpu").numpy()

def save_fig(path):
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# =========================
# Embedding model
# =========================
# class EmbeddingNet(nn.Module):
#     def __init__(self, emb_dim=256, in_ch=3):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
#         self.pool  = nn.MaxPool2d(2)
#         self.gap   = nn.AdaptiveAvgPool2d((1,1))
#         self.fc    = nn.Linear(128, emb_dim)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = F.relu(self.conv3(x))
#         x = self.gap(x).flatten(1)
#         x = self.fc(x)
#         x = F.normalize(x, p=2, dim=1)
#         return x

class EmbeddingNet(nn.Module):
    def __init__(self, emb_dim=256, in_ch=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, emb_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    


# =========================
# Triplet loss
# =========================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, za, zp, zn):
        d_ap = torch.sum((za - zp)**2, dim=1)
        d_an = torch.sum((za - zn)**2, dim=1)
        return F.relu(d_ap - d_an + self.margin).mean()

# =========================
# Validation builder (pairs + gallery)
# =========================
def build_val_index(mot: MOT16, num_ids=12, max_per_id=10, crop_size=(128,64), pad=0.1, to_gray=False):
    """
    Build a small, fixed validation set:
      - pick up to num_ids identities (across sequences)
      - for each id, sample up to max_per_id crops
      - returns tensors, labels, and (seq,id) tags
    """
    # Index from TripletDataset logic (reuse filters)
    from torchvision.ops import box_convert
    idx = {}
    for img_path, seq, frame_id in mot.instances:
        df = mot.labels.get(seq, None)
        if df is None: continue
        rows = df[df["frame"] == frame_id]
        if mot.require_flag and "flag" in rows.columns:
            rows = rows[rows["flag"] == 1]
        if mot.keep_classes is not None and "class" in rows.columns:
            rows = rows[rows["class"].isin(mot.keep_classes)]
        if "visibility" in rows.columns:
            rows = rows[rows["visibility"] >= mot.min_visibility]
        if len(rows) == 0: continue
        xywh = torch.as_tensor(rows[["bb_left","bb_top","bb_width","bb_height"]].values, dtype=torch.float32)
        xyxy = box_convert(xywh, in_fmt="xywh", out_fmt="xyxy").tolist()
        ids  = rows["id"].tolist()
        for k, iid in enumerate(ids):
            idx.setdefault((seq, int(iid)), []).append((img_path, xyxy[k]))

    # choose IDs with >=2 samples
    candidates = [(k, v) for k, v in idx.items() if len(v) >= 2]
    random.shuffle(candidates)
    chosen = candidates[:num_ids]

    H,W = crop_size
    ts = []
    if to_gray: ts.append(transforms.Grayscale())
    ts += [transforms.Resize((H,W)), transforms.ToTensor()]
    tf = transforms.Compose(ts)

    def pad_crop(img, xyxy):
        w,h = img.size
        x1,y1,x2,y2 = xyxy
        pw = pad*(x2-x1); ph = pad*(y2-y1)
        x1=max(0,x1-pw); y1=max(0,y1-ph); x2=min(w-1,x2+pw); y2=min(h-1,y2+ph)
        return img.crop((x1,y1,x2,y2))

    images, labels, tags = [], [], []
    for (seq,iid), lst in chosen:
        random.shuffle(lst)
        for (p, b) in lst[:max_per_id]:
            img = Image.open(p).convert("RGB")
            img = pad_crop(img, b)
            images.append(tf(img))
            labels.append(iid)
            tags.append((seq, iid))
    if len(images) == 0:
        raise RuntimeError("Validation set is empty. Relax filters or increase num_ids.")
    data = torch.stack(images, dim=0)  # NxCxHxW
    labels = torch.tensor(labels, dtype=torch.int64)
    return data, labels, tags

# =========================
# Metrics & Viz
# =========================
def eval_verification(emb, labels, num_thresholds=200):
    """
    Simple verification: same-ID pairs vs different-ID pairs, sweep threshold on cosine similarity.
    Returns thresholds, accuracy curve, and best operating point.
    """
    emb = F.normalize(emb, p=2, dim=1)
    sim = emb @ emb.t()  # NxN
    N = sim.size(0)
    same = (labels.view(-1,1) == labels.view(1,-1))
    # only use upper triangle without diag
    mask = torch.ones_like(sim, dtype=torch.bool).triu(diagonal=1)
    sims = sim[mask].cpu().numpy()
    ys   = same[mask].cpu().numpy().astype(np.int32)

    ths = np.linspace(-1.0, 1.0, num_thresholds)
    accs = []
    for t in ths:
        pred = (sims >= t).astype(np.int32)
        accs.append((pred == ys).mean())
    accs = np.array(accs)
    best_idx = int(accs.argmax())
    return ths, accs, (ths[best_idx], accs[best_idx])

def plot_loss_curve(losses, path):
    plt.figure()
    plt.plot(losses, lw=2)
    plt.xlabel("Training step")
    plt.ylabel("Triplet loss")
    plt.title("Training Loss")
    save_fig(path)

def plot_accuracy_curve(ths, accs, best, path):
    plt.figure()
    plt.plot(ths, accs, lw=2)
    plt.scatter([best[0]],[best[1]], s=40, label=f"best t={best[0]:.2f}, acc={best[1]*100:.1f}%")
    plt.xlabel("Cosine similarity threshold")
    plt.ylabel("Verification accuracy")
    plt.title("Val Accuracy vs Threshold")
    plt.legend()
    save_fig(path)

def tsne_plot(emb, labels, tags, title, path):
    import numpy as np
    from sklearn.manifold import TSNE

    X = emb.detach().cpu().numpy()
    Y = labels.detach().cpu().numpy()

    # Perplexity must be < n_samples; keep it safe and >=5
    n = len(X)
    perp = max(5, min(30, max(5, (n - 1) // 3)))

    # Some sklearn versions don't support n_iter or learning_rate="auto".
    # Try the modern signature first, then fall back.
    try:
        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate=200,   # numeric is safest across versions
            perplexity=perp,
            n_iter=1000,         # may fail on some builds -> fallback below
            verbose=0,
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate=200,
            perplexity=perp,
            verbose=0,
        )

    Z = tsne.fit_transform(X)

    # --- plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    for u in np.unique(Y):
        m = (Y == u)
        plt.scatter(Z[m,0], Z[m,1], s=12, label=str(u), alpha=0.8)
    plt.legend(loc="best", fontsize=6, ncol=2)
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def retrieval_grid(images, emb, labels, out_path, k=5, max_queries=6):
    """
    Build a grid: each row = query | top-k nearest from gallery (cosine).
    """
    emb = F.normalize(emb, p=2, dim=1)
    sims = emb @ emb.t()  # NxN
    N = emb.size(0)
    # prepare PIL tiles
    to_pil = transforms.ToPILImage()
    # choose queries: up to max_queries random
    idxs = list(range(N))
    random.shuffle(idxs)
    queries = idxs[:max_queries]

    tiles = []
    for qi in queries:
        # rank by similarity (exclude self)
        order = torch.argsort(sims[qi], descending=True).tolist()
        order = [j for j in order if j != qi][:k]
        row_imgs = [to_pil(images[qi].cpu())] + [to_pil(images[j].cpu()) for j in order]
        row_labels = [int(labels[qi])] + [int(labels[j]) for j in order]
        row = concat_images_with_labels(row_imgs, row_labels)
        tiles.append(row)
    grid = vstack_images(tiles)
    ensure_dir(Path(out_path).parent)
    grid.save(out_path)

def concat_images_with_labels(imgs: List[Image.Image], labels: List[int]):
    # add small caption under each image
    w, h = imgs[0].size
    pad = 4; font = None
    try:
        font = ImageFont.load_default()
    except:
        pass
    tiles = []
    for im, lab in zip(imgs, labels):
        canvas = Image.new("RGB", (w, h+14), (255,255,255))
        canvas.paste(im, (0,0))
        dr = ImageDraw.Draw(canvas)
        txt = f"ID {lab}"
        dr.text((4,h+2), txt, fill=(0,0,0), font=font)
        tiles.append(canvas)
    return hstack_images(tiles)

def hstack_images(imgs: List[Image.Image]):
    h = max(im.size[1] for im in imgs)
    w = sum(im.size[0] for im in imgs)
    out = Image.new("RGB", (w, h), (255,255,255))
    x = 0
    for im in imgs:
        out.paste(im, (x, 0)); x += im.size[0]
    return out

def vstack_images(imgs: List[Image.Image]):
    w = max(im.size[0] for im in imgs)
    h = sum(im.size[1] for im in imgs)
    out = Image.new("RGB", (w, h), (255,255,255))
    y = 0
    for im in imgs:
        out.paste(im, (0, y)); y += im.size[1]
    return out

# =========================
# Train + Visualize
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="MOT16/train")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/reid_viz")
    ap.add_argument("--seed", type=int, default=42)

    # data/cropping
    ap.add_argument("--crop_h", type=int, default=16)
    ap.add_argument("--crop_w", type=int, default=16)
    ap.add_argument("--pad_ratio", type=float, default=0.10)
    ap.add_argument("--min_visibility", type=float, default=0.2)

    # train
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--epoch_len", type=int, default=20000)
    ap.add_argument("--log_every", type=int, default=50)

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    set_seed(args.seed)
    device = torch.device(args.device)

    # base MOT16 + triplet sampler
    mot = MOT16(root=args.train_root, transform=None, getGroundTruth=True,
                require_flag=True, min_visibility=args.min_visibility)
    triplets = TripletDataset(mot, crop_size=(args.crop_h, args.crop_w),
                              pad_ratio=args.pad_ratio, epoch_len=args.epoch_len,
                              same_seq_negative=True, to_gray=False, aug=True)
    loader = DataLoader(triplets, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=(args.device=='cuda'))

    # model/loss/optim
    model = EmbeddingNet(emb_dim=args.emb_dim, in_ch=3).to(device)
    criterion = TripletLoss(margin=args.margin)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # fixed validation set (small)
    val_images, val_labels, val_tags = build_val_index(
        mot, num_ids=12, max_per_id=8,
        crop_size=(args.crop_h, args.crop_w),
        pad=args.pad_ratio, to_gray=False
    )
    val_images = val_images.to(device)

    # --- initial eval & t-SNE
    model.eval()
    with torch.inference_mode():
        emb0 = model(val_images)
    ths0, accs0, best0 = eval_verification(emb0, val_labels.to(device))
    plot_accuracy_curve(ths0, accs0, best0, os.path.join(args.out_dir, "val_acc_curve_epoch0.png"))
    tsne_plot(emb0, val_labels.to(device), val_tags, "t-SNE (epoch 0)", os.path.join(args.out_dir, "tsne_epoch0.png"))
    retrieval_grid(val_images, emb0, val_labels.to(device), os.path.join(args.out_dir, "retrieval_epoch0.jpg"))

    # --- train
    losses = []
    step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        t0 = time.time()
        run = 0.0

        for a, p, n, _ in loader:
            a=a.to(device, non_blocking=True); p=p.to(device, non_blocking=True); n=n.to(device, non_blocking=True)
            za = model(a); zp = model(p); zn = model(n)
            loss = criterion(za, zp, zn)

            optim.zero_grad(); loss.backward(); optim.step()
            run += float(loss); losses.append(float(loss))
            step += 1
            if step % args.log_every == 0:
                print(f"[e{epoch}/{args.epochs} s{step}] loss={run/args.log_every:.4f}", flush=True)
                run = 0.0

        print(f"[epoch {epoch}] time={time.time()-t0:.1f}s")

        # per-epoch quick val
        model.eval()
        with torch.inference_mode():
            emb = model(val_images)
        ths, accs, best = eval_verification(emb, val_labels.to(device))
        plot_accuracy_curve(ths, accs, best, os.path.join(args.out_dir, f"val_acc_curve_epoch{epoch}.png"))
        tsne_plot(emb, val_labels.to(device), val_tags, f"t-SNE (epoch {epoch})", os.path.join(args.out_dir, f"tsne_epoch{epoch}.png"))
        retrieval_grid(val_images, emb, val_labels.to(device), os.path.join(args.out_dir, f"retrieval_epoch{epoch}.jpg"))

    # save final
    torch.save(model.state_dict(), os.path.join(args.out_dir, "siamese_triplet_murphy.pt"))
    plot_loss_curve(losses, os.path.join(args.out_dir, "train_loss.png"))
    print(f"[DONE] outputs in {args.out_dir}")

if __name__ == "__main__":
    main()
