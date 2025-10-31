#!/usr/bin/env python3
import os, argparse, time, random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torchvision import transforms


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
# Market-1501 helpers
# =========================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def _is_img(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS

def parse_pid_cam(fname: str) -> Tuple[int, int]:
    """Market-1501 pattern: '0002_c1s1_...jpg' -> (2,1). Returns (pid, camid)."""
    name = os.path.splitext(os.path.basename(fname))[0]
    parts = name.split('_')
    try:
        pid = int(parts[0])
    except Exception:
        pid = -1
    try:
        cam_part = parts[1] if len(parts) > 1 else ""
        camid = int(cam_part.split('c')[1].split('s')[0])
    except Exception:
        camid = 1
    return pid, camid


class Market1501Split(Dataset):
    """
    A flat folder split for Market-1501, e.g.:
      - bounding_box_train / gt_bbox
      - bounding_box_test
      - query / gt_query
    Builds a list of (path, pid, camid). By default, skips pid == -1 (distractors).
    """
    def __init__(self, root_dir: str, skip_neg_one: bool = True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int, int]] = []

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Split folder not found: {root_dir}")

        minus1, total = 0, 0
        for fname in sorted(os.listdir(root_dir)):
            path = os.path.join(root_dir, fname)
            if not os.path.isfile(path) or not _is_img(path):
                continue
            pid, cam = parse_pid_cam(fname)
            total += 1
            if pid == -1:
                minus1 += 1
                if skip_neg_one:
                    continue
            self.samples.append((path, pid, cam))

        if len(self.samples) == 0:
            first10 = sorted(os.listdir(root_dir))[:10]
            print(f"[WARN] No usable images in {root_dir}. First 10 entries: {first10}")
        else:
            frac = minus1 / max(1, total)
            if frac > 0.5 and skip_neg_one:
                print(f"[INFO] {minus1}/{total} (~{frac*100:.1f}%) files are pid=-1 and were skipped in {root_dir}.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        path, pid, camid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, path


# =========================
# Triplet dataset for Market-1501
# =========================
class Market1501Triplet(Dataset):
    """
    Random triplets (anchor, positive, negative) sampled from a Market-1501 split
    that contains labeled IDs (pid != -1). Assumes pre-cropped people images.
    """
    def __init__(
        self,
        split: Market1501Split,
        crop_size=(128, 64),
        to_gray: bool = False,
        aug: bool = True,
        epoch_len: int = 20000,
    ):
        self.epoch_len = int(epoch_len)
        self.split = split

        # Group paths by pid; keep only pids with >= 2 images (for A/P)
        pid2paths: Dict[int, List[str]] = {}
        for (p, pid, _) in [(s[0], s[1], s[2]) for s in split.samples]:
            if pid == -1:
                continue
            pid2paths.setdefault(pid, []).append(p)
        # prune singletons
        self.pid2paths = {pid: lst for pid, lst in pid2paths.items() if len(lst) >= 2}
        self.pids = sorted(self.pid2paths.keys())

        if len(self.pids) < 2:
            raise RuntimeError("Not enough identities (>=2) with at least two images each for triplet sampling.")

        H, W = crop_size
        t_list = []
        if to_gray:
            t_list.append(transforms.Grayscale(num_output_channels=3))
        t_list.extend([
            transforms.Resize((H, W)),
        ])
        if aug:
            t_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        t_list.extend([
            transforms.ToTensor(),
            # small normalization helps training; not ImageNet stats since model is scratch
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25]),
        ])
        self.tf = transforms.Compose(t_list)

    def __len__(self): return self.epoch_len

    def _sample_two(self, paths: List[str]) -> Tuple[str, str]:
        a, p = random.sample(paths, 2)
        return a, p

    def __getitem__(self, _):
        # choose positive pid
        pid_pos = random.choice(self.pids)
        pos_paths = self.pid2paths[pid_pos]
        a_path, p_path = self._sample_two(pos_paths)

        # choose negative pid (different)
        pid_neg = random.choice(self.pids)
        while pid_neg == pid_pos:
            pid_neg = random.choice(self.pids)
        n_path = random.choice(self.pid2paths[pid_neg])

        a = self.tf(Image.open(a_path).convert("RGB"))
        p = self.tf(Image.open(p_path).convert("RGB"))
        n = self.tf(Image.open(n_path).convert("RGB"))

        # return also labels and paths (optional)
        return a, p, n, (pid_pos, pid_neg, a_path, p_path, n_path)


# =========================
# Embedding model (yours)
# =========================
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
# Triplet loss (yours)
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
# Validation builder (small fixed subset)
# =========================
def build_val_index(
    split: Market1501Split,
    num_ids=12,
    max_per_id=10,
    crop_size=(16,16),
    to_gray=False
):
    # group by pid
    pid2paths: Dict[int, List[str]] = {}
    for (p, pid, _) in [(s[0], s[1], s[2]) for s in split.samples]:
        if pid == -1:
            continue
        pid2paths.setdefault(pid, []).append(p)

    # candidates with at least 2
    cands = [(pid, lst) for pid, lst in pid2paths.items() if len(lst) >= 2]
    if len(cands) == 0:
        raise RuntimeError("Validation set empty (no PIDs with >=2 images).")

    random.shuffle(cands)
    chosen = cands[:num_ids]

    H, W = crop_size
    t_list = []
    if to_gray: t_list.append(transforms.Grayscale(num_output_channels=3))
    t_list += [transforms.Resize((H,W)), transforms.ToTensor(),
               transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])]
    tf = transforms.Compose(t_list)

    images, labels, tags = [], [], []
    for (pid, paths) in chosen:
        random.shuffle(paths)
        for p in paths[:max_per_id]:
            img = Image.open(p).convert("RGB")
            images.append(tf(img))
            labels.append(pid)
            tags.append(("market1501", pid))
    data = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.int64)
    return data, labels, tags


# =========================
# Metrics & Viz (yours)
# =========================
def eval_verification(emb, labels, num_thresholds=200):
    emb = F.normalize(emb, p=2, dim=1)
    sim = emb @ emb.t()
    same = (labels.view(-1,1) == labels.view(1,-1))
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
    plt.figure(); plt.plot(losses, lw=2)
    plt.xlabel("Training step"); plt.ylabel("Triplet loss"); plt.title("Training Loss")
    save_fig(path)

def plot_accuracy_curve(ths, accs, best, path):
    plt.figure()
    plt.plot(ths, accs, lw=2)
    plt.scatter([best[0]],[best[1]], s=40, label=f"best t={best[0]:.2f}, acc={best[1]*100:.1f}%")
    plt.xlabel("Cosine similarity threshold"); plt.ylabel("Verification accuracy")
    plt.title("Val Accuracy vs Threshold"); plt.legend()
    save_fig(path)

def tsne_plot(emb, labels, tags, title, path):
    import numpy as np
    from sklearn.manifold import TSNE

    X = emb.detach().cpu().numpy()
    Y = labels.detach().cpu().numpy()

    n = len(X)
    if n < 3:  # t-SNE needs a few samples
        print(f"[TSNE] Too few samples (n={n}); skipping plot: {path}")
        return

    # Perplexity must be < n_samples; keep it safe and >=5
    perp = max(5, min(30, (n - 1) // 3 if n > 3 else 5))

    # --- Try a broad-compat signature, then fall back to the leanest one
    try:
        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate=200,   # numeric works across versions
            perplexity=perp,
            n_iter=1000,         # may not exist in your sklearn
            verbose=0,
        )
    except TypeError:
        try:
            # Older/variant builds often accept this
            tsne = TSNE(
                n_components=2,
                init="pca",
                learning_rate=200,
                perplexity=perp,
                verbose=0,
            )
        except TypeError:
            # Absolute minimal fallback
            tsne = TSNE(n_components=2)

    Z = tsne.fit_transform(X)

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


def concat_images_with_labels(imgs: List[Image.Image], labels: List[int]):
    w, h = imgs[0].size
    try: font = ImageFont.load_default()
    except: font = None
    tiles = []
    for im, lab in zip(imgs, labels):
        canvas = Image.new("RGB", (w, h+14), (255,255,255))
        canvas.paste(im, (0,0))
        dr = ImageDraw.Draw(canvas)
        dr.text((4,h+2), f"ID {lab}", fill=(0,0,0), font=font)
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

def retrieval_grid(images, emb, labels, out_path, k=5, max_queries=6):
    emb = F.normalize(emb, p=2, dim=1)
    sims = emb @ emb.t()
    N = emb.size(0)
    to_pil = transforms.ToPILImage()
    idxs = list(range(N)); random.shuffle(idxs); queries = idxs[:max_queries]
    rows = []
    for qi in queries:
        order = torch.argsort(sims[qi], descending=True).tolist()
        order = [j for j in order if j != qi][:k]
        row_imgs = [to_pil(images[qi].cpu())] + [to_pil(images[j].cpu()) for j in order]
        row_labels = [int(labels[qi])] + [int(labels[j]) for j in order]
        rows.append(concat_images_with_labels(row_imgs, row_labels))
    grid = vstack_images(rows); ensure_dir(Path(out_path).parent); grid.save(out_path)


# =========================
# Train + Visualize
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market_root", default="Market-1501-v15.09.15", help="folder containing splits")
    ap.add_argument("--train_split", default="bounding_box_train", choices=["bounding_box_train","gt_bbox"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out_dir", default="runs/reid_market_siamese")
    ap.add_argument("--seed", type=int, default=42)

    # data
    ap.add_argument("--crop_h", type=int, default=128)
    ap.add_argument("--crop_w", type=int, default=64)
    ap.add_argument("--to_gray", action="store_true", help="convert to 3ch grayscale before aug/resize")
    ap.add_argument("--aug", action="store_true", help="enable simple color/flip augments")
    ap.add_argument("--skip_pid_minus1", action="store_true", help="skip pid=-1 in training split")
    ap.add_argument("--val_ids", type=int, default=12)
    ap.add_argument("--val_max_per_id", type=int, default=8)

    # train
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--epoch_len", type=int, default=20000)
    ap.add_argument("--log_every", type=int, default=50)

    args = ap.parse_args()
    ensure_dir(args.out_dir); set_seed(args.seed)
    device = torch.device(args.device)

    # ---- train split
    train_dir = os.path.join(args.market_root, args.train_split)
    train_split = Market1501Split(
        root_dir=train_dir,
        skip_neg_one=args.skip_pid_minus1 or True,   # default True
        transform=None
    )
    triplets = Market1501Triplet(
        train_split,
        crop_size=(args.crop_h, args.crop_w),
        to_gray=args.to_gray,
        aug=args.aug or True,
        epoch_len=args.epoch_len
    )
    loader = DataLoader(triplets, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=(args.device=='cuda'),
                        drop_last=True)

    # ---- small fixed validation set from the same split (for quick curves)
    val_images, val_labels, val_tags = build_val_index(
        train_split,
        num_ids=args.val_ids,
        max_per_id=args.val_max_per_id,
        crop_size=(args.crop_h, args.crop_w),
        to_gray=args.to_gray
    )
    val_images = val_images.to(device)

    # ---- model/loss/optim
    model = EmbeddingNet(emb_dim=args.emb_dim, in_ch=3).to(device)
    criterion = TripletLoss(margin=args.margin)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ---- initial eval
    model.eval()
    with torch.inference_mode():
        emb0 = model(val_images)
    ths0, accs0, best0 = eval_verification(emb0, val_labels.to(device))
    plot_accuracy_curve(ths0, accs0, best0, os.path.join(args.out_dir, "val_acc_curve_epoch0.png"))
    tsne_plot(emb0, val_labels.to(device), val_tags, "t-SNE (epoch 0)", os.path.join(args.out_dir, "tsne_epoch0.png"))
    retrieval_grid(val_images, emb0, val_labels.to(device), os.path.join(args.out_dir, "retrieval_epoch0.jpg"))

    # ---- train
    losses = []; step = 0
    for epoch in range(1, args.epochs+1):
        model.train(); t0 = time.time(); run = 0.0
        for a, p, n, _ in loader:
            a=a.to(device, non_blocking=True); p=p.to(device, non_blocking=True); n=n.to(device, non_blocking=True)
            za = model(a); zp = model(p); zn = model(n)
            loss = criterion(za, zp, zn)
            optim.zero_grad(); loss.backward(); optim.step()
            run += float(loss); losses.append(float(loss)); step += 1
            if step % args.log_every == 0:
                print(f"[e{epoch}/{args.epochs} s{step}] loss={run/args.log_every:.4f}", flush=True)
                run = 0.0
        print(f"[epoch {epoch}] time={time.time()-t0:.1f}s")

        # quick val
        model.eval()
        with torch.inference_mode():
            emb = model(val_images)
        ths, accs, best = eval_verification(emb, val_labels.to(device))
        plot_accuracy_curve(ths, accs, best, os.path.join(args.out_dir, f"val_acc_curve_epoch{epoch}.png"))
        tsne_plot(emb, val_labels.to(device), val_tags, f"t-SNE (epoch {epoch})", os.path.join(args.out_dir, f"tsne_epoch{epoch}.png"))
        retrieval_grid(val_images, emb, val_labels.to(device), os.path.join(args.out_dir, f"retrieval_epoch{epoch}.jpg"))

    # ---- save
    torch.save(model.state_dict(), os.path.join(args.out_dir, "siamese_triplet_market1501.pt"))
    plot_loss_curve(losses, os.path.join(args.out_dir, "train_loss.png"))
    print(f"[DONE] outputs in {args.out_dir}")

if __name__ == "__main__":
    main()
