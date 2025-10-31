import os
import torch
import random
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from PIL import Image
import cv2
import numpy as np
import torch.nn.functional as F

# data / models
from MOT16 import MOT16, detection_collate
from train_siamese import EmbeddingNet
from similarity_tracker import SimpleTracker

tf = transforms.ToTensor()
id_colors = {}
random.seed(42)

def load_data():
    testMOT16data = MOT16(root='MOT16/test', transform=tf, getGroundTruth=False, useDetections=True)
    return DataLoader(
        testMOT16data, batch_size=16, shuffle=False,
        pin_memory=True, collate_fn=detection_collate,
        num_workers=2, persistent_workers=True
    )

def load_model(weights_pth, device):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    loadWeights = torch.load(weights_pth, map_location=device, weights_only=True)
    model.load_state_dict(loadWeights)
    model.to(device).eval()
    return model

def load_siamese(weights_path, device, emb_dim=256):
    model = EmbeddingNet(emb_dim=emb_dim, in_ch=3).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def convert_box_for_siamese(crop, device):
    crop = Image.fromarray(crop).convert('RGB')
    tfm = transforms.Compose([transforms.Resize((128, 64)), transforms.ToTensor()])
    return tfm(crop).unsqueeze(0).to(device)

def get_embedding(siamese, crop_tensor):
    with torch.no_grad():
        emb = siamese(crop_tensor)
        emb = F.normalize(emb, dim=1)  # cosine space
    return emb.cpu().numpy()

def load_pretrained(device):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.to(device).eval()
    return model

def get_color_for_id(track_id):
    if track_id not in id_colors:
        id_colors[track_id] = (random.randint(64,255), random.randint(64,255), random.randint(64,255))
    return id_colors[track_id]

# --------- detector with extra NMS / top-K ----------
def get_detections(model, image, device, thresh, save=True, nms_iou=0.6, max_det=40):
    img = Image.open(image).convert("RGB")
    img_tensor = tf(img).to(device)
    with torch.no_grad():
        pred = model([img_tensor])[0]

    boxes  = pred["boxes"].detach().cpu()
    scores = pred["scores"].detach().cpu()
    labels = pred["labels"].detach().cpu()

    keep = (labels == 1) & (scores >= thresh)
    boxes, scores = boxes[keep], scores[keep]

    if boxes.numel() > 0:
        keep_idx = nms(boxes, scores, nms_iou)
        keep_idx = keep_idx[:max_det]
        boxes  = boxes[keep_idx].numpy()
        scores = scores[keep_idx].numpy()
    else:
        boxes, scores = boxes.numpy(), scores.numpy()

    img_cv = np.array(img)[:, :, ::-1].copy() if save else None
    return boxes, scores, img_cv

def _expand_box(box, img_w, img_h, ctx=0.30):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w/2, y1 + h/2
    nw, nh = w * (1 + ctx), h * (1 + ctx)
    ex1, ey1 = int(max(0, cx - nw/2)), int(max(0, cy - nh/2))
    ex2, ey2 = int(min(img_w-1, cx + nw/2)), int(min(img_h-1, cy + nh/2))
    return np.array([ex1, ey1, ex2, ey2], dtype=np.int32)

def play_sequence(model, siamese, sequence_dir, device, thresh, save_path, vidName):
    frames = sorted([os.path.join(sequence_dir, f) for f in os.listdir(sequence_dir) if f.endswith(".jpg")])

    out_vid = None
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        sample = cv2.imread(frames[0])
        h, w = sample.shape[:2]
        video_file = os.path.join(save_path, f"{vidName}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(video_file, fourcc, 30, (w, h))

    tracker = SimpleTracker(
        max_lost=30,
        recover_window=120,         # give more chance to roll back old IDs
        reactivate_app_cos=0.65,
        reactivate_iou_gate=0.20,
        n_init=3,
        iou_gate=0.6,               # stricter geometry to avoid swaps
        alpha_app=0.6,
        ema_momentum=0.9,
        high_score_th=0.75,
        dist_gate_scale=1,
        dir_penalty=0.10
    )

    NEW_TRACK_MIN_SCORE = 0.9
    MIN_WH = 12

    for frame_path in frames:
        boxes, scores, img_cv = get_detections(model, frame_path, device, thresh, save=True)
        detections = []

        h, w = img_cv.shape[:2]
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            if score < NEW_TRACK_MIN_SCORE:       # don’t birth from weak dets
                continue
            if (x2 - x1) < MIN_WH or (y2 - y1) < MIN_WH:
                continue

            ex1, ey1, ex2, ey2 = _expand_box(box, w, h, ctx=0.30)
            crop = img_cv[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue

            crop_tensor = convert_box_for_siamese(crop, device)
            emb = get_embedding(siamese, crop_tensor)
            # pass score through so the tracker can do cascade matching
            detections.append((emb, box, float(score)))

        tracks = tracker.update(detections)

        for tid, tdata in tracks.items():
            x1, y1, x2, y2 = tdata["box"].astype(int)
            color = get_color_for_id(tid)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_cv, f"ID {tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Detections", img_cv)
        if out_vid:
            out_vid.write(img_cv)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    if out_vid:
        out_vid.release()
        print(f"Saved video to {video_file}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to model params")
    ap.add_argument("--sequence", required=True, help="path to frames for tracking, e.g. 'MOT16/test/MOT16-01/'")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testSequence = f"{args.sequence}img1/"
    parent = os.path.basename(os.path.dirname(os.path.dirname(testSequence)))
    model_name = args.model.replace(".pth", "")

    siamese = load_siamese("runs/input_128_64/reid_viz/siamese_triplet_murphy.pt", device)
    video_name = f"{parent}_{model_name}_final"

    if args.model.lower() == 'pretrained':
        pretrained = load_pretrained(device)
        play_sequence(pretrained, siamese, testSequence, device, thresh=0.8, save_path="videos/", vidName=video_name)
    else:
        finetuned = load_model(args.model, device)
        play_sequence(finetuned, siamese, testSequence, device, thresh=0.8, save_path="videos/", vidName=video_name)

if __name__ == "__main__":
    main()
