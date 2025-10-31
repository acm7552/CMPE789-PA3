# similarity_tracker.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

def box_center(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def iou(a, b):
    # a,b: [x1,y1,x2,y2]
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1,bx1), max(ay1,by1)
    inter_x2, inter_y2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)

def cosine(u, v):
    # u:(D,), v:(D,)
    nu = np.linalg.norm(u) + 1e-12
    nv = np.linalg.norm(v) + 1e-12
    return float(np.dot(u, v) / (nu*nv))

class SimpleTracker:
    """
    DeepSORT-lite: appearance(EMA) + IoU gating + constant-velocity motion.
    detections: list of (emb_np[1,D], box_np[4])
    """
    def __init__(self,
                 max_lost=30,
                 similarity_thresh=0.3,
                 iou_gate=0.3,
                 ema_momentum=0.9,
                 n_init=3,
                 alpha_app=0.6,
                 # NEW params:
                 recover_window=90,           # keep lost tracks this many frames for reactivation
                 reactivate_app_cos=0.65,     # min cosine to reactivate (appearance)
                 reactivate_iou_gate=0.20,    # looser IoU for reactivation
                 gallery_size=10):            # #embs kept per track for robust matching

        self.max_lost      = max_lost
        self.sim_thresh    = similarity_thresh
        self.iou_gate      = iou_gate
        self.ema_m         = ema_momentum
        self.n_init        = n_init
        self.alpha_app     = alpha_app

        self.next_id = 1
        self.tracks = {}   # id -> dict
        self.recover_window    = recover_window
        self.reactivate_app_cos = reactivate_app_cos
        self.reactivate_iou_gate = reactivate_iou_gate
        self.gallery_size      = gallery_size

        self.tracks = {}            # active/tentative
        self.lost_tracks = {}       # tid -> dict (recoverable pool)

    def _predict(self, trk):
        # constant-velocity center prediction
        x1,y1,x2,y2 = trk["box"]
        cx, cy = (x1+x2)/2., (y1+y2)/2.
        w, h   = (x2-x1), (y2-y1)
        cx_p = cx + trk["vx"]
        cy_p = cy + trk["vy"]
        # size persistence
        return np.array([cx_p - w/2, cy_p - h/2, cx_p + w/2, cy_p + h/2], dtype=np.float32)

    def _update_velocity(self, trk, new_box, beta=0.7):
        ox1, oy1, ox2, oy2 = trk["box"]
        ocx, ocy = (ox1+ox2)/2., (oy1+oy2)/2.
        nx1, ny1, nx2, ny2 = new_box
        ncx, ncy = (nx1+nx2)/2., (ny1+ny2)/2.
        vx_new = ncx - ocx
        vy_new = ncy - ocy
        trk["vx"] = beta*trk["vx"] + (1-beta)*vx_new
        trk["vy"] = beta*trk["vy"] + (1-beta)*vy_new
        trk["updated_this_frame"] = True


    def _update_appearance(self, trk, emb):
        e = emb.reshape(-1)
        trk["emb"] = self.ema_m*trk["emb"] + (1-self.ema_m)*e
        if "gallery" in trk:
            trk["gallery"].append(e.copy())


    def _new_track(self, emb, box):
        e = emb.reshape(-1)
        tid = self.next_id; self.next_id += 1
        self.tracks[tid] = dict(
            box=box.astype(np.float32),
            emb=e.copy(),                  # EMA head
            vx=0.0, vy=0.0,
            age=1, hits=1, lost=0,
            confirmed=False,
            gallery=deque([e.copy()], maxlen=self.gallery_size)
        )

    def _predict_k(self, trk, k):
        # predict k frames ahead with constant velocity on center
        x1,y1,x2,y2 = trk["box"]
        cx, cy = (x1+x2)/2., (y1+y2)/2.
        w,  h  = (x2-x1), (y2-y1)
        cx_p = cx + k * trk.get("vx", 0.0)
        cy_p = cy + k * trk.get("vy", 0.0)
        return np.array([cx_p - w/2, cy_p - h/2, cx_p + w/2, cy_p + h/2], dtype=np.float32)

    def _cosine_best(self, trk_emb_gallery, det_emb):
        det = det_emb.reshape(-1)
        best = -1.0
        for g in trk_emb_gallery:
            # cosine
            nu = np.linalg.norm(g) + 1e-12
            nv = np.linalg.norm(det) + 1e-12
            best = max(best, float(np.dot(g, det) / (nu*nv)))
        return best


    def _build_cost(self, preds, dets):
        # cost = alpha*(1-cos) + (1-alpha)*(1-IoU), gated by IoU
        T, D = len(preds), len(dets)
        cost = np.full((T, D), 1e3, dtype=np.float32)
        for ti,(tid, trk) in enumerate(preds):
            pbox = trk["pred_box"]
            pemb = trk["emb"]
            for dj,(demb, dbox) in enumerate(dets):
                i = iou(pbox, dbox)
                if i < self.iou_gate:        # gate
                    continue
                c = 1.0 - cosine(pemb, demb.reshape(-1))
                cost[ti, dj] = self.alpha_app*c + (1-self.alpha_app)*(1.0 - i)
        return cost

    def update(self, detections):
        # detections: list of (emb, box)
        # 1) predict
        for tid, trk in self.tracks.items():
            trk["pred_box"] = self._predict(trk)

        # 2) match by cost (Hungarian)
        preds = list(self.tracks.items())
        cost = self._build_cost(preds, detections) if preds and detections else None
        matched, u_tracks, u_dets = [], list(range(len(preds))), list(range(len(detections)))

        

        if cost is not None:
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < (self.sim_thresh + 0.6):  # loose cap; we already gated by IoU
                    matched.append((r, c))
            matched_t = {r for (r,_) in matched}
            matched_d = {c for (_,c) in matched}
            u_tracks = [i for i in range(len(preds)) if i not in matched_t]
            u_dets   = [j for j in range(len(detections)) if j not in matched_d]

        # 3) update matched
        for r, c in matched:
            tid, trk = preds[r]
            demb, dbox = detections[c]
            self._update_velocity(trk, dbox)
            trk["box"] = dbox.astype(np.float32)
            self._update_appearance(trk, demb)
            trk["age"]  += 1
            trk["hits"] += 1
            trk["lost"]  = 0
            if not trk["confirmed"] and trk["hits"] >= self.n_init:
                trk["confirmed"] = True

        # 4) try IoU-only fallback for unmatched tracks (no good appearance)
        if u_tracks and u_dets:
            # greedy IoU matching
            used_d = set()
            for ui in list(u_tracks):
                tid, trk = preds[ui]
                best_j, best_iou = -1, 0.0
                for j in u_dets:
                    _, dbox = detections[j]
                    i = iou(trk["pred_box"], dbox)
                    if i > self.iou_gate and i > best_iou:
                        best_iou, best_j = i, j
                if best_j >= 0 and best_j not in used_d:
                    _, dbox = detections[best_j]
                    demb, _ = detections[best_j]
                    self._update_velocity(trk, dbox)
                    trk["box"] = dbox.astype(np.float32)
                    self._update_appearance(trk, demb)
                    trk["age"]  += 1
                    trk["hits"] += 1
                    trk["lost"]  = 0
                    if not trk["confirmed"] and trk["hits"] >= self.n_init:
                        trk["confirmed"] = True
                    used_d.add(best_j)
                    u_tracks.remove(ui)
                    u_dets.remove(best_j)

            # 4.5) Reactivation: try to match remaining detections to lost tracks
        reactivated_dets = set()
        if u_dets and self.lost_tracks:
            # greedy by best combined score
            for j in list(u_dets):
                demb, dbox = detections[j]
                best_tid, best_score = None, 1e9
                for tid, ltrk in self.lost_tracks.items():
                    # predict where it should be after ltrk["recover_age"] frames
                    pbox = self._predict_k(ltrk, ltrk.get("recover_age", 0))
                    i = iou(pbox, dbox)
                    if i < self.reactivate_iou_gate:
                        continue
                    cosb = self._cosine_best(ltrk.get("gallery", [ltrk["emb"]]), demb)
                    if cosb < self.reactivate_app_cos:
                        continue
                    # combined cost (lower is better), same alpha as normal
                    cost = self.alpha_app * (1.0 - cosb) + (1.0 - self.alpha_app) * (1.0 - i)
                    if cost < best_score:
                        best_score, best_tid = cost, tid

                if best_tid is not None:
                    # REACTIVATE: move lost track back to active with SAME ID
                    trk = self.lost_tracks[best_tid]
                    # velocity update vs predicted -> actual
                    self._update_velocity(trk, dbox)
                    trk["box"]   = dbox.astype(np.float32)
                    self._update_appearance(trk, demb)
                    trk["lost"]  = 0
                    trk["age"]   += 1
                    trk["hits"]  += 1
                    trk["confirmed"] = True
                    trk["updated_this_frame"] = True

                    # put back to active and remove from recover pool
                    self.tracks[best_tid] = trk
                    del self.lost_tracks[best_tid]

                    reactivated_dets.add(j)

            # remove reactivated detections from unmatched set
            u_dets = [j for j in u_dets if j not in reactivated_dets]


        # 5) create new tracks for remaining detections
        for j in u_dets:
            demb, dbox = detections[j]
            self._new_track(demb, dbox)

            # 6) mark unmatched tracks as lost / move to recover pool
        to_delete = []
        to_recover = []
        for tid, trk in self.tracks.items():
            if trk.get("updated_this_frame", False):
                trk["updated_this_frame"] = False
                continue
            trk["lost"] += 1
            if trk["lost"] > self.max_lost:
                # move confirmed tracks to recoverable pool
                if trk.get("confirmed", False):
                    trk["recover_age"] = 0
                    self.lost_tracks[tid] = trk
                    to_recover.append(tid)
                else:
                    to_delete.append(tid)

        for tid in to_recover:
            del self.tracks[tid]
        for tid in to_delete:
            del self.tracks[tid]

        # age the recoverable pool and purge old ones
        purge = []
        for tid, ltrk in self.lost_tracks.items():
            ltrk["recover_age"] += 1
            if ltrk["recover_age"] > self.recover_window:
                purge.append(tid)
        for tid in purge:
            del self.lost_tracks[tid]


        # 7) return only confirmed tracks (or include tentative if you want)
        out = {}
        for tid, trk in self.tracks.items():
            # draw only if it’s confirmed AND got matched this frame (lost == 0)
            if trk["confirmed"] and trk["lost"] == 0:
                out[tid] = {"box": trk["box"].copy()}
        return out