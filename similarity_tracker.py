# similarity_tracker.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

# ----------------- geometry / metrics -----------------
def iou(a, b):
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
    nu = np.linalg.norm(u) + 1e-12
    nv = np.linalg.norm(v) + 1e-12
    return float(np.dot(u, v) / (nu*nv))

def box_center(b):
    x1, y1, x2, y2 = b
    return (0.5*(x1+x2), 0.5*(y1+y2))

def box_diag(b):
    x1, y1, x2, y2 = b
    return ((x2-x1)**2 + (y2-y1)**2) ** 0.5

# ----------------- tracker -----------------
class SimpleTracker:
    """
    DeepSORT-lite++:
      • Cascade matching (high-score stage then regular)
      • IoU + cosine + center-distance + direction penalty
      • Strong-match locking, anti-swap post-check
      • Constant-velocity prediction (EMA)
      • Reactivation from recoverable pool
      • Birth cooldown near active/lost predictions
      • Return only confirmed & matched tracks (no ghosts)
    detections: list of (emb_np[1,D], box_np[4])
    """
    def __init__(self,
                 max_lost=30,
                 similarity_thresh=0.3,
                 iou_gate=0.4,
                 ema_momentum=0.9,
                 n_init=3,
                 alpha_app=0.6,
                 # reactivation / gallery
                 recover_window=90,
                 reactivate_app_cos=0.65,
                 reactivate_iou_gate=0.20,
                 gallery_size=10,
                 # cascade / gates
                 high_score_th=0.70,
                 dist_gate_scale=0.75,      # max center jump ≈ 0.75 * box diagonal
                 dir_penalty=0.10):         # extra cost if going opposite predicted motion
        self.max_lost      = max_lost
        self.sim_thresh    = similarity_thresh
        self.iou_gate      = iou_gate
        self.ema_m         = ema_momentum
        self.n_init        = n_init
        self.alpha_app     = alpha_app

        self.recover_window     = recover_window
        self.reactivate_app_cos = reactivate_app_cos
        self.reactivate_iou_gate = reactivate_iou_gate
        self.gallery_size       = gallery_size

        self.high_score_th = high_score_th
        self.dist_gate_scale = dist_gate_scale
        self.dir_penalty = dir_penalty

        self.next_id     = 1
        self.tracks      = {}   # active/tentative
        self.lost_tracks = {}   # recoverable pool

    # ---------- kinematics / appearance ----------
    def _predict(self, trk):
        x1, y1, x2, y2 = trk["box"]
        cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
        w, h   = (x2-x1), (y2-y1)
        cx_p = cx + trk.get("vx", 0.0)
        cy_p = cy + trk.get("vy", 0.0)
        return np.array([cx_p - w/2, cy_p - h/2, cx_p + w/2, cy_p + h/2], dtype=np.float32)

    def _predict_k(self, trk, k):
        x1, y1, x2, y2 = trk["box"]
        cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
        w, h   = (x2-x1), (y2-y1)
        cx_p = cx + k * trk.get("vx", 0.0)
        cy_p = cy + k * trk.get("vy", 0.0)
        return np.array([cx_p - w/2, cy_p - h/2, cx_p + w/2, cy_p + h/2], dtype=np.float32)

    def _update_velocity(self, trk, new_box, beta=0.7):
        ox1, oy1, ox2, oy2 = trk["box"]
        ocx, ocy = 0.5*(ox1+ox2), 0.5*(oy1+oy2)
        nx1, ny1, nx2, ny2 = new_box
        ncx, ncy = 0.5*(nx1+nx2), 0.5*(ny1+ny2)
        vx_new = ncx - ocx
        vy_new = ncy - ocy
        trk["vx"] = beta*trk.get("vx", 0.0) + (1-beta)*vx_new
        trk["vy"] = beta*trk.get("vy", 0.0) + (1-beta)*vy_new

    def _update_appearance(self, trk, emb):
        e = emb.reshape(-1)
        trk["emb"] = self.ema_m*trk["emb"] + (1-self.ema_m)*e
        if "gallery" in trk:
            trk["gallery"].append(e.copy())

    def _cosine_best(self, trk_emb_gallery, det_emb):
        det = det_emb.reshape(-1)
        best = -1.0
        for g in trk_emb_gallery:
            nu = np.linalg.norm(g) + 1e-12
            nv = np.linalg.norm(det) + 1e-12
            best = max(best, float(np.dot(g, det) / (nu*nv)))
        return best

    def _new_track(self, emb, box):
        e = emb.reshape(-1)
        tid = self.next_id; self.next_id += 1
        self.tracks[tid] = dict(
            box=box.astype(np.float32),
            emb=e.copy(),
            vx=0.0, vy=0.0,
            age=1, hits=1, lost=0,
            confirmed=False,
            gallery=deque([e.copy()], maxlen=self.gallery_size)
        )

    # ---------- cost / gating ----------
    def _center_dist_ok(self, pbox, dbox):
        # limit center jump relative to current size
        gate = self.dist_gate_scale * max(box_diag(pbox), 1.0)
        (pcx, pcy) = box_center(pbox)
        (dcx, dcy) = box_center(dbox)
        return ((pcx-dcx)**2 + (pcy-dcy)**2) ** 0.5 <= gate

    def _direction_penalty(self, trk, dbox):
        # if detection jumps opposite to predicted motion vector, add a small penalty
        vx, vy = trk.get("vx", 0.0), trk.get("vy", 0.0)
        if abs(vx)+abs(vy) < 1e-6:
            return 0.0
        (pcx, pcy) = box_center(trk["pred_box"])
        (dcx, dcy) = box_center(dbox)
        mx, my = vx, vy
        jx, jy = (dcx - pcx), (dcy - pcy)
        dot = mx*jx + my*jy
        return self.dir_penalty if dot < 0 else 0.0

    def _pair_cost(self, trk, demb, dbox):
        pbox, pemb = trk["pred_box"], trk["emb"]
        i = iou(pbox, dbox)
        if i < self.iou_gate or not self._center_dist_ok(pbox, dbox):
            return 1e3
        cosv = cosine(pemb, demb.reshape(-1))
        base = self.alpha_app*(1.0 - cosv) + (1.0 - self.alpha_app)*(1.0 - i)
        # slight bias favoring confirmed/older tracks to keep identities
        bias = 0.02 if trk.get("confirmed", False) else 0.0
        # direction consistency penalty
        base += self._direction_penalty(trk, dbox)
        return base + bias

    # ---------- birth suppression near active/lost ----------
    def _near_recent_pred(self, dbox, iou_thr_active=0.5, iou_thr_lost=0.3):
        for _, trk in self.tracks.items():
            if iou(trk.get("pred_box", trk["box"]), dbox) >= iou_thr_active:
                return True
        for _, ltrk in self.lost_tracks.items():
            pbox = self._predict_k(ltrk, ltrk.get("recover_age", 0))
            if iou(pbox, dbox) >= iou_thr_lost:
                return True
        return False

    # -------------------------------------------------------
    def update(self, detections_with_scores):
        """
        detections_with_scores: list of (emb, box, score) or (emb, box) if caller omits score.
        """
        # standardize tuple shape
        dets = []
        for d in detections_with_scores:
            if len(d) == 2:
                emb, box = d; score = 1.0
            else:
                emb, box, score = d
            dets.append((emb, box, float(score)))

        # 1) predict
        for tid, trk in self.tracks.items():
            trk["pred_box"] = self._predict(trk)

        preds = list(self.tracks.items())
        matched = []
        used_rows, used_cols = set(), set()

        # ---------- Stage 0: lock very strong matches ----------
        locked_pairs = []
        if preds and dets:
            for ti, (tid, trk) in enumerate(preds):
                best_j, best_score = -1, 1e9
                for dj, (demb, dbox, dsc) in enumerate(dets):
                    if dj in used_cols:
                        continue
                    # strong gates
                    if dsc < max(self.high_score_th, 0.75):
                        continue
                    i = iou(trk["pred_box"], dbox)
                    if i < max(0.5, self.iou_gate):
                        continue
                    if not self._center_dist_ok(trk["pred_box"], dbox):
                        continue
                    cosv = cosine(trk["emb"], demb.reshape(-1))
                    if cosv < 0.80:
                        continue
                    c = self._pair_cost(trk, demb, dbox)
                    if c < best_score:
                        best_score, best_j = c, dj
                if best_j >= 0:
                    locked_pairs.append((ti, best_j))
                    used_rows.add(ti)
                    used_cols.add(best_j)

        # helper to run a Hungarian pass over remaining rows/cols with a custom list of detection indices
        def hungarian_pass(remaining_rows, remaining_cols):
            local_matches = []
            if not remaining_rows or not remaining_cols:
                return local_matches
            sub_cost = np.full((len(remaining_rows), len(remaining_cols)), 1e3, dtype=np.float32)
            for rr, ti in enumerate(remaining_rows):
                tid, trk = preds[ti]
                for cc, dj in enumerate(remaining_cols):
                    demb, dbox, _ = dets[dj]
                    sub_cost[rr, cc] = self._pair_cost(trk, demb, dbox)
            rows, cols = linear_sum_assignment(sub_cost)
            for r, c in zip(rows, cols):
                if sub_cost[r, c] < (self.sim_thresh + 0.6):
                    local_matches.append((remaining_rows[r], remaining_cols[c]))
            return local_matches

        # ---------- Stage 1: cascade on high-score detections ----------
        rem_rows = [i for i in range(len(preds)) if i not in used_rows]
        hi_cols  = [j for j,(e,b,s) in enumerate(dets) if j not in used_cols and s >= self.high_score_th]
        matched += locked_pairs
        matched += hungarian_pass(rem_rows, hi_cols)

        # ---------- Stage 2: match the rest ----------
        used_rows = {r for (r,_) in matched}
        used_cols = {c for (_,c) in matched}
        rem_rows = [i for i in range(len(preds)) if i not in used_rows]
        lo_cols  = [j for j in range(len(dets)) if j not in used_cols]
        matched += hungarian_pass(rem_rows, lo_cols)

        # ----- POST-MATCH: anti-swap check -----
        def dist(a, b):
            ax, ay = box_center(a); bx, by = box_center(b)
            return ((ax-bx)**2 + (ay-by)**2) ** 0.5

        swaps_fixed = set()
        match_by_row = {r: c for (r, c) in matched}
        for r1, c1 in list(matched):
            if (r1, c1) in swaps_fixed:
                continue
            tid1, trk1 = preds[r1]
            _, db1, _ = dets[c1]
            for r2, c2 in list(matched):
                if r2 <= r1 or (r2, c2) in swaps_fixed:
                    continue
                tid2, trk2 = preds[r2]
                _, db2, _ = dets[c2]
                p1, p2 = trk1["pred_box"], trk2["pred_box"]
                cross_geom = dist(p1, db2) + dist(p2, db1) + 1e-6 < dist(p1, db1) + dist(p2, db2)
                if not cross_geom:
                    continue
                # appearance preference
                def cos(trk_emb, det_emb):
                    return cosine(trk_emb, det_emb.reshape(-1))
                cos11 = cos(trk1["emb"], dets[c1][0]); cos12 = cos(trk1["emb"], dets[c2][0])
                cos22 = cos(trk2["emb"], dets[c2][0]); cos21 = cos(trk2["emb"], dets[c1][0])
                prefer_diagonal = (cos11 + cos22) >= (cos12 + cos21)
                if prefer_diagonal and match_by_row.get(r1) == c2 and match_by_row.get(r2) == c1:
                    matched.remove((r1, c2)); matched.remove((r2, c1))
                    matched.append((r1, c1)); matched.append((r2, c2))
                    swaps_fixed.add((r1, c1)); swaps_fixed.add((r2, c2))

        # unmatched
        matched_t = {r for (r,_) in matched}
        matched_d = {c for (_,c) in matched}
        u_tracks = [i for i in range(len(preds)) if i not in matched_t]
        u_dets   = [j for j in range(len(dets)) if j not in matched_d]

        # 3) update matched
        for r, c in matched:
            tid, trk = preds[r]
            demb, dbox, _ = dets[c]
            self._update_velocity(trk, dbox)
            trk["box"] = dbox.astype(np.float32)
            self._update_appearance(trk, demb)
            trk["age"]  += 1
            trk["hits"] += 1
            trk["lost"]  = 0
            trk["updated_this_frame"] = True
            if not trk["confirmed"] and trk["hits"] >= self.n_init:
                trk["confirmed"] = True

        # 4) IoU-only fallback for unmatched tracks
        if u_tracks and u_dets:
            used_d = set()
            for ui in list(u_tracks):
                tid, trk = preds[ui]
                best_j, best_iou = -1, 0.0
                for j in u_dets:
                    _, dbox, _ = dets[j]
                    i = iou(trk["pred_box"], dbox)
                    if i > self.iou_gate and i > best_iou and self._center_dist_ok(trk["pred_box"], dbox):
                        best_iou, best_j = i, j
                if best_j >= 0 and best_j not in used_d:
                    demb, dbox, _ = dets[best_j]
                    self._update_velocity(trk, dbox)
                    trk["box"] = dbox.astype(np.float32)
                    self._update_appearance(trk, demb)
                    trk["age"]  += 1
                    trk["hits"] += 1
                    trk["lost"]  = 0
                    trk["updated_this_frame"] = True
                    if not trk["confirmed"] and trk["hits"] >= self.n_init:
                        trk["confirmed"] = True
                    used_d.add(best_j)
                    u_tracks.remove(ui)
                    u_dets.remove(best_j)

        # 4.5) Reactivation from lost pool
        reactivated_dets = set()
        if u_dets and self.lost_tracks:
            for j in list(u_dets):
                demb, dbox, _ = dets[j]
                best_tid, best_score = None, 1e9
                for tid, ltrk in self.lost_tracks.items():
                    pbox = self._predict_k(ltrk, ltrk.get("recover_age", 0))
                    i = iou(pbox, dbox)
                    if i < self.reactivate_iou_gate:
                        continue
                    cosb = self._cosine_best(ltrk.get("gallery", [ltrk["emb"]]), demb)
                    if cosb < self.reactivate_app_cos:
                        continue
                    cost = self.alpha_app*(1.0 - cosb) + (1.0 - self.alpha_app)*(1.0 - i)
                    if cost < best_score:
                        best_score, best_tid = cost, tid
                if best_tid is not None:
                    trk = self.lost_tracks[best_tid]
                    self._update_velocity(trk, dbox)
                    trk["box"]   = dbox.astype(np.float32)
                    trk["lost"]  = 0
                    trk["age"]   += 1
                    trk["hits"]  += 1
                    trk["confirmed"] = True
                    trk["updated_this_frame"] = True
                    self._update_appearance(trk, demb)
                    self.tracks[best_tid] = trk
                    del self.lost_tracks[best_tid]
                    reactivated_dets.add(j)
            u_dets = [j for j in u_dets if j not in reactivated_dets]

        # 5) create new tracks (suppress births near predicted positions)
        for j in u_dets:
            demb, dbox, _ = dets[j]
            if self._near_recent_pred(dbox, iou_thr_active=0.5, iou_thr_lost=0.3):
                continue
            self._new_track(demb, dbox)

        # 6) age / move to recoverable / purge
        to_delete, to_recover = [], []
        for tid, trk in self.tracks.items():
            if trk.get("updated_this_frame", False):
                trk["updated_this_frame"] = False
                continue
            trk["lost"] += 1
            if trk["lost"] > self.max_lost:
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

        purge = []
        for tid, ltrk in self.lost_tracks.items():
            ltrk["recover_age"] += 1
            if ltrk["recover_age"] > self.recover_window:
                purge.append(tid)
        for tid in purge:
            del self.lost_tracks[tid]

        # 7) return ONLY confirmed & matched-this-frame tracks
        out = {}
        for tid, trk in self.tracks.items():
            if trk["confirmed"] and trk["lost"] == 0:
                out[tid] = {"box": trk["box"].copy()}
        return out
