from scipy.spatial.distance import cosine

class SimpleTracker:
    def __init__(self, max_lost=30, similarity_thresh=0.5):
        self.next_id = 0
          # id {"emb": embedding, "box": box, "lost": 0}
        self.tracks = {}
        self.max_lost = max_lost
        self.sim_thresh = similarity_thresh

    def _match(self, emb, tracks):
        best_id = None
        best_sim = -1
        for tid, tdata in tracks.items():
             # cosine similarity
            sim = 1 - cosine(emb.flatten(), tdata["emb"].flatten()) 
            if sim > best_sim:
                best_sim, best_id = sim, tid
        return best_id, best_sim

    def update(self, detections):
        """
        detections: list of tuples (emb, box)
        """
        updated_tracks = {}
        # try to match each detection to existing track
        for emb, box in detections:
            match_id, sim = self._match(emb, self.tracks)
            if match_id is not None and sim > self.sim_thresh:
                # found a matching id
                updated_tracks[match_id] = {"emb": emb, "box": box, "lost": 0}
            else:
                # otherwise create new ID
                updated_tracks[self.next_id] = {"emb": emb, "box": box, "lost": 0}
                self.next_id += 1

        # carry over unmatched tracks
        for tid, tdata in self.tracks.items():
            if tid not in updated_tracks:
                tdata["lost"] += 1
                if tdata["lost"] < self.max_lost:
                    updated_tracks[tid] = tdata

        self.tracks = updated_tracks
        return self.tracks