"""Training-time augmentation: transplant real detached electron / photon hits into events.

A small fraction of training events with a triggering positron receive one donor cluster
from purity.preprocessing.brem_donors: real delta-ray, Compton or Bhabha hits that sat
detached from their own positron's track. The cluster is taken from a donor whose
positron had a similar polar angle, rotated about z to the host positron's azimuth and
placed at the host positron's true start, with its time offset kept relative to the host
positron's median hit time. Each transplanted hit is then measured the way the converter
measures real hits: its z selects the plane and hence the view, its transverse coordinate
snaps to that view's strip centre, and a hit landing on a strip that already holds a hit
within 2 ns merges into it (energy summed, PDG bits combined). Hits outside the ATAR are
dropped. The transplanted hits carry the labels the simulation gives such hits: trigger
origin (0), the donor's electron / photon bits, and the host positron's slice.

Transplanted clusters are real topologies rather than uniform random hits, so the model is
not taught that an isolated hit is noise, which would suppress real one- or two-hit
features such as short muon stubs or the start of a 90-degree positron.
"""
import numpy as np

from purity.preprocessing.atar_strip_positions import DEFAULT_LUT, HALF_PITCH, HALF_PLANE, TOL

MERGE_NS = 2.0
POSITION_COLS = ("atar_x", "atar_y", "atar_z", "atar_t", "atar_E", "atar_view", "atar_pdg",
                 "atar_origin", "atar_slice", "atar_slice_mean_t",
                 "atar_truth_x", "atar_truth_y", "atar_truth_z", "atar_truth_t")


class BremTransplant:
    """Callable (row, key) -> row; the row is returned unchanged unless it is selected.

    prob        fraction of events with a triggering positron that receive a cluster
    theta_window_deg  donors are drawn among those whose positron polar angle is within
                this of the host's (the nearest donor if none)
    seed        selection and donor choice depend only on (seed, key), so an event drawn
                twice in an epoch receives the same cluster
    max_hits    events that would exceed this many hits (ATAR + LYSO) are left unchanged
    """

    def __init__(self, donors_path, prob=0.025, theta_window_deg=5.0, seed=0, max_hits=250,
                 lut_path=DEFAULT_LUT):
        d = np.load(donors_path)
        self.offsets = d["offsets"]
        order = np.argsort(d["theta"])
        self.theta_sorted = np.degrees(d["theta"][order])
        self.cluster_by_theta = order
        self.hits = {k: d[k] for k in ("dx", "dy", "dz", "dt", "E", "pdg")}
        lut = np.load(lut_path)
        view, centre, z = lut["view"], lut["centre"].astype(float), lut["z"].astype(float)
        self.centres = {v: np.unique(centre[view == v]) for v in (0, 1)}
        planes = [(pz, v) for v in (0, 1) for pz in np.unique(z[view == v])]
        planes.sort()
        self.plane_z = np.array([p[0] for p in planes])
        self.plane_view = np.array([p[1] for p in planes], np.int64)
        self.prob, self.theta_window, self.seed, self.max_hits = prob, theta_window_deg, seed, max_hits

    def _pick(self, rng, theta_deg):
        lo = np.searchsorted(self.theta_sorted, theta_deg - self.theta_window)
        hi = np.searchsorted(self.theta_sorted, theta_deg + self.theta_window)
        if hi > lo:
            j = rng.integers(lo, hi)
        else:
            j = int(np.clip(np.searchsorted(self.theta_sorted, theta_deg), 0, len(self.theta_sorted) - 1))
        c = self.cluster_by_theta[j]
        return slice(self.offsets[c], self.offsets[c + 1])

    def _measure(self, px, py, pz):
        """(view, measured x, y, z) for a 3D point, or None if it lies outside the ATAR."""
        k = int(np.argmin(np.abs(self.plane_z - pz)))
        if abs(self.plane_z[k] - pz) > HALF_PLANE + TOL:
            return None
        v = int(self.plane_view[k])
        own, other = (px, py) if v == 0 else (py, px)
        cs, co = self.centres[v], self.centres[1 - v]
        i = int(np.argmin(np.abs(cs - own)))
        if abs(cs[i] - own) > HALF_PITCH + TOL or not (co[0] - HALF_PITCH <= other <= co[-1] + HALF_PITCH):
            return None
        return (v, cs[i], 0.0, self.plane_z[k]) if v == 0 else (v, 0.0, cs[i], self.plane_z[k])

    def __call__(self, row, key):
        if "atar_x" not in row or row["atar_x"] is None:
            return row
        origin = np.asarray(row["atar_origin"], np.int64)
        pdg = np.asarray(row["atar_pdg"], np.int64)
        pos = (origin == 0) & ((pdg & 4) > 0)
        if not pos.any():
            return row
        rng = np.random.default_rng([self.seed, int(key)])
        if rng.random() >= self.prob:
            return row
        theta, phi = row.get("truth_theta", np.nan), row.get("truth_phi", np.nan)
        ps = np.array([row.get(f"truth_positron_start_{c}", np.nan) for c in "xyz"], float)
        if not (np.isfinite(theta) and np.isfinite(phi) and np.all(np.isfinite(ps))):
            return row
        n_lyso = len(row["lyso_x"]) if row.get("lyso_x") is not None else 0
        s = self._pick(rng, np.degrees(theta))
        if len(origin) + n_lyso + (s.stop - s.start) > self.max_hits:
            return row

        cols = {c: np.array(row[c]) for c in POSITION_COLS if c in row and row[c] is not None}
        t_pos = float(np.median(cols["atar_t"][pos]))
        tt_pos = float(np.median(cols["atar_truth_t"][pos])) if "atar_truth_t" in cols else None
        pos_slices = cols["atar_slice"][pos]
        vals, cnt = np.unique(pos_slices, return_counts=True)
        host_slice = vals[np.argmax(cnt)]
        slice_t = cols["atar_slice_mean_t"][np.flatnonzero(cols["atar_slice"] == host_slice)[0]]

        c, sn = np.cos(phi), np.sin(phi)
        h = self.hits
        added = 0
        for i in range(s.start, s.stop):
            px = ps[0] + c * h["dx"][i] - sn * h["dy"][i]
            py = ps[1] + sn * h["dx"][i] + c * h["dy"][i]
            pz = ps[2] + h["dz"][i]
            m = self._measure(px, py, pz)
            if m is None:
                continue
            v, mx, my, mz = m
            t_new = t_pos + h["dt"][i]
            same = ((cols["atar_view"] == v) & np.isclose(cols["atar_z"], mz)
                    & np.isclose(cols["atar_x"] if v == 0 else cols["atar_y"], mx if v == 0 else my)
                    & (np.abs(cols["atar_t"] - t_new) < MERGE_NS))
            if same.any():
                j = int(np.flatnonzero(same)[0])
                cols["atar_E"][j] += h["E"][i]
                cols["atar_pdg"][j] |= int(h["pdg"][i])
                continue
            new = {"atar_x": mx, "atar_y": my, "atar_z": mz, "atar_t": t_new, "atar_E": h["E"][i],
                   "atar_view": v, "atar_pdg": int(h["pdg"][i]), "atar_origin": 0,
                   "atar_slice": host_slice, "atar_slice_mean_t": slice_t,
                   "atar_truth_x": px, "atar_truth_y": py, "atar_truth_z": pz,
                   "atar_truth_t": (tt_pos + h["dt"][i]) if tt_pos is not None else t_new}
            for k in cols:
                cols[k] = np.append(cols[k], np.asarray(new[k], dtype=cols[k].dtype))
            added += 1
        row = row.copy()
        for k, v in cols.items():
            row[k] = v
        row["augmented_hits"] = added
        return row
