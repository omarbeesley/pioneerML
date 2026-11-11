"""PURITYModel's positron-direction head as a stand-alone module, and the cached inputs it reads.

DirHead has the same two modules, with the same names and shapes, as the model (models/main.py:
dir_attn_pool, positron_dir_head), so a head's state dict loads into either. Cache holds the parts
written by dump_inputs.py: per tagged hit the 450 network features, per event the predicted
pion stop, the truth direction and the quantities of the angle loss's match gate.
"""
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentionalAggregation

JK, HID = 450, 150
PREFIX = ("dir_attn_pool.", "positron_dir_head.")


class DirHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dir_attn_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(JK, HID), nn.GELU(), nn.Linear(HID, 1)))
        self.positron_dir_head = nn.Sequential(
            nn.Linear(2 * JK + 3, 2 * HID), nn.GELU(), nn.Linear(2 * HID, HID), nn.GELU(), nn.Linear(HID, 3))

    def forward(self, h, ev, view, pion):
        """h [n, 450] hit features, ev [n] event index in the batch, view [n] 0 = x / 1 = y strips,
        pion [B, 3] predicted pion stop. Returns [B, 3] unit directions, as the model computes them."""
        B = pion.size(0)
        pools = []
        for v in (0, 1):
            m = view == v
            pools.append(self.dir_attn_pool(h[m], ev[m], dim_size=B) if bool(m.any()) else h.new_zeros(B, JK))
        return F.normalize(self.positron_dir_head(torch.cat(pools + [pion], dim=-1)), p=2, dim=-1, eps=1e-6)


def head_from_checkpoint(path, device="cpu"):
    """The direction head of a PURITYModel checkpoint, or of a file saved by train.py."""
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("model", sd.get("head", sd))
    head = DirHead()
    head.load_state_dict({k: v for k, v in sd.items() if k.startswith(PREFIX)}, strict=True)
    return head.to(device)


def parts(root):
    """Completed part directories under root (root itself if it is one)."""
    if os.path.exists(f"{root}/meta.json"):
        return [root]
    return sorted(d.rstrip("/") for d in glob.glob(f"{root}/*/") if os.path.exists(f"{d}meta.json"))


class Cache:
    """Dumped parts concatenated on `device`, keeping the hits whose trigger probability exceeds
    trig_thr. Hits stay float16 and are widened per batch.

    gate: the angle loss's match gate (train_utils.py): the event has a trigger positron and more
    than half of its true positron hits are tagged at trigger and MIP probability > 0.5.
    """

    def __init__(self, root, trig_thr, device, log=print):
        ps = parts(root)
        assert ps, f"no completed parts under {root}"
        metas = [json.load(open(f"{p}/meta.json")) for p in ps]
        assert all(m["trig_thr"] <= trig_thr for m in metas), "cache was dumped at a higher trigger threshold"
        keep = [np.load(f"{p}/hit_trig.npy") > trig_thr for p in ps]
        n_ev = [m["n_events"] for m in metas]
        nh, ne = int(sum(k.sum() for k in keep)), int(sum(n_ev))
        self.device, self.n_events, self.n_hits, self.part_dirs, self.part_events = device, ne, nh, ps, n_ev
        self.H = torch.empty((nh, JK), dtype=torch.float16, device=device)
        self.view = torch.empty(nh, dtype=torch.uint8, device=device)
        counts = torch.zeros(ne, dtype=torch.long)
        inter = torch.zeros(ne)
        ev = {k: [] for k in ("pion", "tar", "dir", "htp", "ntrue", "theta", "phtp")}
        h0 = e0 = 0
        for p, k, n in zip(ps, keep, n_ev):
            idx = np.flatnonzero(k)
            H = np.load(f"{p}/H.npy", mmap_mode="r")
            for a in range(0, len(idx), 2_000_000):           # bounded host memory
                sl = idx[a:a + 2_000_000]
                self.H[h0 + a:h0 + a + len(sl)] = torch.from_numpy(np.ascontiguousarray(H[sl])).to(device)
            self.view[h0:h0 + len(idx)] = torch.from_numpy(np.load(f"{p}/hit_view.npy")[idx]).to(device)
            hev = torch.from_numpy(np.load(f"{p}/hit_ev.npy")[idx].astype(np.int64)) + e0
            assert bool((hev[1:] >= hev[:-1]).all()), "hits are not grouped by event"
            counts.index_add_(0, hev, torch.ones(len(hev), dtype=torch.long))
            good = (np.load(f"{p}/hit_trig.npy")[idx] > 0.5) & (np.load(f"{p}/hit_mip.npy")[idx] > 0.5) \
                & (np.load(f"{p}/hit_true.npy")[idx] > 0)
            inter.index_add_(0, hev, torch.from_numpy(good.astype(np.float32)))
            for name in ev:
                ev[name].append(np.load(f"{p}/ev_{name}.npy"))
            h0 += len(idx); e0 += n
            log(f"  loaded {p}: {n} events, {len(idx)} hits")
        self.off = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]).to(device)
        cat = lambda name: torch.from_numpy(np.concatenate(ev[name]))
        self.pion, self.tar, self.own = cat("pion").to(device), cat("tar").to(device), cat("dir").to(device)
        self.htp, self.ntrue = cat("htp").bool(), cat("ntrue").float()
        self.own_theta, self.own_phtp = cat("theta"), cat("phtp")
        self.gate = (self.htp & (inter / self.ntrue.clamp(min=1.0) > 0.5)).to(device)
        log(f"  cache {root}: {ne} events, {nh} hits at trigger > {trig_thr}, "
            f"{int(self.gate.sum())} events pass the match gate, {self.H.element_size() * nh * JK / 2**30:.1f} GiB")

    def gather(self, events):
        """Hits of the given events (LongTensor on device): features float32, batch-local event index, view."""
        s = self.off[events]
        n = self.off[events + 1] - s
        loc = torch.repeat_interleave(torch.arange(len(events), device=self.device), n)
        first = torch.cumsum(n, 0) - n
        idx = torch.arange(int(n.sum()), device=self.device) - first[loc] + s[loc]
        return self.H[idx].float(), loc, self.view[idx]

    @torch.no_grad()
    def predict(self, head, batch=20000):
        """Unit directions of every event, [n_events, 3] on device."""
        out = torch.empty(self.n_events, 3, device=self.device)
        for a in range(0, self.n_events, batch):
            e = torch.arange(a, min(a + batch, self.n_events), device=self.device)
            h, loc, view = self.gather(e)
            out[e] = head(h, loc, view, self.pion[e])
        return out


def angle_deg(a, b):
    """Angle between unit vectors, degrees."""
    return torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1.0, 1.0)))
