"""Historical tail-score analysis functions, extracted without model inference.

Only the optional learned logistic fusion fit imports PyTorch, locally on CPU.
"""
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_trapz = getattr(np, "trapezoid", np.trapz)


def roc_auc(scores, labels):
    """ROC curve + AUC, numpy only. labels in {0,1}; higher score = signal."""
    order = np.argsort(-scores, kind="mergesort")
    s = labels[order].astype(np.float64)
    P = max(labels.sum(), 1.0)
    N = max(len(labels) - labels.sum(), 1.0)
    tpr = np.concatenate([[0.0], np.cumsum(s) / P])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - s) / N])
    return fpr, tpr, float(_trapz(tpr, fpr))

def binned_mean(x, y, bins):
    """Mean of y in bins of x. Returns (centers, means)."""
    idx = np.digitize(x, bins) - 1
    centers, means = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        centers.append(0.5 * (bins[b] + bins[b + 1]))
        means.append(float(y[m].mean()))
    return np.array(centers), np.array(means)

def slope(centers, means):
    if len(centers) < 2:
        return float("nan")
    return float(np.polyfit(centers, means, 1)[0])

def fit_logistic(X, y, steps=800, lr=0.05):
    """Tiny logistic-regression stacker (torch). X:[N,F] (standardized), y:[N] {0,1}.
    Returns (weights[F], bias)."""
    import torch
    import torch.nn.functional as F
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xt @ w + b, yt)
        loss.backward(); opt.step()
    return w.detach().numpy(), float(b.detach())

def head_fusion(data, output_dir, target_eff=0.90):
    """Orthogonality of the heads + combined ('multiplied') keep-score.
    target_eff = per-head signal efficiency used to set the leak-through cuts."""
    keys = ["pie", "topo", "muon", "pileup"]
    # +1: higher score => more pie-like; -1: veto (lower => more pie-like)
    pie_dir = {"pie": 1.0, "topo": 1.0, "muon": -1.0, "pileup": -1.0}
    is_pie = data["is_pie"].astype(int)
    N = len(is_pie)
    S = np.stack([data[k] for k in keys], axis=1)              # sigmoid scores
    L = np.stack([data[k + "_logit"] for k in keys], axis=1)   # raw logits
    print("\n=== head fusion / orthogonality ===", flush=True)

    # (1) within-class score correlation (redundancy)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    for ax, (cls, mask) in zip(axes, [("signal (pi->e nu)", is_pie == 1),
                                      ("background (non-signal background)", is_pie == 0)]):
        if mask.sum() < 3:
            continue
        C = np.corrcoef(S[mask].T)
        print(f"\nscore correlation [{cls}]:")
        print("         " + " ".join(f"{k:>7}" for k in keys))
        for i, k in enumerate(keys):
            print(f"  {k:>6} " + " ".join(f"{C[i, j]:+8.4f}" for j in range(len(keys))))
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=45, fontsize=8)
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=8)
        ax.set_title(f"score corr\n{cls}", fontsize=9)
        for i in range(len(keys)):
            for j in range(len(keys)):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{output_dir}/fusion_corr.png", dpi=130); plt.close(fig)

    # (2) background leak-through overlap at matched signal efficiency
    sig, bkg = is_pie == 1, is_pie == 0
    n_sig, n_bkg = int(sig.sum()), int(bkg.sum())
    # Each head's threshold is set to keep target_eff of the PIE (signal) events
    # individually (a relative operating point derived from the pie-score
    # quantiles, NOT a fixed score). The combined (AND of all heads) signal
    # efficiency is therefore lower; it's printed below for context.
    keeps, thr = {}, {}
    for k in keys:
        s = data[k]
        if pie_dir[k] > 0:                       # keep if score > threshold
            thr[k] = float(np.quantile(s[sig], 1 - target_eff)); keeps[k] = s > thr[k]
        else:                                    # veto: keep if score < threshold
            thr[k] = float(np.quantile(s[sig], target_eff));     keeps[k] = s < thr[k]
    print(f"\nbackground leak-through @ {target_eff:.0%} PER-HEAD signal eff "
          f"(N_pie={n_sig}, N_michel={n_bkg}):")
    for k in keys:
        op = ">" if pie_dir[k] > 0 else "<"
        nk, nsk = int(keeps[k][bkg].sum()), int(keeps[k][sig].sum())
        print(f"  {k:>6}: keep if score {op} {thr[k]:.4f}   "
              f"leak={nk / n_bkg:.6f} ({nk}/{n_bkg})   "
              f"sig_keep={nsk / n_sig:.6f} ({nsk}/{n_sig})")
    print("pairwise double-leak (both keep the SAME non-signal background event) vs independence:")
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            la, lb = keeps[a][bkg], keeps[b][bkg]
            n_both = int((la & lb).sum())
            both, indep = n_both / n_bkg, float(la.mean() * lb.mean())
            ratio = both / indep if indep > 0 else float("nan")
            print(f"  {a:>6} & {b:<7} both={both:.6f} ({n_both}/{n_bkg})  "
                  f"indep~{indep:.6f}  ratio={ratio:.4f}  (1=orthogonal, >>1=coincident)")
    all_keep_sig = np.ones(n_sig, bool); all_leak = np.ones(n_bkg, bool)
    for k in keys:
        all_keep_sig &= keeps[k][sig]; all_leak &= keeps[k][bkg]
    n_keep, n_all = int(all_keep_sig.sum()), int(all_leak.sum())
    print(f"  ALL heads combined: sig_eff={n_keep / n_sig:.6f} ({n_keep}/{n_sig})   "
          f"leak={n_all / n_bkg:.6f} ({n_all}/{n_bkg})   <- 'irreducible' non-signal background")

    # (3) combined keep-score vs single heads (held-out split)
    rng = np.random.default_rng(0)
    perm = rng.permutation(N); h = N // 2
    fit_i, ev_i = perm[:h], perm[h:]
    signs = np.array([pie_dir[k] for k in keys])
    single_auc = {}
    print(f"\npie-discrimination AUC on held-out half (N_eval={len(ev_i)}, keep-direction):")
    for k in keys:
        single_auc[k] = roc_auc(pie_dir[k] * L[ev_i, keys.index(k)], is_pie[ev_i])[2]
        print(f"  {k:>9}: {single_auc[k]:.6f}")
    bestk = max(single_auc, key=single_auc.get); best = single_auc[bestk]

    sum_logit = (L * signs).sum(1)                       # "multiply" = add logits
    auc_sum = roc_auc(sum_logit[ev_i], is_pie[ev_i])[2]
    mu, sd = L[fit_i].mean(0), L[fit_i].std(0) + 1e-6     # learned stacker
    w, b = fit_logistic((L[fit_i] - mu) / sd, is_pie[fit_i])
    learned = ((L[ev_i] - mu) / sd) @ w + b
    auc_learn = roc_auc(learned, is_pie[ev_i])[2]
    print(f"  {'sum-logit':>9}: {auc_sum:.6f}   (delta vs best single = {auc_sum - best:+.6f})")
    print(f"  {'learned':>9}: {auc_learn:.6f}   (delta vs best single = {auc_learn - best:+.6f})")
    print(f"  learned weights (pie,topo,muon,pileup) = {np.array2string(w, precision=2)}")

    fig, ax = plt.subplots(figsize=(5, 5))
    for label, sc in [(f"best single ({bestk})", pie_dir[bestk] * L[ev_i, keys.index(bestk)]),
                      ("sum-of-logits", sum_logit[ev_i]),
                      ("learned stack", learned)]:
        fpr, tpr, auc = roc_auc(sc, is_pie[ev_i])
        ax.plot(fpr, tpr, label=f"{label}  AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("non-signal background acceptance (FPR)"); ax.set_ylabel("pi->e nu efficiency (TPR)")
    ax.set_title("combined keep-score vs best single head")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{output_dir}/fusion_roc.png", dpi=130); plt.close(fig)

    # (4) 2-D score scatter: pie vs muon, colored by truth (subsampled)
    sub = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for mask, c, lab in [(is_pie == 1, "tab:blue", "pi->e nu"),
                         (is_pie == 0, "tab:red", "non-signal background")]:
        xs, ys = data["pie"][mask], data["muon"][mask]
        if len(xs) > 15000:
            j = sub.choice(len(xs), 15000, replace=False); xs, ys = xs[j], ys[j]
        ax.scatter(xs, ys, s=3, alpha=0.25, c=c, label=lab)
    ax.set_xlabel("PieTagger score"); ax.set_ylabel("MuonVeto score")
    ax.set_title("keep = high pie, low muon"); ax.legend(fontsize=8, markerscale=3)
    fig.tight_layout(); fig.savefig(f"{output_dir}/fusion_scatter_pie_muon.png", dpi=130)
    plt.close(fig)

def score_hists(data, output_dir, n_bins=50):
    """Per-head 1-D score distributions: pi->e nu vs non-signal background, overlaid."""
    keys = ["pie", "topo", "muon", "pileup"]
    titles = {"pie": "PieTagger", "topo": "PieTopo",
              "muon": "MuonVeto", "pileup": "PileupVeto"}
    is_pie = data["is_pie"].astype(int)
    sig, bkg = is_pie == 1, is_pie == 0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, k in zip(axes.flat, keys):
        ax.hist(data[k][sig], bins=bins, density=True, histtype="step",
                lw=1.8, color="tab:blue", label=f"pi->e nu (n={int(sig.sum())})")
        ax.hist(data[k][bkg], bins=bins, density=True, histtype="step",
                lw=1.8, color="tab:red", label=f"non-signal background (n={int(bkg.sum())})")
        ax.set_title(f"{titles[k]} score", fontsize=10)
        ax.set_xlabel("score"); ax.set_ylabel("density (per class)")
        ax.set_yscale("log")           # log-y so the overlap tails are visible
        ax.legend(fontsize=8)
    fig.suptitle("per-head score: pi->e nu vs non-signal background (area-normalized)")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/score_hists.png", dpi=130)
    plt.close(fig)

def dif_analysis(data, output_dir, score_key, pos_key, name, fname,
                 n_bins=50, mask=None, tag=""):
    """In-flight (muDIF/piDIF) veto head: three-class score distribution
    (<name> / pi->e nu / non-signal background) + ROC for <name> vs pi->e nu (the discrimination
    that matters — reject the DIF class while keeping signal) and <name> vs non-signal background,
    + a rejection-at-fixed-signal-eff table + survival-efficiency-vs-cut. Classes
    from truth labels: pie=is_pie==1, <name>=<pos_key>==1, non-signal background=neither.

    score_key/pos_key select the head score and its positive truth label; name is
    the display label ('muDIF'/'piDIF') and fname the file/print prefix. mask:
    optional boolean subset (e.g. truth_acceptance==1); tag: file/title suffix."""
    is_pie = data["is_pie"].astype(int)
    is_pos = data[pos_key].astype(int)
    score = data[score_key]
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        is_pie, is_pos, score = is_pie[m], is_pos[m], score[m]
    pie    = is_pie == 1
    pos    = is_pos == 1
    michel = (is_pie == 0) & (is_pos == 0)
    n_pie, n_pos, n_michel = int(pie.sum()), int(pos.sum()), int(michel.sum())
    ttl = "  [acceptance==1]" if tag else ""
    print(f"\n=== {name} head{ttl} ===  pie={n_pie}  {name}={n_pos}  michel={n_michel}",
          flush=True)
    if n_pos == 0:
        print(f"[skip] no {name} events — include tail_reveal_{fname}_eval.parquet in --data",
              flush=True)
        return

    # (1) overlaid score distributions (log-y so the overlap tails are visible)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for msk, c, lab, n in [(pos,    "tab:green", name,       n_pos),
                           (pie,    "tab:blue",  "pi->e nu", n_pie),
                           (michel, "tab:red",   "non-signal background",   n_michel)]:
        if n > 0:
            ax.hist(score[msk], bins=bins, density=True, histtype="step", lw=1.9,
                    color=c, label=f"{lab} (n={n})")
    ax.set_yscale("log")
    ax.set_xlabel(f"{name}-veto score (sigmoid)"); ax.set_ylabel("density (per class)")
    ax.set_title(f"{name} head score: {name} vs pi->e nu vs non-signal background" + ttl)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{output_dir}/{fname}_score_hists{tag}.png", dpi=130); plt.close(fig)

    # (2) ROC: the DIF class is the positive class (should score HIGH); the other
    # class is the negative. AUC = separability. DIF-vs-pie is the operational key.
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for other_mask, other_lab, col in [(pie,    "pi->e nu", "tab:blue"),
                                       (michel, "non-signal background",   "tab:red")]:
        if int(other_mask.sum()) == 0:
            continue
        sel = pos | other_mask
        fpr, tpr, auc = roc_auc(score[sel], is_pos[sel])   # label 1 = DIF class
        ax.plot(fpr, tpr, color=col, label=f"{name} vs {other_lab}  AUC={auc:.4f}")
        print(f"AUC  {name} vs {other_lab:9s} = {auc:.6f}", flush=True)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel(f"other class tagged as {name} (FPR)")
    ax.set_ylabel(f"{name} tagged (TPR)")
    ax.set_title(f"{name} veto ROC" + ttl)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{output_dir}/{fname}_roc{tag}.png", dpi=130); plt.close(fig)

    # (3) operating points: DIF rejection at fixed pi->e nu (signal) efficiency.
    # The veto keeps low-score events, so a pie-efficiency target sets score < thr.
    if n_pie > 0:
        print(f"{name} rejection at fixed pi->e nu signal efficiency "
              "(cut keeps score < thr):", flush=True)
        for sig_eff in (0.99, 0.98, 0.95, 0.90):
            thr = float(np.quantile(score[pie], sig_eff))
            pos_rej = float((score[pos] >= thr).mean())
            michel_rej = float((score[michel] >= thr).mean()) if n_michel else float("nan")
            print(f"  pie_eff={sig_eff:.2f}  thr={thr:.4f}  "
                  f"{fname}_rej={pos_rej:.4f}  michel_rej={michel_rej:.4f}", flush=True)

    # (4) survival efficiency vs the veto cut. The veto keeps score < cut, so each
    # curve is the score CDF. pi->e nu (signal) and non-signal background are KEPT (near 1) -> LEFT
    # LINEAR axis; the DIF class is rejected, so its survival = leakage is small ->
    # RIGHT LOG axis (the separate scale, the only way to read leakage).
    cuts = np.linspace(0.0, 1.0, 201)

    def _surv(msk):                               # fraction of class with score < cut
        s = np.sort(score[msk])
        if len(s) == 0:
            return np.full(cuts.shape, np.nan)
        return np.searchsorted(s, cuts, side="left") / len(s)

    eff_pie, eff_michel, eff_pos = _surv(pie), _surv(michel), _surv(pos)

    fig, ax = plt.subplots(figsize=(8, 5))
    h1, = ax.plot(cuts, eff_pie,    color="tab:blue", lw=2.0, label="pi->e nu (signal, kept)")
    h2, = ax.plot(cuts, eff_michel, color="tab:red",  lw=2.0, label="non-signal background (kept)")
    ax.set_xlabel(f"{name} veto-score cut  (keep events with score < cut)")
    ax.set_ylabel("survival efficiency — pi->e nu / non-signal background  (linear)")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.02); ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    h3, = ax2.plot(cuts, np.where(eff_pos > 0, eff_pos, np.nan),
                   color="tab:green", lw=2.0, ls="--", label=f"{name} (survival = leakage)")
    ax2.set_yscale("log")
    ax2.set_ylim(0.5 / max(n_pos, 1), 1.5)
    ax2.set_ylabel(f"{name} survival / leakage  (log)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    ax.set_title(f"survival efficiency vs {name}-veto cut" + ttl)
    ax.legend(handles=[h1, h2, h3], loc="center left", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{output_dir}/{fname}_eff_vs_cut{tag}.png", dpi=130); plt.close(fig)

def mudif_analysis(data, output_dir, n_bins=50, mask=None, tag=""):
    dif_analysis(data, output_dir, "muon_dif", "is_mudif", "muDIF", "mudif",
                 n_bins=n_bins, mask=mask, tag=tag)

def pidif_analysis(data, output_dir, n_bins=50, mask=None, tag=""):
    dif_analysis(data, output_dir, "pion_dif", "is_pidif", "piDIF", "pidif",
                 n_bins=n_bins, mask=mask, tag=tag)

def dif_tail_bias(data, output_dir, score_key, pos_key, name, fname,
                  e_split=56.0, pie_eff=0.5, accept_min=0.5, e_max=75.0, include_dead=True):
    """Does the muDIF/piDIF veto sculpt the pi->e nu DEPOSITED-energy spectrum?

    Set the veto cut (keep score < thr) so the accepted pi->e nu efficiency
    equals `pie_eff`, then split accepted pi->e nu by deposited (calorimeter)
    energy at `e_split` MeV (peak vs tail) and compare the surviving fractions.

    Why DEPOSITED energy, not the truth positron energy: pi->e nu is two-body,
    so the INITIAL positron is ~monoenergetic (~69.8 MeV) and a 56 MeV split is
    degenerate there. The TAIL the R_e/mu measurement cares about lives in the
    deposited/calorimeter spectrum (shower leakage, radiative pi->e nu gamma),
    where events fall below the peak. An energy-DEPENDENT veto efficiency biases
    the measured tail fraction; to first order the multiplicative bias on the
    tail fraction is eff(tail)/eff(peak) -- 1.0 means the cut is unbiased.

    `e_max` (default 75 MeV) drops pi->e nu with unphysically high deposited
    energy: the positron is monoenergetic at ~69.8 MeV, so deposits above ~75 MeV
    are parquet volume/double-counting artifacts (the GetVolume/GetCaloID LYSO
    bug), not real, and are excluded from both the threshold and the peak bin.

    score_key/pos_key/name/fname select the head (muon_dif/is_mudif/'muDIF'/'mudif'
    or pion_dif/is_pidif/'piDIF'/'pidif'). Also reports the DIF survival (leakage)
    and suppression at the same cut. All on the acceptance-passing population."""
    score   = data[score_key]
    is_pie  = data["is_pie"].astype(int)
    is_pos  = data[pos_key].astype(int)
    # Reco positron energy = live calorimeter deposit + dead-material loss. Adding dead_E
    # back is essential: dead-material loss grows with angle, so a live-only "tail" is
    # partly an angle artifact (a full-energy positron mislabeled as tail). pi->e nu is
    # monoenergetic ~69.8 MeV, so this is the energy the peak/tail split should cut on.
    dep_E   = data["dep_energy"] + (data["dead_E"] if include_dead else 0.0)
    acc = (data["acceptance"] >= accept_min) if "acceptance" in data \
        else np.ones(len(score), bool)
    finite = np.isfinite(dep_E)
    good_E = dep_E <= e_max          # drop unphysical >e_max deposits (volume artifacts)

    n_artifact = int(((is_pie == 1) & acc & finite & (dep_E > e_max)).sum())
    pie = (is_pie == 1) & acc & finite & good_E
    pos = (is_pos == 1) & acc
    n_pie, n_pos = int(pie.sum()), int(pos.sum())

    print(f"\n=== {name} veto: pi->e nu tail bias @ accepted-pie eff={pie_eff:.2f}"
          f"  (split {e_split:g} MeV on DEPOSITED energy) ===", flush=True)
    print(f"  dropped {n_artifact} accepted-pie with deposited E > {e_max:g} MeV "
          f"(unphysical / volume artifacts)", flush=True)
    if n_pie == 0:
        print("[skip] no accepted pi->e nu events", flush=True)
        return

    # veto keeps score < thr; the pie_eff-quantile of accepted-pie scores is the
    # threshold whose fraction-below == pie_eff (matches dif_analysis convention).
    thr = float(np.quantile(score[pie], pie_eff))

    def eff_err(mask):
        s = score[mask]
        n = len(s)
        if n == 0:
            return float("nan"), float("nan"), 0
        e = float((s < thr).mean())
        return e, float(np.sqrt(max(e * (1.0 - e), 0.0) / n)), n

    eff_all, err_all, _ = eff_err(pie)
    eff_hi, err_hi, n_hi = eff_err(pie & (dep_E >= e_split))   # peak
    eff_lo, err_lo, n_lo = eff_err(pie & (dep_E <  e_split))   # tail
    diff = eff_lo - eff_hi
    diff_err = float(np.sqrt(err_lo ** 2 + err_hi ** 2))
    if eff_hi > 0 and eff_lo > 0:
        ratio = eff_lo / eff_hi
        ratio_err = ratio * np.sqrt((err_lo / eff_lo) ** 2 + (err_hi / eff_hi) ** 2)
    else:
        ratio, ratio_err = float("nan"), float("nan")

    print(f"  cut: keep {name}-score < thr={thr:.4f}", flush=True)
    print(f"  accepted pi->e nu  N={n_pie:6d}  overall eff={eff_all:.4f} +/- {err_all:.4f}"
          f"  (target {pie_eff:.2f})", flush=True)
    print(f"    peak  E >= {e_split:g} MeV   N={n_hi:6d}  eff={eff_hi:.4f} +/- {err_hi:.4f}",
          flush=True)
    print(f"    tail  E <  {e_split:g} MeV   N={n_lo:6d}  eff={eff_lo:.4f} +/- {err_lo:.4f}",
          flush=True)
    print(f"    BIAS  eff(tail) - eff(peak) = {diff:+.4f} +/- {diff_err:.4f}", flush=True)
    print(f"    BIAS  eff(tail) / eff(peak) = {ratio:.4f} +/- {ratio_err:.4f}"
          f"   (mult. bias on tail fraction; 1.0 = unbiased)", flush=True)
    if n_lo < 30:
        print(f"    [warn] only {n_lo} accepted-pie tail events (<{e_split:g} MeV) — "
              f"eff(tail) is stat-limited; raise --max_events / use a bigger pie eval set",
              flush=True)

    eff_p, err_p, n_p = eff_err(pos)
    if n_p == 0:
        print(f"  accepted {name}: none in sample (include the {fname} eval parquet)", flush=True)
    elif eff_p > 0:
        print(f"  accepted {name}     N={n_p:6d}  survival(leakage)={eff_p:.4f} +/- {err_p:.4f}"
              f"  rejection={1.0 - eff_p:.4f}  suppression={1.0 / eff_p:.1f}x", flush=True)
    else:
        print(f"  accepted {name}     N={n_p:6d}  survival(leakage)=0/{n_p} "
              f"(>{n_p:d}x suppression; stat-limited)", flush=True)

    # plot: accepted-pie survival vs deposited energy -> flat at pie_eff = unbiased
    surv = (score[pie] < thr).astype(float)
    e = dep_E[pie]
    bins = np.linspace(0.0, e_max, 26)
    idx = np.digitize(e, bins) - 1
    cx, cy, ce = [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        n = int(m.sum())
        if n < 3:
            continue
        ee = float(surv[m].mean())
        cx.append(0.5 * (bins[b] + bins[b + 1]))
        cy.append(ee)
        ce.append(np.sqrt(max(ee * (1.0 - ee), 0.0) / n))
    fig, ax = plt.subplots(figsize=(8, 5))
    if cx:
        ax.errorbar(cx, cy, yerr=ce, fmt="o-", color="tab:blue", lw=1.6, ms=4,
                    capsize=2, label="accepted pi->e nu survival")
    ax.axhline(pie_eff, color="k", ls=":", lw=1.0, label=f"target eff = {pie_eff:.2f}")
    ax.axvline(e_split, color="tab:red", ls="--", lw=1.2, label=f"{e_split:g} MeV split")
    ax.annotate(f"tail eff={eff_lo:.3f}", (e_split - 1, max(eff_lo, 0.02)),
                ha="right", fontsize=9, color="tab:red")
    ax.annotate(f"peak eff={eff_hi:.3f}", (e_split + 1, eff_hi), ha="left",
                fontsize=9, color="tab:red")
    ax.set_xlabel("deposited (calorimeter) energy (MeV)")
    ax.set_ylabel(f"survival at {name} cut (accepted pi->e nu eff={pie_eff:.2f})")
    ax.set_title(f"{name} veto: pi->e nu survival vs deposited energy (flat = unbiased)")
    ax.set_ylim(0.0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/{fname}_tail_bias.png", dpi=130)
    plt.close(fig)

def mudif_tail_bias(data, output_dir, e_split=56.0, pie_eff=0.5, accept_min=0.5, e_max=75.0,
                    include_dead=True):
    dif_tail_bias(data, output_dir, "muon_dif", "is_mudif", "muDIF", "mudif",
                  e_split=e_split, pie_eff=pie_eff, accept_min=accept_min, e_max=e_max,
                  include_dead=include_dead)

def pidif_tail_bias(data, output_dir, e_split=56.0, pie_eff=0.5, accept_min=0.5, e_max=75.0,
                    include_dead=True):
    dif_tail_bias(data, output_dir, "pion_dif", "is_pidif", "piDIF", "pidif",
                  e_split=e_split, pie_eff=pie_eff, accept_min=accept_min, e_max=e_max,
                  include_dead=include_dead)
