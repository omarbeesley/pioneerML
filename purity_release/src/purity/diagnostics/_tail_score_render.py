"""Historical render sequence separated from inference."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ._tail_score_functions import *

def render(data, args):
    is_pie = data["is_pie"].astype(int)
    # ---- ROC: holistic vs topology head ----
    fig, ax = plt.subplots(figsize=(5, 5))
    aucs = {}
    for name, key in [("PieTagger (holistic)", "pie"), ("PieTopo (time-group graph)", "topo")]:
        fpr, tpr, auc = roc_auc(data[key], is_pie)
        aucs[key] = auc
        ax.plot(fpr, tpr, label=f"{name}  AUC={auc:.4f}")
        print(f"AUC  {name:28s} = {auc:.6f}", flush=True)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("non-signal acceptance (FPR)")
    ax.set_ylabel("pi->e nu efficiency (TPR)")
    ax.set_title("pi->e nu vs all non-signal backgrounds")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{args.output_dir}/roc.png", dpi=130); plt.close(fig)

    # ---- veto ROCs: muon / pileup vetoes vs their OWN truth labels ----
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, key, lbl in [
        ("MuonVeto (muon_present)",     "muon",   data["muon_present"].astype(int)),
        ("PileupVeto (pileup_present)", "pileup", data["pileup_present"].astype(int)),
    ]:
        if len(np.unique(lbl)) < 2:
            print(f"[skip] {name}: a truth class is absent")
            continue
        fpr, tpr, auc = roc_auc(data[key], lbl)
        ax.plot(fpr, tpr, label=f"{name}  AUC={auc:.4f}")
        print(f"AUC  {name:28s} = {auc:.6f}", flush=True)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("muon / pileup veto ROC (vs own truth)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{args.output_dir}/veto_roc.png", dpi=130); plt.close(fig)

    # ---- flatness (unbiasedness): mean score vs energy / decay time ----
    sig = is_pie == 1
    bkg = is_pie == 0
    # Two families:
    #  - non-signal background BACKGROUND: pie / pie_topo fake-rate must be flat, else the
    #    subtracted non-signal background shape is distorted.
    #  - pi->e nu SIGNAL: the muon / pileup VETO scores must be flat, else the
    #    veto sculpts the surviving signal spectrum. (pi->e nu is ~monoenergetic,
    #    so the signal-vs-energy plot is usually degenerate and auto-skipped.)
    vars_ = [("dep_energy", "truth deposited energy (MeV)", False),
             ("ptime",      "truth decay time (ns)",        True)]
    families = [
        ("bkg", bkg, "non-signal background",   [("pie", "PieTagger"), ("topo", "PieTopo")],
         "mean pie-score (non-signal background fake rate)"),
        ("sig", sig, "pi->e nu", [("muon", "MuonVeto"), ("pileup", "PileupVeto")],
         "mean veto-score (pi->e nu)"),
    ]
    for fam, pop, poplabel, series, ylabel in families:
        for var, xlabel, drop_sentinel in vars_:
            x = data[var][pop]
            ys = {k: data[k][pop] for k, _ in series}
            m = np.isfinite(x)
            if drop_sentinel:                # truth_positron_t = -1000 if out of window
                m &= x > -100.0
            x = x[m]; ys = {k: v[m] for k, v in ys.items()}
            if len(x) < 20:
                print(f"[skip] {fam} flatness vs {var}: only {len(x)} events", flush=True)
                continue
            bins = np.unique(np.quantile(x, np.linspace(0, 1, args.n_bins + 1)))
            if len(bins) < 3:
                print(f"[skip] {fam} flatness vs {var}: degenerate "
                      f"(<2 bins, e.g. monoenergetic signal)", flush=True)
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            for k, nm in series:
                c, mn = binned_mean(x, ys[k], bins)
                ax.plot(c, mn, "o-", label=f"{nm} (slope={slope(c, mn):.2e})")
                print(f"{fam} flatness vs {var:6s}  {nm:10s} slope = {slope(c, mn):.4e}", flush=True)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_title(f"{poplabel}: score vs {xlabel} -- flat = unbiased")
            ax.legend(fontsize=8); fig.tight_layout()
            fig.savefig(f"{args.output_dir}/flatness_{fam}_{var}.png", dpi=130); plt.close(fig)

    # ---- per-head score distributions (signal vs background) ----
    score_hists(data, args.output_dir)

    # ---- muDIF / piDIF heads: 3-class score distribution + ROC + operating points ----
    # ...also on the acceptance-passing subset (truth_acceptance==1), the physically
    # relevant population (pion-stop fiducial + positron angle < 120).
    acc_mask = (data["acceptance"] >= 0.5) if "acceptance" in data else None
    for analysis, label in [(mudif_analysis, "muDIF"), (pidif_analysis, "piDIF")]:
        analysis(data, args.output_dir)
        if acc_mask is not None:
            if int(acc_mask.sum()) > 0:
                analysis(data, args.output_dir, mask=acc_mask, tag="_accept")
            else:
                print(f"[skip] no acceptance==1 events for the {label} accept-only plots",
                      flush=True)

    # ---- pi->e nu peak-vs-tail bias of each DIF veto at a fixed pie efficiency ----
    mudif_tail_bias(data, args.output_dir,
                    e_split=args.bias_e_split, pie_eff=args.bias_pie_eff,
                    e_max=args.bias_e_max, include_dead=not args.bias_no_dead_E)
    pidif_tail_bias(data, args.output_dir,
                    e_split=args.bias_e_split, pie_eff=args.bias_pie_eff,
                    e_max=args.bias_e_max, include_dead=not args.bias_no_dead_E)

    # ---- head fusion / orthogonality + combined keep-score ----
    head_fusion(data, args.output_dir, target_eff=args.leak_sig_eff)

    print(f"\nwrote roc.png, veto_roc.png, score_hists.png, "
          f"{{mudif,pidif}}_score_hists[_accept].png, {{mudif,pidif}}_roc[_accept].png, "
          f"{{mudif,pidif}}_eff_vs_cut[_accept].png, {{mudif,pidif}}_tail_bias.png, "
          f"flatness_{{bkg,sig}}_*.png, fusion_corr.png, fusion_roc.png, "
          f"fusion_scatter_pie_muon.png to {args.output_dir}/", flush=True)
