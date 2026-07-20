"""Build a physically-motivated toy reco dataset from the eval pools.

Used to develop/test the R_e/mu analysis (esp. the time fit with a PILEUP component) before a
trained model exists. Writes {tag}_events.parquet in the same schema benchmark.py produces, so
plot_part1.py and purity_analysis.run_analysis consume it unchanged.

Assumptions (user-specified):
  * Gaussian resolution on energy, time, angle (sigma_E, sigma_t, sigma_theta).
  * PERFECT pion-stop reconstruction (acceptance fiducial uses truth pion stop).
  * 5% of events that HAVE a second (pileup) positron -> the WRONG positron is reconstructed
    as the trigger: its time is ~uniform in the readout window and its energy ~ the michel
    (pileup) deposit spectrum.  These produce the flat 'pileup' component the time fit models.
    They are flagged `toy_wrong_positron` so pure prompt/delayed templates can be built from
    the correctly-reconstructed events (mimicking pure-MC control samples).
"""
import argparse
import os
import numpy as np
import pandas as pd

# acceptance rule constants (models_v2.py:1299-1309 / constants.py)
ACCEPT_Z = (1.2, 4.8)
ACCEPT_XY = 8.0
ACCEPT_THETA_MAX = np.deg2rad(120.0)
WINDOW = (-300.0, 500.0)


class ToyConfig:
    def __init__(self, sigma_E=2.0, sigma_t=1.0, sigma_theta=0.1, wrong_frac=0.05, seed=0):
        self.sigma_E = sigma_E          # MeV
        self.sigma_t = sigma_t          # ns
        self.sigma_theta = sigma_theta  # rad
        self.wrong_frac = wrong_frac     # fraction of pileup events with wrong-positron reco
        self.seed = seed


def _build_pool(d, tag, cfg, rng, pileup_E_pool):
    n = len(d)
    z = d.truth_pion_stop_z.to_numpy(); x = d.truth_pion_stop_x.to_numpy(); y = d.truth_pion_stop_y.to_numpy()
    fid = (z > ACCEPT_Z[0]) & (z < ACCEPT_Z[1]) & (np.abs(x) < ACCEPT_XY) & (np.abs(y) < ACCEPT_XY)  # perfect pion stop

    theta_r = d.truth_theta.to_numpy() + rng.normal(0, cfg.sigma_theta, n)   # smeared angle
    angle_ok = theta_r < ACCEPT_THETA_MAX

    # POSITRON energy (what the calorimeter measures) -- NOT live_E, which is total deposited
    # energy including the pion+muon stopping in the ATAR (that fakes a ~70 MeV michel peak).
    E_t = d.truth_positron_energy.clip(lower=0).to_numpy()
    t_t = d.truth_positron_t.to_numpy()
    inwin_t = t_t > -999
    E_r = E_t + rng.normal(0, cfg.sigma_E, n)                               # smeared energy
    t_r = np.where(inwin_t, t_t + rng.normal(0, cfg.sigma_t, n), -999.0)    # smeared time

    # --- 5% wrong-positron among events that have a pileup (second) positron ---
    has_pileup = d.truth_has_atar_pileup.to_numpy() > 0.5
    wrong = has_pileup & (rng.uniform(0, 1, n) < cfg.wrong_frac)
    nw = int(wrong.sum())
    if nw:
        t_r[wrong] = rng.uniform(WINDOW[0], WINDOW[1], nw)                  # pileup positron: flat in time
        E_r[wrong] = rng.choice(pileup_E_pool, nw) + rng.normal(0, cfg.sigma_E, nw)  # michel-like energy

    accepted = fid & angle_ok    # htp assumed perfect; the wrong-positron case still fires htp

    return pd.DataFrame({
        # reco
        "pred_accepted": np.where(accepted, 1.0, 0.0).astype(np.float32),
        "pred_positron_energy": np.clip(E_r, 0, None).astype(np.float32),
        "pred_dead_energy": np.zeros(n, np.float32),          # folded into pred_positron_energy here
        "pred_positron_time_ns": t_r.astype(np.float32),
        "pred_positron_time_spread_ns": np.abs(rng.normal(cfg.sigma_t, 0.3, n)).astype(np.float32),
        "pred_htp": np.ones(n, np.float32),
        "pred_pos_iou": np.ones(n, np.float32),
        # truth passthrough (as benchmark.py writes)
        "event_id": np.arange(n, dtype=np.int64),
        "truth_gen_weight": d.gen_weight.to_numpy(np.float64),
        "truth_positron_t": t_t.astype(np.float32),
        "truth_dead_E": d.dead_E.to_numpy(np.float32),
        "truth_htp": np.ones(n, np.int8),
        "truth_acceptance": d.truth_acceptance.to_numpy(np.int32),
        "truth_positron_energy": d.truth_positron_energy.to_numpy(np.float32),
        "truth_is_pie": d.truth_is_pie.to_numpy(np.int8),
        "truth_has_muon": d.truth_has_muon.to_numpy(np.int8),
        # toy bookkeeping (truth for the pileup component)
        "toy_wrong_positron": wrong.astype(np.int8),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline/mixed_standard")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sigma_E", type=float, default=2.0)
    ap.add_argument("--sigma_t", type=float, default=1.0)
    ap.add_argument("--sigma_theta", type=float, default=0.1)
    ap.add_argument("--wrong_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = ToyConfig(args.sigma_E, args.sigma_t, args.sigma_theta, args.wrong_frac, args.seed)
    rng = np.random.default_rng(cfg.seed)

    michel = pd.read_parquet(f"{args.src_dir}/pimu_benchmark_5_11/data.parquet")
    # pileup (second) positron energy source: in-window michel positron energies (Michel spectrum)
    mE = michel.truth_positron_energy.clip(lower=0).to_numpy()
    pileup_E_pool = mE[(michel.truth_positron_t.to_numpy() > -999) & (mE > 0)]

    for src, tag in [("pie_benchmark_5_11", "pie_eval"), ("pimu_benchmark_5_11", "pimu_eval")]:
        d = pd.read_parquet(f"{args.src_dir}/{src}/data.parquet")
        ev = _build_pool(d, tag, cfg, rng, pileup_E_pool)
        ev.to_parquet(f"{args.out_dir}/{tag}_events.parquet")
        print(f"  toy {tag}: {len(ev)} events, {int(ev.toy_wrong_positron.sum())} wrong-positron "
              f"({100*ev.toy_wrong_positron.mean():.2f}%), accepted {100*ev.pred_accepted.mean():.1f}%")
    print(f"toy reco -> {args.out_dir}  (sigma_E={cfg.sigma_E} sigma_t={cfg.sigma_t} sigma_theta={cfg.sigma_theta} wrong={cfg.wrong_frac})")


if __name__ == "__main__":
    main()
