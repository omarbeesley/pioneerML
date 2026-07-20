"""I/O + standardization for the PIONEER R_e/mu analysis.

Loads the two eval pools (pie = pi->e nu, michel = pi->mu->e) and reduces each event
to a small standardized frame the measurement code consumes. Works in two modes:

  mode='truth'  -> use MC-truth fields (validate the framework logic with no model)
  mode='reco'   -> use the model's predictions from benchmark.py's {tag}_events.parquet

The physical branching ratio is carried by `gen_weight` (pie ~1.233e-4, michel ~1); every
histogram fill MUST be gen_weight-weighted or the pie:michel normalization is wrong.

Sentinels: truth_positron_t = -1000 and reco pred time ~ -999 mean 'positron not in the
readout window' -> excluded from spectrum/time fits (this window inefficiency is physical).
"""
import numpy as np
import pandas as pd

# ---- analysis constants (overridable via AnalysisConfig) ----
E_SPLIT_MEV = 56.0     # high/low energy bin boundary
# NO upper energy cap by default: events reconstructed above the pi->e peak (reco ~90-120 MeV,
# truth deposited ~70) are pileup-merge OVER-reconstructions of genuine in-acceptance signal.
# A 75 MeV cap sits just above the pi->e peak and cuts them channel-ASYMMETRICALLY (pie + merged
# pileup goes over the cap, michel at 30-50 MeV stays under) -> -0.40% R_e/mu acceptance bias on
# gatefix10M; uncapped the bias is -0.04%. Use a finite cap only as a cross-check.
E_MAX_MEV   = float("inf")
MICHEL_ENDPOINT_MEV = 55.0   # michel positron endpoint ~52.8; above this a michel-pool event is
                             # actually a direct pi->e nu (provided by the dedicated pie pool) -> drop
                             # to avoid double-counting pi->e across the two pools
TIME_WINDOW = (-300.0, 500.0)   # mixer readout window (ns), matches pileup_mixer
T_SENTINEL  = -999.0   # anything <= this is 'no in-window positron'


class AnalysisConfig:
    def __init__(self, mode="truth", e_split=E_SPLIT_MEV, e_max=E_MAX_MEV,
                 window=TIME_WINDOW, energy="true_ke",
                 time_col="pred_positron_time_ns", clean_tol=5.0):
        assert mode in ("truth", "reco")
        assert energy in ("deposited", "true_ke")
        self.mode = mode
        self.e_split = e_split
        self.e_max = e_max
        self.window = window
        self.energy = energy       # truth-mode energy variable
        self.time_col = time_col   # reco positron-time column: plain mean or consensus (dominant-slice)
        self.clean_tol = clean_tol # ns; |reco - truth_trigger| < clean_tol => reco found the trigger


def _col(df, name, default=np.nan):
    return df[name].to_numpy() if name in df.columns else np.full(len(df), default)


def load_pool(path, tag, cfg):
    """Load one eval pool parquet -> standardized event frame.

    Standardized columns:
      tag       : 'pie' | 'michel'
      w         : gen_weight (physical BR normalization)
      cls       : 'pie' | 'mue' | 'other'   (trigger-positron class)
      E         : energy that fills the spectrum (MeV)
      t         : positron time (ns); NaN if out of window
      in_window : bool, positron time is inside the readout window
      accepted  : bool, event passes the (model or truth) acceptance
    """
    df = pd.read_parquet(path)
    # michel pool = pi->mu->e only: drop the direct pi->e nu decays present at their natural
    # BR (~1.2e-4) in the unforced sample -- pi->e is provided by the dedicated pie pool.
    # Primary cut: the kPienu decay-mode bit (0x1) in event_type, which also catches the
    # RADIATIVE pi->e gamma tail (positron pulled below the michel endpoint, ~1.4% of pie)
    # that the energy cut leaks.  Energy cut kept as fallback for pools without the column.
    if tag != "pie":
        et_col = next((c for c in ("truth_event_type", "event_type") if c in df.columns), None)
        if et_col is not None:
            et = df[et_col].astype("int64")
            # drop direct pi->e (kPienu, 0x1) AND radiative muon decays (kMurad, 0x200):
            # RMD is not modeled by the current reconstruction (no tagger) and its weighted
            # events dominate the high-bin variance; excluded from pools by policy.
            df = df[((et & 0x1) == 0) & ((et & 0x200) == 0)].reset_index(drop=True)
        if "truth_positron_energy" in df.columns:
            df = df[df["truth_positron_energy"] <= MICHEL_ENDPOINT_MEV].reset_index(drop=True)
    n = len(df)
    out = pd.DataFrame(index=np.arange(n))
    out["tag"] = tag
    # weight: mixed parquet calls it 'gen_weight'; benchmark.py events parquet 'truth_gen_weight'
    wcol = "gen_weight" if "gen_weight" in df.columns else "truth_gen_weight"
    out["w"] = _col(df, wcol, 1.0).astype(float)

    # --- trigger-positron class = the generated POOL (these are pure MC control samples;
    #     the pie pool's trigger is always prompt pi->e, the michel pool's is delayed
    #     pi->mu->e).  Per-event truth flags are kept for diagnostics only, since
    #     truth_has_muon is contaminated by pileup muons and misses ~3% of michels. ---
    out["cls"] = "pie" if tag == "pie" else "mue"
    out["truth_is_pie"] = _col(df, "truth_is_pie", 0) == 1
    out["truth_has_muon"] = _col(df, "truth_has_muon", 0) == 1
    # wrong-positron / pileup contamination (toy bookkeeping). 'clean' = correctly-reco
    # trigger, used to build pure prompt/delayed time templates. Absent -> all clean.
    wrong = _col(df, "toy_wrong_positron", 0) > 0
    out["toy_wrong"] = wrong

    # --- energy, time, acceptance: truth vs reco ---
    if cfg.mode == "truth":
        if cfg.energy == "deposited":
            E = np.clip(_col(df, "live_E", 0.0), 0, None) + np.clip(_col(df, "dead_E", 0.0), 0, None)
        else:
            E = _col(df, "truth_positron_energy", np.nan)
        t = _col(df, "truth_positron_t", T_SENTINEL - 1)
        accepted = _col(df, "truth_acceptance", 0.0) > 0.5
    else:  # reco
        E = np.clip(_col(df, "pred_positron_energy", 0.0), 0, None) + \
            np.clip(_col(df, "pred_dead_energy", 0.0), 0, None)
        t = _col(df, cfg.time_col, T_SENTINEL - 1)          # plain mean or consensus (cfg.time_col)
        accepted = _col(df, "pred_accepted", 0.0) >= 0.5

    lo, hi = cfg.window
    in_window = (t > T_SENTINEL) & (t >= lo) & (t <= hi)
    # clean = the reconstructed positron IS the trigger (its reco time matches the truth trigger
    # time), NOT an accidental.  This is what the efficiency must count to stay consistent with
    # the physical fit (which sends accidental-reconstructed events into the flat term).  In
    # truth mode the reco time IS the truth time, so any in-window trigger positron is clean.
    tt_trig = _col(df, "truth_positron_t", T_SENTINEL - 1)
    if cfg.mode == "truth":
        clean = tt_trig > T_SENTINEL
    else:
        clean = (tt_trig > T_SENTINEL) & (np.abs(t - tt_trig) < cfg.clean_tol)
    out["E"] = E
    out["t"] = np.where(in_window, t, np.nan)
    out["in_window"] = in_window
    out["accepted"] = accepted.astype(bool)
    out["clean"] = clean & (~wrong)
    return out


def load_all(pie_path, michel_path, cfg):
    """Load both pools and concatenate into one physically-normalized sample.

    Luminosity match: gen_weight carries the per-stopped-pion BR (pie ~1.23e-4, michel ~1),
    so Sw_pie/Sw_mue = (N_pie/N_michel)*BR only equals the physical BR when the two pools
    hold the SAME number of stopped pions.  Unbalanced pools (e.g. 1e5 pie + 1e7 michel)
    otherwise under-weight the smaller pool by N_michel/N_pie -> R_e/mu wrong by that factor.
    Rescale the pie pool onto the michel pool's luminosity so the measurement is pool-size
    agnostic.  (Balanced pools -> factor 1, no change.)"""
    pie = load_pool(pie_path, "pie", cfg)
    mic = load_pool(michel_path, "michel", cfg)
    n_pie, n_mic = len(pie), len(mic)
    if n_pie > 0 and n_mic > 0:
        pie["w"] = pie["w"] * (n_mic / n_pie)
    return pd.concat([pie, mic], ignore_index=True)


def selected(df, require_window=True):
    """The measurement sample: accepted (+ in-window + energy below E_MAX)."""
    m = df["accepted"].to_numpy()
    if require_window:
        m = m & df["in_window"].to_numpy()
    return df[m]


def energy_bin(E, e_split=E_SPLIT_MEV):
    """'high' if E >= e_split else 'low'."""
    return np.where(np.asarray(E) >= e_split, "high", "low")
