# Hard events the main PURITY model reconstructs correctly

A curated set of **96 difficult π→μ→e events** that the main PURITY model reconstructs correctly,
chosen to showcase the model's capability. Each event is hard in one of three physically distinct
ways, and in every case the model correctly picks the daughter positron out of the clutter and
reconstructs its energy, time, and direction.

The model's predictions are **already stored in the parquet**, so generating displays is a single
step that needs only `pandas` and `matplotlib` — no model, GPU, or re-evaluation.

- **"Correct"** means: positron-hit IoU ≥ 0.9 (which hits are the positron), reconstructed energy
  within 3.5 MeV of truth, and reconstructed time within 6 ns of truth. In practice **95 of the 96
  have IoU exactly 1.0** (the other 0.95), mean |ΔE| ≈ 1.1 MeV, mean |Δt| ≈ 0.05 ns.

## The three "hard" categories

| Category | n | What makes it hard |
|---|---|---|
| `pimu_slice` | 33 | The **triggering pion and muon deposit energy in the same readout time-slice** (prompt, ~4 ns). Their energies overlap in the detector and the model must not confuse that blob with the positron. |
| `pileup` | 31 | The signal decay is **buried under several overlapping pile-up events** (3–6 independent events in the tracker). The model must isolate the one triggering positron. |
| `scatter` | 32 | The daughter **positron scatters sharply** while crossing the silicon tracker (curling / kinked track). The model must follow it and still recover its energy. |

## The accidental-close selection (`hard_events_accidental.parquet`)

A separately curated fourth set of **27 events** targeting the sharpest possible confusion for a
vertex-based reconstruction: an **accidental positron from a different decay has a track endpoint
within 1 mm of the point where the signal positron is created** (its emission vertex at the stopped
muon). In several events the accidental *stops* right at that vertex; in one, a second michel
positron is emitted 0.59 mm and only 10 ns away from the signal decay. The model still tags the
signal positron perfectly (**all 27 at IoU = 1.0**, not one accidental hit picked up), with mean
|ΔE| ≈ 0.8 MeV and sub-0.1 ns time residuals.

Selection: for each pile-up positron track (≥ 5 hits, ≥ 3 mm long), take its two endpoints (extreme
hits along the track axis, true-3D coordinates) and require the closer one to lie < 1 mm from the
signal positron's true emission point; ranked by that distance (0.26–1.0 mm). The selection was
independently re-derived and adversarially checked: each accidental is a coherent single-decay
positron track, the endpoint distances are conservative (hit-cloud minima are even closer), and the
emission point coincides with the stopped-muon position to ~0.02 mm. Same schema and correctness
gate as the main set, plus extra columns: `acc_dend_mm` (endpoint → vertex distance),
`acc_dend_dt_ns` (time between that endpoint and the signal decay), `acc_end_kind` (0 ≈ track's
early end, 1 ≈ late end — time-based and noisy, treat as indicative), `acc_len_mm` (accidental
track length), `acc_d3min_mm`/`acc_tgap_ns`/`acc_nclose` (track-to-track proximity, for reference),
and `truth_positron_start_x/y/z` (the vertex).

Displays for this category additionally **ring the accidental hits near the vertex (olive) in the
MODEL panels** — correctly untagged next to the tagged signal track — and draw the accidental
positron's arrival time in the energy-vs-time panel.

## Clean reference signals (`clean_signal_events.parquet`)

Twenty typical clean events for contrast with the hard sets: the **first 10 of each channel, in
file order** — no cherry-picking beyond requiring no pile-up / accidental / decay-in-flight and
that the model reconstructs them correctly (all 20 at IoU = 1.0, |ΔE| ≤ 2.8 MeV, |Δt| ≤ 0.06 ns):

- **10 `pie` (π→eν)**: the pion stops and emits a prompt **monoenergetic 69.8 MeV** positron
  (decay times 10–74 ns).
- **10 `pimue` (π→μ→e)**: the pion stops, the muon stops, and a **delayed Michel positron** is
  emitted (18–50 MeV, 30–414 ns); the energy-vs-time panel shows the prompt π/μ deposits and the
  well-separated late positron.

These come from the pie and pimu benchmark files respectively (the `source` column records which),
rendered the same way (`--input clean_signal_events.parquet`).

## Files

| File | What it is |
|---|---|
| `hard_events_100.parquet` | **The main dataset** (96 events, three categories). One row per event, holding the per-hit tracker data, the truth particle labels, **and the stored model predictions** (per-hit positron tag, reco energy/time/direction). |
| `hard_events_accidental.parquet` | **The accidental-close dataset** (27 events, category `accidental_close`): an accidental positron's track endpoint lands at the signal positron's emission vertex. Same schema plus the `acc_*` proximity columns. |
| `clean_signal_events.parquet` | **Clean reference signals** (20 events): the first 10 clean `pie` (π→eν) + first 10 clean `pimue` (π→μ→e) in file order, no pile-up / accidental / decay-in-flight, all IoU = 1.0. Same schema plus a `source` column naming the benchmark file each came from. |
| `hard_events_100.csv`, `hard_events_accidental.csv` | Human-readable indexes (`orig_idx`, `category`, reco-vs-truth summary metrics — no hit arrays). |
| `make_event_displays.py` | Renders one PNG event display per event, directly from either parquet. |
| `pimu_slice_*.png`, `pileup_*.png`, `scatter_*.png`, `accidental_close_*.png` | Already-rendered example displays (two per category). |

## Making event displays

One command — needs only `pandas` and `matplotlib`:

```bash
python make_event_displays.py --input hard_events_100.parquet        --outdir displays
python make_event_displays.py --input hard_events_accidental.parquet --outdir displays
```

Renders every event to `displays/<category>_<orig_idx>.png`. Options:

- `--category {pimu_slice,pileup,scatter,accidental_close}` — only that category
- `--index <orig_idx>` — a single event
- `--dpi <n>` — output resolution (default 140)

## What each display shows

Two projections (top **x–z**, side **y–z**), each shown **TRUTH** next to **MODEL**:

- **TRUTH** — hits colored by particle: pion red, muon blue, π+μ-merged-pixel purple, signal
  positron green, other EM olive.
- **MODEL** — every hit faded grey except the positron the model identified, drawn bold green with
  its reconstructed direction arrow. It lands on the true positron.

A right-hand **energy-vs-time** panel shows the tracker hits, with the model's reconstructed
positron time landing on the true positron time. A one-line result across the top states the
reconstructed energy and time versus truth.

## Parquet columns

**Per-event scalars:** `orig_idx` (event id), `category`.
**Stored model predictions:** `recoE` (MeV), `pred_time` (ns), `pred_accepted`, `pred_dir_x/y/z`
(reconstructed positron direction), `iou` (positron-hit IoU vs truth).
**Truth:** `truth_positron_energy`, `truth_positron_t`, `truth_has_muon`, `truth_has_atar_pileup`,
`truth_pion_stop_x/y/z`.

**Per-hit arrays** (one list per event, one entry per tracker hit): geometry `hx,hy,hz` (mm),
`hE` (MeV), `ht` (ns); `hview` (0 = x–z, 1 = y–z); `hslice` (readout time-slice id); `horigin`
(source event id: 0 = triggering decay, ≥1 = pile-up); truth labels `hpion,hmuon,hmip_truth`
(0/1); model outputs `htrig,hmip,hpion_prob` (per-hit probabilities), `htrig_pos` (model-tagged
positron hit, 0/1), `htrue_pos` (true triggering-positron hit, 0/1). Calorimeter arrays
`lyso_z,lyso_E,lyso_t,lyso_pdg` are carried but not drawn.

The summary metrics (`pm_share`, `n_origins`, `scatter`, `pos_track_len_mm`, `n_positron_hits`,
`n_hits`, `dE`, `dt`, …) are in `hard_events_100.csv`.
