# PURITY — Preliminary Studies Inventory

**Status:** preliminary results below are on low/interim statistics. A high-stats production is
running now (main model `gatefix10M` ep14 on **10M non-radiative pie + ~20M in-window michel**);
refreshed R_e/μ closure and DIF-suppression numbers expected **by Monday**.

All plots are collected under **[`unified_reco/preliminary_studies/`](unified_reco/preliminary_studies/)**,
one subfolder per study (self-contained — copy/zip the whole folder). Model = PURITY (single
multi-task graph-transformer: pion stop, positron direction/energy/time, acceptance in one forward
pass). Two variants appear: the **main** reconstruction model (R_e/μ pipeline) and the
**tail-reveal / veto** model (muon-DIF, pion-DIF, pie, topology heads).

---

## 1. Main-model reconstruction vs truth → [`01_reconstruction/`](unified_reco/preliminary_studies/01_reconstruction/)
One network reproduces the standard-analysis observables; truth-closure **0.997**.

| Plot | Shows |
|---|---|
| [energy_by_class.png](unified_reco/preliminary_studies/01_reconstruction/energy_by_class.png) | positron energy reco vs truth, per class |
| [dead_energy_resolution.png](unified_reco/preliminary_studies/01_reconstruction/dead_energy_resolution.png) | dead-layer (undetected) energy recovery |
| [htp_performance.png](unified_reco/preliminary_studies/01_reconstruction/htp_performance.png) | hit-to-particle grouping performance |
| [time_residual.png](unified_reco/preliminary_studies/01_reconstruction/time_residual.png), [time_residual_2d.png](unified_reco/preliminary_studies/01_reconstruction/time_residual_2d.png) | positron time residuals |
| [time_spread.png](unified_reco/preliminary_studies/01_reconstruction/time_spread.png) | positron-group time spread (pileup guard) |
| [components_vs_truth.png](unified_reco/preliminary_studies/01_reconstruction/components_vs_truth.png), [components_vs_truth_osl.png](unified_reco/preliminary_studies/01_reconstruction/components_vs_truth_osl.png) | live/dead/LYSO energy component decomposition |
| [turnon.png](unified_reco/preliminary_studies/01_reconstruction/turnon.png), [acc_shape.png](unified_reco/preliminary_studies/01_reconstruction/acc_shape.png) | acceptance turn-on vs energy |

## 2. R_e/μ time-fit measurement → [`02_remu_timefit/`](unified_reco/preliminary_studies/02_remu_timefit/)
Per-bin prompt+delayed fit, sideband-anchored pileup, shift-to-pion-stop + analytic
extrapolation to t′=0. **Closure = 1.0021 (recovers input SM exactly).**

| Plot | Shows |
|---|---|
| [timefit_show.png](unified_reco/preliminary_studies/02_remu_timefit/timefit_show.png) | the corrected per-bin time fit |
| [template_explain.png](unified_reco/preliminary_studies/02_remu_timefit/template_explain.png) | prompt/delayed/pileup template construction |
| [lowbin_fit.png](unified_reco/preliminary_studies/02_remu_timefit/lowbin_fit.png), [remix_fit_K9.png](unified_reco/preliminary_studies/02_remu_timefit/remix_fit_K9.png) | low-energy-bin fit + event-mix closure |
| [time_components_10M.png](unified_reco/preliminary_studies/02_remu_timefit/time_components_10M.png), [time_components_excl.png](unified_reco/preliminary_studies/02_remu_timefit/time_components_excl.png) | time-spectrum component breakdown |
| [time_spectrum_argmax_role.png](unified_reco/preliminary_studies/02_remu_timefit/time_spectrum_argmax_role.png) | role of argmax time selection |
| [time_fits.png](unified_reco/preliminary_studies/02_remu_timefit/time_fits.png), [energy_spectrum.png](unified_reco/preliminary_studies/02_remu_timefit/energy_spectrum.png) | consensus time fits + energy spectrum |

## 3. Muon-merge gate fix → [`03_muon_merge_gatefix/`](unified_reco/preliminary_studies/03_muon_merge_gatefix/)
Early-decay michels merge the muon Bragg into the positron ATAR pixels → **+2.4%
decay-time-correlated fake-pie**. Fix: drop pixels with pion/muon prob > 0.05 from the
positron-energy gate. **Zero pie-efficiency cost.**

| Plot | Shows |
|---|---|
| [muon_merge_artifact.png](unified_reco/preliminary_studies/03_muon_merge_gatefix/muon_merge_artifact.png) | the merge artifact (muon energy leaking into pie gate) |
| [gatefix_closure.png](unified_reco/preliminary_studies/03_muon_merge_gatefix/gatefix_closure.png) | fake-pie removed after the gate fix |
| [muon_dist_distribution.png](unified_reco/preliminary_studies/03_muon_merge_gatefix/muon_dist_distribution.png) | muon-stop distance driving the merge |

## 4. muon-DIF veto → [`04_mudif_veto/`](unified_reco/preliminary_studies/04_mudif_veto/)
Orthogonal `MuonDIFVetoHead` rejecting boosted, prompt muon-DIF → positron.

| Plot | Shows |
|---|---|
| [mudif_rejection.png](unified_reco/preliminary_studies/04_mudif_veto/mudif_rejection.png), [mudif_suppression.png](unified_reco/preliminary_studies/04_mudif_veto/mudif_suppression.png) | rejection / suppression curves |
| [mudif_optimal_cut.png](unified_reco/preliminary_studies/04_mudif_veto/mudif_optimal_cut.png), [mudif_eff_vs_cut_accept.png](unified_reco/preliminary_studies/04_mudif_veto/mudif_eff_vs_cut_accept.png) | working-point / eff-vs-cut |
| [mudif_score_vs_nhits.png](unified_reco/preliminary_studies/04_mudif_veto/mudif_score_vs_nhits.png), [mudif_score_vs_nhits_log.png](unified_reco/preliminary_studies/04_mudif_veto/mudif_score_vs_nhits_log.png) | score vs hit multiplicity |
| [mudif_invisible.png](unified_reco/preliminary_studies/04_mudif_veto/mudif_invisible.png) | the irreducible ("invisible") remainder |
| [event_displays/](unified_reco/preliminary_studies/04_mudif_veto/event_displays/) (6), [survivor_displays/](unified_reco/preliminary_studies/04_mudif_veto/survivor_displays/) (18) | event displays: tagged + survivors |

## 5. pion-DIF veto → [`05_pidif_veto/`](unified_reco/preliminary_studies/05_pidif_veto/)
`PionDIFVetoHead`. Key result: **suppression is dominated by the truth acceptance cut**
(acceptance alone ≈ 16.7× on the low bin); veto-first suppression ~90–620×.

| Plot | Shows |
|---|---|
| [pidif_eff_vs_cut_accept.png](unified_reco/preliminary_studies/05_pidif_veto/pidif_eff_vs_cut_accept.png) | pie-eff vs pion-DIF cut, with acceptance |
| [mudif_vs_pidif_eff.png](unified_reco/preliminary_studies/05_pidif_veto/mudif_vs_pidif_eff.png) | muDIF vs piDIF efficiency comparison |
| [notvetoed_displays/](unified_reco/preliminary_studies/05_pidif_veto/notvetoed_displays/) (20) | displays of piDIF the model does NOT veto |

## 6. DIF tail-fraction bias (key new finding) → [`06_dif_tail_bias/`](unified_reco/preliminary_studies/06_dif_tail_bias/)
The DIF veto heads bias the pie tail. **muDIF bias = `atar_posE` Bhabha/δ-ray bookkeeping**
(positron δ/Bhabha electron energy dropped in the ATAR; Bhabha-corrected removes ~60% of the
bias at 50% eff, 0.972→0.989). **piDIF bias = spurious θ confound** (pie & piDIF have identical
θ distributions, yet the score tracks θ → fixable with DisCo θ-decorrelation).

| Plot | Shows |
|---|---|
| [tail_bias_mechanism.png](unified_reco/preliminary_studies/06_dif_tail_bias/tail_bias_mechanism.png) | the bias mechanism (energy bookkeeping) |
| [tail_bias_vs_theta.png](unified_reco/preliminary_studies/06_dif_tail_bias/tail_bias_vs_theta.png) | tail bias vs θ (the confound) |
| [pie_eff_vs_energy.png](unified_reco/preliminary_studies/06_dif_tail_bias/pie_eff_vs_energy.png), [pie_eff_vs_energy_pidif.png](unified_reco/preliminary_studies/06_dif_tail_bias/pie_eff_vs_energy_pidif.png) | pie efficiency vs deposited energy, muDIF & piDIF |

## 7. Tail-reveal / veto-model performance → [`07_tail_veto_model/`](unified_reco/preliminary_studies/07_tail_veto_model/)
Multi-head veto model; DisCo-decorrelated from energy and time; score fusion; full cutflow.

| Plot | Shows |
|---|---|
| [roc.png](unified_reco/preliminary_studies/07_tail_veto_model/roc.png), [roc_sweep.png](unified_reco/preliminary_studies/07_tail_veto_model/roc_sweep.png), [veto_roc.png](unified_reco/preliminary_studies/07_tail_veto_model/veto_roc.png) | per-head + veto ROC |
| [score_hists.png](unified_reco/preliminary_studies/07_tail_veto_model/score_hists.png), [tradeoff.png](unified_reco/preliminary_studies/07_tail_veto_model/tradeoff.png) | score distributions / eff-purity tradeoff |
| [flatness_sig_energy.png](unified_reco/preliminary_studies/07_tail_veto_model/flatness_sig_energy.png), [flatness_bkg_energy.png](unified_reco/preliminary_studies/07_tail_veto_model/flatness_bkg_energy.png), [flatness_sig_ptime.png](unified_reco/preliminary_studies/07_tail_veto_model/flatness_sig_ptime.png), [flatness_bkg_ptime.png](unified_reco/preliminary_studies/07_tail_veto_model/flatness_bkg_ptime.png) | DisCo flatness vs energy & positron-time |
| [fusion_roc.png](unified_reco/preliminary_studies/07_tail_veto_model/fusion_roc.png), [fusion_corr.png](unified_reco/preliminary_studies/07_tail_veto_model/fusion_corr.png), [fusion_scatter_pie_muon.png](unified_reco/preliminary_studies/07_tail_veto_model/fusion_scatter_pie_muon.png) | score fusion (pie × muon) |
| [efficiency_vs_energy.png](unified_reco/preliminary_studies/07_tail_veto_model/efficiency_vs_energy.png), [joint_efficiency_vs_energy.png](unified_reco/preliminary_studies/07_tail_veto_model/joint_efficiency_vs_energy.png), [box_eff_vs_energy.png](unified_reco/preliminary_studies/07_tail_veto_model/box_eff_vs_energy.png), [eff_heatmaps.png](unified_reco/preliminary_studies/07_tail_veto_model/eff_heatmaps.png) | efficiency vs energy (per head + joint) |
| [cutflow/](unified_reco/preliminary_studies/07_tail_veto_model/cutflow/) `cutflow_*.png` (10) | staged cutflow: acceptance → muon-veto → pileup-veto → topo → pie |

## 8. Diagnostics / forensics → [`08_diagnostics/`](unified_reco/preliminary_studies/08_diagnostics/)
| Plot | Shows |
|---|---|
| [lyso_forensics.png](unified_reco/preliminary_studies/08_diagnostics/lyso_forensics.png) | LYSO energy forensics |
| [micshape.png](unified_reco/preliminary_studies/08_diagnostics/micshape.png), [fullneg.png](unified_reco/preliminary_studies/08_diagnostics/fullneg.png) | michel spectrum shape / negative-energy tail |
| [normalize_bias_test.png](unified_reco/preliminary_studies/08_diagnostics/normalize_bias_test.png), [normalize_bias_fpfn.png](unified_reco/preliminary_studies/08_diagnostics/normalize_bias_fpfn.png) | energy-normalization bias (FP/FN) |
| [accidental_displays/](unified_reco/preliminary_studies/08_diagnostics/accidental_displays/) (12) | accidental-pileup tagging displays |

---

## What's running now → higher stats by Monday
- **Main-model R_e/μ (non-radiative):** 10.3M non-rad pie + ~20M in-window michel through `gatefix10M`
  ep14 → `analysis_main` (non-rad acceptance handled: pie normalized `br_e`, michel kMurad-filtered).
- **DIF suppression tables** (muDIF Bhabha-corrected, piDIF acceptance-decomposed) at full stats.
- **Tail `pie_score` michel-suppression curve** on the ~20M in-window michel.
