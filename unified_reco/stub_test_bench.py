IOU_MIN = 0.95

fig, ax = plt.subplots(figsize=(7, 4.5))
for tag, truth, preds, color in [
    ('Michel', pimu_truth, pimu_preds, 'tab:blue'),
    ('Pienu',  pie_truth,  pie_preds,  'tab:red'),
]:
    htp_gate = (truth['htp'] == 1) & (preds['htp'] != SENTINEL) & (preds['htp'] > 0.5)
    iou_ok   = preds['pos_iou'] >= IOU_MIN
    gate     = htp_gate & iou_ok
    n_pre  = int(htp_gate.sum()); n_post = int(gate.sum())
    c, rms, err = angle_rms_curve_cos(truth, preds, bins_cos, gate)
    ax.errorbar(c, rms, yerr=err, fmt='o-', color=color, capsize=3,
                label=f'{tag}  (n={n_post})')
ax.set_xlabel(r'$\cos(\theta_{\mathrm{True}})$')
ax.set_ylabel(r'RMS($\theta_{\mathrm{True}} - \theta_{\mathrm{Reco}}$) [deg]')
ax.set_title(f'PURITY Positron angle RMS')
ax.set_ylim(2.0, 14.1)
ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout(); watermark(); plt.show()


def plot_role_confusion(slices, tag, include_anchor=False):
    mask = np.ones_like(slices['role_truth'], dtype=bool)
    if not include_anchor:
        mask &= ~slices['is_anchor']
    rt = slices['role_truth'][mask]
    rp = slices['role_pred'][mask]
    cm = np.zeros((3, 3), dtype=np.int64)  # [truth, pred]
    for t, p in zip(rt, rp):
        if 0 <= t < 3 and 0 <= p < 3:
            cm[t, p] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    names = ['Background', 'μ', 'e']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for a, m, ttl, fmt in [(axes[0], cm, 'Counts', 'd'),
                           (axes[1], cm_norm, 'Row-normalized', '.4f')]:
        im = a.imshow(m, cmap='Blues')
        a.set_xticks(range(3)); a.set_yticks(range(3))
        a.set_xticklabels(names); a.set_yticklabels(names)
        a.set_xlabel('Predicted'); a.set_ylabel('Truth')
        a.set_title(f'{tag}: {ttl}')
        for i in range(3):
            for j in range(3):
                a.text(j, i, format(m[i,j], fmt), ha='center', va='center',
                       color='white' if m[i,j] > m.max()*0.5 else 'black')
        fig.colorbar(im, ax=a)
    fig.tight_layout(); watermark(); plt.show()

plot_role_confusion(pie_slices,  'pie')
plot_role_confusion(pimu_slices, 'pimu')


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

def _filter_energy(truth, preds):
    v = (preds['positron_energy'] != SENTINEL) & np.isfinite(truth['positron_energy'])
    v &= (preds['dead_energy'] != SENTINEL)
    #v &= (truth['htp'] == 1) & (preds['htp'] != SENTINEL) & (preds['htp'] > 0.5) & (preds['accepted'] >= 0.5)
    v &=  (preds['htp'] != SENTINEL) & (preds['htp'] > 0.5) & (preds['accepted'] >= 0.5)
    total_E_reco = preds['positron_energy'][v] #+ preds['dead_energy'][v]   # add the recovered dead-material loss
    return truth['positron_energy'][v], total_E_reco

te_pie, pe_pie = _filter_energy(pie_truth,  pie_preds)
te_pim, pe_pim = _filter_energy(pimu_truth, pimu_preds)

COLOR_PIE  = 'red'
COLOR_PIMU = 'blue'

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)  # independent y
bins = np.linspace(0, 20, 50)

for ax, te, pe, tag, color in [
    (axes[0], te_pie, pe_pie, rf'$\pi \to e\ \ (N={len(te_pie):,})$',         COLOR_PIE),
    (axes[1], te_pim, pe_pim, rf'$\pi \to \mu \to e\ \ (N={len(te_pim):,})$', COLOR_PIMU),
]:
    # Truth: filled translucent histogram
    #ax.hist(te, bins=bins, histtype='stepfilled', color=color, alpha=0.30,
    #        edgecolor=color, linewidth=1.0, label='truth')
    # Reco: solid bold outline in black
    ax.hist(pe, bins=bins, histtype='step', color='black', linewidth=1.8, label='reco')
    ax.set_xlabel(r'$E_{e^+}$  [MeV]')
    ax.set_ylabel('counts')
    ax.set_title(tag)
    ax.legend(loc='upper right')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    #ax.set_yscale('log')
    #ax.set_xlim(68, 73)

fig.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

def acceptance_confusion(truth, preds):
    t = truth['acceptance'] == 1
    p = (preds['accepted'] != SENTINEL) & (preds['accepted'] >= 0.5)
    tp = int(( t &  p).sum())
    fn = int(( t & ~p).sum())
    fp = int((~t &  p).sum())
    tn = int((~t & ~p).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)  # rows: truth, cols: pred

cm_pie  = acceptance_confusion(pie_truth,  pie_preds)
cm_pimu = acceptance_confusion(pimu_truth, pimu_preds)

def metrics(cm):
    tn, fp = cm[0]; fn, tp = cm[1]
    eff  = tp / max(tp + fn, 1)
    fake = fp / max(fp + tn, 1)
    pur  = tp / max(tp + fp, 1)
    return eff, fake, pur

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
for ax, cm, title, color in [
    (axes[0], cm_pie,  r'$\pi \to e$',        'Reds'),
    (axes[1], cm_pimu, r'$\pi \to \mu \to e$', 'Blues'),
]:
    im = ax.imshow(cm, cmap=color, aspect='equal')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['reject', 'accept'])
    ax.set_yticklabels(['reject', 'accept'])
    ax.set_xlabel('predicted')
    ax.set_ylabel('truth')
    vmax = cm.max()
    # Annotate counts + row-normalized fractions
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sum
    for i in range(2):
        for j in range(2):
            txt_color = 'white' if cm[i, j] > 0.6 * vmax else 'black'
            ax.text(j, i, f'{cm[i, j]:,}',
                    ha='center', va='center', color=txt_color, fontsize=14)

    eff, fake, pur = metrics(cm)
    ax.set_title(f'{title}\n'
                 rf'$\epsilon={eff:.3f}\ \ \ f={fake:.4f}\ \ \ P={pur:.3f}$',
                 fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()
plt.show()

# Print raw counts and derived metrics for the paper
for tag, cm in [('pie', cm_pie), ('pimu', cm_pimu)]:
    tn, fp = cm[0]; fn, tp = cm[1]
    eff, fake, pur = metrics(cm)
    print(f'{tag:5s}: TN={tn:6d}  FP={fp:5d}  FN={fn:5d}  TP={tp:6d}  '
          f'eff={eff:.4f}  fake={fake:.5f}  purity={pur:.4f}  '
          f'surplus={fp-fn:+d}')



