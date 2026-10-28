#!/usr/bin/env python
"""Derive the AI_final tail figure scripts from the working ones.

Four changes, all mechanical, so that the physics code stays in one place:
  1. predictions come from the train-mode dropout evaluations rather than the eval-mode ones;
  2. colours match the AI_final convention, pi->e blue and pi->mu->e red;
  3. figures are sized for the printed page instead of for a screen, so the fonts survive
     being scaled into a \\textwidth box;
  4. every figure is written as PDF as well as PNG, into figures/AI_final.

Run it, then run each generated script on a compute node. Never on the login node: they read
tens of GB of parquet.
"""
import os
import re

U = "/home/obeesley/pioneerML/unified_reco"
S = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"

SOURCES = [
    (f"{U}/tail_thesis_figures.py", f"{U}/tail_final_figs.py"),
    (f"{U}/tail_mudif_cutflow.py", f"{U}/tail_final_mudif.py"),
    (f"{S}/tail_reco_cutflows_ep19.py", f"{U}/tail_final_reco.py"),
]

HELPER = '''

def _save(fig, name):
    """Write both formats; the PDF is what the thesis includes."""
    for _ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{_ext}")
    print(f"wrote {OUT}/{name}.pdf", flush=True)

'''


def shrink(m):
    w, h = float(m.group(1)), float(m.group(2))
    k = 0.56 if w > 9.0 else (0.72 if w > 7.0 else 0.86)
    return f"figsize=({w * k:.2f}, {h * k:.2f})"


def convert(src, dst):
    s = open(src).read()
    log = {}

    def sub(pat, rep, key, literal=False, count=0):
        nonlocal s
        if literal:
            c = s.count(pat)
            s = s.replace(pat, rep)
        else:
            s, c = re.subn(pat, rep, s, count=count)
        log[key] = c

    # 1. inputs: train-mode dropout evaluations
    sub("michel_eval_inwin_rs", "michel_eval_inwin_drop", "michel dir", literal=True)
    sub("norad_eval2_rs", "norad_eval2_drop", "pie dir", literal=True)
    sub("predictions_evalmode_bak.parquet", "predictions_drop.parquet",
        "central cache", literal=True)

    # 2. colours: AI_final convention
    sub(r'"#d62728"', '"#0072B2"', "C_PIE")
    sub(r'"#1f77b4"', '"#D55E00"', "C_MIC")
    sub(r'"#2ca02c"', '"#009E73"', "C_PID")
    sub(r'"#9467bd"', '"#CC79A7"', "C_MUD")

    # 3. print sizing
    sub(r"tail_figstyle\.apply\(\)", "tail_figstyle.apply(scale=0.86)", "style scale")
    sub(r"figsize=\(([\d.]+), ([\d.]+)\)", shrink, "figsizes")

    # 4. output: PDF + PNG into AI_final
    sub(r'OUT = "/home/obeesley/thesis_ref/figures/AI"',
        'OUT = "/home/obeesley/thesis_ref/figures/AI_final"', "outdir")
    if "def _save(" not in s:
        anchor = re.search(r"^(OUT = .*AI_final.*)$", s, flags=re.M)
        if anchor:
            s = s[:anchor.end()] + "\n" + HELPER.rstrip() + s[anchor.end():]
            log["save helper"] = 1
    sub(r'fig\.savefig\(f"\{OUT\}/([A-Za-z0-9_]+)\.png"\)', r'_save(fig, "\1")', "savefig f-string")
    sub(r'fig\.savefig\(out\)', r'_save(fig, os.path.basename(str(out)).replace(".png", ""))',
        "savefig var")

    # the stats file may not exist yet in a fresh output directory
    sub(r'allstats = json\.load\(open\(f"\{OUT\}/tail_stats\.json"\)\)',
        'allstats = json.load(open(f"{OUT}/tail_stats.json")) if os.path.exists(f"{OUT}/tail_stats.json") else {}',
        "stats guard")

    open(dst, "w").write(s)
    print(f"{os.path.basename(src)} -> {os.path.basename(dst)}")
    for k, v in log.items():
        if v:
            print(f"    {k:18s} {v}")
    missed = [k for k, v in log.items() if not v and k in ("outdir",)]
    if missed:
        print(f"    WARNING: no match for {missed}")


for src, dst in SOURCES:
    convert(src, dst)
