# -*- coding: utf-8 -*-
"""Build the 7 submission main figures (JITC: max 7 figures+tables). v2
Equal-height row layout; landscape panels where full width is needed."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
TMP = 'submission/02_figures/_panels'
OUT = 'submission/02_figures'
os.makedirs(TMP, exist_ok=True)

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False})
C = {'ab_DNT': '#c0392b', 'conv_T': '#2471a3', 'gd_T': '#1e8449'}

# ---------------------------------------------------------------- panels
def p_fig1f():
    d = pd.read_csv('results/GSE182109/gate_v2_counts.csv')
    d = d[d.gate_v2 != 'non_T'].copy()
    tot = d.n.sum()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.bar(d.gate_v2, d.n, color=[C.get(g, '#7f8c8d') for g in d.gate_v2], width=.62)
    for x, n in enumerate(d.n):
        ax.text(x, n * 1.15, f'{n:,}\n({n / tot * 100:.1f}%)', ha='center', fontsize=7.5)
    ax.set_yscale('log'); ax.set_ylim(500, d.n.max() * 12)
    ax.set_ylabel('T cells (log scale)')
    ax.set_title('GSE182109, stringent v2 gate', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{TMP}/fig1f.png', dpi=300); plt.close(fig)

def p_fig1g():
    d = pd.read_csv('results/GSE182109/gate_v2_by_histology.csv')
    labels = {'LGG': 'LGG', 'ndGBM': 'newly dx GBM', 'rGBM': 'recurrent GBM'}
    gates = ['ab_DNT', 'conv_T', 'gd_T']
    x = np.arange(len(d)); w = .26
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for i, g in enumerate(gates):
        ax.bar(x + (i - 1) * w, d[g], w, label=g, color=C[g])
    ax.set_xticks(x); ax.set_xticklabels([labels[h] for h in d.histology], fontsize=8)
    ax.set_ylabel('% of all cells')
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title('Gate fraction by histology', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{TMP}/fig1g.png', dpi=300); plt.close(fig)

def p_fig2a():
    d = pd.read_csv('results/GSE163108/tcr_dnt_sharing.csv')
    n = len(d)
    priv = int((~d.in_conv_T & ~d.in_gdT).sum())
    sc = int((d.in_conv_T & ~d.in_gdT).sum())
    sg = int((~d.in_conv_T & d.in_gdT).sum())
    sb = int((d.in_conv_T & d.in_gdT).sum())
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    vals = [priv, sc, sg, sb]
    labs = ['private\n(neither)', 'shared w/\nconv T', 'shared w/\nγδ T', 'shared w/\nboth']
    bars = ax.bar(labs, vals, color=['#c0392b', '#2471a3', '#1e8449', '#7d3c98'], width=.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + n * .012,
                f'{v}\n({v / n * 100:.1f}%)', ha='center', fontsize=7.5)
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_ylabel(f'DN αβ clonotypes (n = {n})')
    fig.tight_layout(); fig.savefig(f'{TMP}/fig2a.png', dpi=300); plt.close(fig)

def p_fig2d():
    d = pd.read_csv('results/GSE163108/tcr_expansion_summary.csv')
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.bar(d.gate, d.pct_cells_in_expanded_clones,
           color=[C.get(g, '#7f8c8d') for g in d.gate], width=.6)
    for x, v in enumerate(d.pct_cells_in_expanded_clones):
        ax.text(x, v + 1.2, f'{v:.1f}%', ha='center', fontsize=8)
    ax.set_ylabel('% cells in expanded clones (≥2)')
    ax.set_ylim(0, 52)
    fig.tight_layout(); fig.savefig(f'{TMP}/fig2d.png', dpi=300); plt.close(fig)

def p_fig3e():
    d = pd.read_csv('results/GSE163108/diffpot_cell_level.csv', index_col=0)
    d = d.dropna(subset=['cytotrace', 'dpt'])
    from scipy import stats
    rho, p = stats.spearmanr(d.cytotrace, d.dpt)
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    for g in ['conv_T', 'gd_T', 'ab_DNT']:
        s = d[d.gate == g]
        ax.scatter(s.dpt, s.cytotrace, s=3, alpha=.35, color=C[g], label=g, linewidths=0)
    ax.set_xlabel('diffusion pseudotime'); ax.set_ylabel('CytoTRACE potential')
    ax.legend(frameon=False, fontsize=7.5, markerscale=3, loc='upper right')
    ax.set_title(f'Spearman ρ = {rho:.2f}', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{TMP}/fig3e.png', dpi=300); plt.close(fig)

def p_fig4a():
    d = pd.read_csv('results/GSE163108/gate_function_scores.csv', index_col=0)
    mods = ['cytotoxicity', 'IFN_gamma', 'exhaustion', 'naive_memory', 'tissue_residency']
    nice = ['cytotox.', 'IFN-γ', 'exhaustion', 'naive/\nmemory', 'tissue\nresidency']
    gates = ['ab_DNT', 'conv_T', 'gd_T']
    x = np.arange(len(mods)); w = .26
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    for i, g in enumerate(gates):
        ax.bar(x + (i - 1) * w, d.loc[g, mods], w, label=g, color=C[g])
    ax.axhline(0, color='k', lw=.6)
    ax.set_xticks(x); ax.set_xticklabels(nice, fontsize=8)
    ax.set_ylabel('mean module score (z)')
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc='upper right')
    fig.tight_layout(); fig.savefig(f'{TMP}/fig4a.png', dpi=300); plt.close(fig)

def p_fig4c():
    d = pd.read_csv('results/GSE163108/regulon_DNT_vs_conv.csv')
    d = d.reindex(d.delta.abs().sort_values().index).tail(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    ax.barh(d.TF, d.delta, color='#c0392b', height=.62)
    for y, (v, p) in enumerate(zip(d.delta, d.padj)):
        ax.text(v - .012, y, f'FDR={p:.0e}', va='center', ha='right', fontsize=6.5)
    ax.axvline(0, color='k', lw=.6)
    ax.set_xlabel('regulon activity, DN αβ T cells − conventional T cells')
    ax.set_xlim(d.delta.min() * 1.4, .02)
    fig.tight_layout(); fig.savefig(f'{TMP}/fig4c.png', dpi=300); plt.close(fig)

def p_fig5d():
    d = pd.read_csv('results/GSE163108/visium_colocalization.csv').sort_values('mean_rho')
    lab = [p.replace('|', ' vs ').replace('ab_DNT', 'DN αβ').replace('gd_T', 'γδ')
            .replace('conv_T', 'conv T').replace('Macrophage', 'macrophage')
            .replace('Tumor', 'tumor') for p in d.pair]
    fig, ax = plt.subplots(figsize=(6.9, 2.4))
    cols = ['#c0392b' if 'macrophage' in l and ('DN' in l or 'γδ' in l) else '#95a5a6' for l in lab]
    ax.barh(lab, d.mean_rho, color=cols, height=.6)
    for y, (r, np_, ns) in enumerate(zip(d.mean_rho, d.n_pos, d.n_samples)):
        ax.text(r + .015, y, f'{np_}/{ns} sections positive', va='center', fontsize=7)
    ax.axvline(0, color='k', lw=.6); ax.set_xlim(-.05, 1.32)
    ax.set_xlabel('mean per-section Spearman ρ between spot-level signature scores (Visium, 16 sections)')
    fig.tight_layout(); fig.savefig(f'{TMP}/fig5d.png', dpi=300); plt.close(fig)

for fn in [p_fig1f, p_fig1g, p_fig2a, p_fig2d, p_fig3e, p_fig4a, p_fig4c, p_fig5d]:
    fn()
print('panels drawn')

# ---------------------------------------------------------------- compose
F163, F182, FCX = 'results/GSE163108/figs', 'results/GSE182109/figs', 'results/CELLxGENE/figs'

def trim(im, tol=14):
    bg = Image.new('RGB', im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg).convert('L').point(lambda p: 255 if p > tol else 0)
    bb = diff.getbbox()
    return im.crop(bb) if bb else im

FONT = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 42)

def compose(name, rows, width_in=7.2, dpi=300):
    W = int(width_in * dpi)
    pad = int(.02 * dpi); gap = int(.04 * dpi); head = int(.17 * dpi)
    rows = [[(l, trim(Image.open(p).convert('RGB'))) for l, p in row] for row in rows]
    H, y = pad, pad
    layout = []
    for row in rows:
        aspects = [im.width / im.height for _, im in row]
        avail = W - pad * 2 - gap * (len(row) - 1)
        h = int(avail / sum(aspects))
        layout.append((row, h))
        H += h + head + gap
    canvas = Image.new('RGB', (W, H + pad), 'white')
    for row, h in layout:
        x = pad
        for l, im in row:
            w = int(h * im.width / im.height)
            im2 = im.resize((w, h), Image.LANCZOS)
            canvas.paste(im2, (x, y + head))
            ImageDraw.Draw(canvas).text((x + 2, y + 2), l, fill='black', font=FONT)
            x += w + gap
        y += h + head + gap
    canvas = canvas.crop((0, 0, W, y))
    canvas.save(f'{OUT}/{name}.png', dpi=(dpi, dpi))
    canvas.save(f'{OUT}/{name}.tif', dpi=(dpi, dpi), compression='tiff_lzw')
    print(f'{name}: {canvas.size[0] / dpi:.1f} x {canvas.size[1] / dpi:.1f} in')

compose('Figure1', [
    [('a', f'{F163}/fig1_umap_subset.png')],
    [('b', f'{F163}/fig2_dotplot_markers.png'), ('c', f'{F163}/fig3_signature_heatmap.png')],
    [('d', f'{F163}/fig4_sample_composition.png'), ('e', f'{F182}/fig11_gse182109_umap_v2.png')],
    [('f', f'{TMP}/fig1f.png'), ('g', f'{TMP}/fig1g.png')],
])
compose('Figure2', [
    [('a', f'{TMP}/fig2a.png'), ('b', f'{F163}/fig7_tcr_sharing.png')],
    [('c', f'{F163}/fig5_tcr_clone_size.png')],
    [('d', f'{TMP}/fig2d.png'), ('e', f'{F163}/fig6_tcr_sig_vs_clonesize.png')],
])

def split_fig17():
    im = Image.open(f'{F163}/fig17_gse16011_infiltration.png').convert('RGB')
    w, h = im.size
    outs = []
    for i in range(3):
        part = im.crop((int(w * i / 3), 0, int(w * (i + 1) / 3), h))
        outs.append(part)
    for i, part in enumerate(outs):
        part.save(f'{TMP}/fig17_p{i+1}.png', dpi=(300, 300))
    return outs

compose('Figure3', [
    [('a', f'{F163}/fig9_umap_dpt.png'), ('b', f'{F163}/fig8_pseudotime_profiles.png')],
    [('c', f'{F163}/fig18b_umap_diffpot.png')],
    [('d', f'{TMP}/fig3e.png'), ('e', f'{F182}/fig21_gse182109_diffpot.png')],
])
compose('Figure4', [
    [('a', f'{TMP}/fig4a.png'), ('b', f'{F163}/fig15_regulon_heatmap.png')],
    [('c', f'{TMP}/fig4c.png')],
])
compose('Figure5', [
    [('a', f'{F182}/fig13_LR_heatmap.png'), ('b', f'{F182}/fig14_LR_top_pairs.png')],
    [('c', f'{TMP}/fig5d.png')],
    [('d', f'{F163}/fig16_visium_maps.png')],
])
compose('Figure6', [
    [('a', f'{FCX}/fig19_cellxgene_umaps.png'), ('b', f'{FCX}/fig19c_atlas_fractions.png')],
    [('c', f'{FCX}/fig20_cellxgene_LR.png')],
    [('d', f'{FCX}/fig20c_atlas_LR.png'), ('e', f'{FCX}/fig22_signature_transfer.png')],
])
split_fig17()
compose('Figure7', [
    [('a', f'{F163}/fig9_clinical_clean.png')],
    [('f', f'{TMP}/fig17_p1.png'), ('g', f'{TMP}/fig17_p2.png'), ('h', f'{TMP}/fig17_p3.png')],
])
print('all figures composed')
