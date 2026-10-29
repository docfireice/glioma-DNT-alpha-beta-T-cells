# -*- coding: utf-8 -*-
"""Module 10: external validation on GSE131928 (Neftel et al., Cell 2019).
Two independent platforms deposited as processed TPM matrices:
  - Smart-seq2 (7930 cells, deep coverage)
  - 10x (~20k cells)
Goal: replicate the DNT / gamma-delta T gating strategy and the SPP1/MIF
macrophage-to-T communication axis in an independent lab cohort.
Outputs: fig19 (UMAPs), fig20 (cross-cohort composition + LR genes), csv tables.
"""
import os, gzip
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE131928')
DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'GSE131928')
FIGS = os.path.join(BASE, 'figs')
os.makedirs(FIGS, exist_ok=True)

GD_MARKERS = ['TRDC', 'TRDV1', 'TRDV2', 'TRGC1', 'TRDV3']
TAM_MARKERS = ['CD68', 'LYZ', 'CD14', 'C1QA', 'C1QB', 'C1QC', 'FCGR3A']
THR = 1.0  # TPM threshold


def load_tpm(path):
    print(f'  reading {os.path.basename(path)} ...', flush=True)
    df = pd.read_csv(path, sep='\t', index_col=0)
    print(f'  raw {df.shape} (genes x cells)', flush=True)
    return df


def analyse(df, plat):
    df = df[~df.index.duplicated()]
    ad = sc.AnnData(df.T.astype(np.float32))
    ad.obs_names = ad.obs_names.str.replace('.1$', '', regex=True)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    ad.layers['tpm'] = ad.X.copy()
    genes = set(ad.var_names)

    # ---- gating ----
    def expr(g):
        if g not in genes:
            return np.zeros(ad.n_obs)
        v = ad[:, g].X
        return v.toarray().ravel() if sp.issparse(v) else np.asarray(v).ravel()

    cd3 = expr('CD3D') + expr('CD3E')
    is_t = cd3 > THR
    gd = np.zeros(ad.n_obs, bool)
    for g in GD_MARKERS:
        if g in genes:
            gd |= expr(g) > THR
    cd4 = expr('CD4') > THR
    cd8 = expr('CD8A') > THR
    trac = expr('TRAC') > 0 if 'TRAC' in genes else np.ones(ad.n_obs, bool)
    gd = gd & is_t
    ab_dnt = is_t & (~gd) & (~cd4) & (~cd8) & trac
    conv = is_t & (~gd) & (cd4 | cd8)
    tam = np.zeros(ad.n_obs, bool)
    for g in TAM_MARKERS:
        if g in genes:
            tam |= expr(g) > THR
    tam = tam & (~is_t)

    gate = np.array(['other'] * ad.n_obs, dtype=object)
    gate[tam] = 'TAM'
    gate[conv] = 'conv_T'
    gate[gd] = 'gd_T'
    gate[ab_dnt] = 'ab_DNT'
    ad.obs['gate'] = gate

    # ---- LR axis expression ----
    for g in ['SPP1', 'MIF', 'CD44', 'CD74', 'CD274']:
        if g in genes:
            ad.obs[f'{g}_expr'] = expr(g)

    # ---- signature scores (module signature matrix top genes per class) ----
    sig = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE163108',
                                   'signature_matrix.csv'), index_col=0)
    for cls in ['ab_DNT', 'gd_T']:
        gl = [g for g in sig[cls].nlargest(50).index if g in genes]
        if gl:
            sc.tl.score_genes(ad, gl, score_name=f'{cls}_sig')

    # ---- quick embed ----
    sc.pp.highly_variable_genes(ad, n_top_genes=3000)
    sc.pp.pca(ad, n_comps=20, use_highly_variable=True)
    sc.pp.neighbors(ad, n_neighbors=15)
    sc.tl.umap(ad)
    print(f'  [{plat}] T cells: {is_t.sum()} | gd_T: {gd.sum()} | ab_DNT: {ab_dnt.sum()} '
          f'| conv_T: {conv.sum()} | TAM: {tam.sum()}', flush=True)
    # SPP1+ TAM fraction and receptor positivity in T cells
    spp1_tam = (ad.obs.loc[gate == 'TAM', 'SPP1_expr'] > THR).mean() if (gate == 'TAM').sum() > 0 else np.nan
    cd44_t = (ad.obs.loc[is_t, 'CD44_expr'] > THR).mean() if is_t.sum() else np.nan
    cd74_t = (ad.obs.loc[is_t, 'CD74_expr'] > THR).mean() if is_t.sum() else np.nan
    mif_tam = (ad.obs.loc[gate == 'TAM', 'MIF_expr'] > THR).mean() if (gate == 'TAM').sum() > 0 else np.nan
    print(f'  [{plat}] TAM SPP1+ {spp1_tam:.2f} MIF+ {mif_tam:.2f} | T-cell CD44+ {cd44_t:.2f} CD74+ {cd74_t:.2f}', flush=True)
    return ad, is_t


print('[1] Smart-seq2', flush=True)
df1 = load_tpm(os.path.join(DATA, 'GSM3828672_Smartseq2_GBM_IDHwt_processed_TPM.tsv.gz'))
ad_ss2, is_t_ss2 = analyse(df1, 'Smartseq2')
del df1

print('[2] 10x', flush=True)
df2 = load_tpm(os.path.join(DATA, 'GSM3828673_10X_GBM_IDHwt_processed_TPM.tsv.gz'))
ad_10x, is_t_10x = analyse(df2, '10x')
del df2

# ---------- fig19: UMAPs ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, ad, name in zip(axes, [ad_ss2, ad_10x], ['Smart-seq2', '10x']):
    order = ['conv_T', 'ab_DNT', 'gd_T', 'TAM', 'other']
    cmap = {'conv_T': '#4C72B0', 'ab_DNT': '#C44E52', 'gd_T': '#55A868',
            'TAM': '#DD8452', 'other': '#DDDDDD'}
    for g in order:
        m = ad.obs['gate'] == g
        if m.sum() == 0:
            continue
        ax.scatter(ad.obsm['X_umap'][m, 0], ad.obsm['X_umap'][m, 1], s=2,
                   c=cmap[g], label=f'{g} ({m.sum()})')
    ax.set_title(f'GSE131928 {name}'); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=3, fontsize=8, loc='best')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig19_gse131928_umaps.png'), dpi=150); plt.close()

# ---------- fig20: cross-cohort composition + LR ----------
summary = pd.DataFrame({
    'cohort': ['GSE163108', 'GSE182109', 'GSE131928-SS2', 'GSE131928-10x'],
    'platform': ['10x', '10x', 'Smart-seq2', '10x'],
    'T_cells': [24756 - 419, 14536 + 11308 + 1769, int(is_t_ss2.sum()), int(is_t_10x.sum())],
    'ab_DNT': [801, 11308, int((ad_ss2.obs.gate == 'ab_DNT').sum()), int((ad_10x.obs.gate == 'ab_DNT').sum())],
    'gd_T': [34, 1769, int((ad_ss2.obs.gate == 'gd_T').sum()), int((ad_10x.obs.gate == 'gd_T').sum())],
    'conv_T': [23921, 14536, int((ad_ss2.obs.gate == 'conv_T').sum()), int((ad_10x.obs.gate == 'conv_T').sum())],
})
summary['ab_DNT_pct'] = 100 * summary['ab_DNT'] / summary['T_cells']
summary['gd_T_pct'] = 100 * summary['gd_T'] / summary['T_cells']
summary.to_csv(os.path.join(BASE, 'cross_cohort_composition.csv'), index=False)
print(summary.to_string(index=False), flush=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
x = np.arange(len(summary)); w = 0.38
ax.bar(x - w / 2, summary['ab_DNT_pct'], w, color='#C44E52', label='ab_DNT')
ax.bar(x + w / 2, summary['gd_T_pct'], w, color='#55A868', label='gd_T')
ax.set_xticks(x); ax.set_xticklabels(summary['cohort'], rotation=20, ha='right', fontsize=8)
ax.set_ylabel('% of CD3+ T cells'); ax.legend(); ax.set_title('Cross-cohort composition')

ax = axes[1]
for ad, plat, c in [(ad_ss2, 'Smart-seq2', '#4C72B0'), (ad_10x, '10x', '#DD8452')]:
    m = ad.obs.gate.isin(['conv_T', 'ab_DNT', 'gd_T'])
    v1 = ad.obs.loc[ad.obs.gate == 'TAM', 'SPP1_expr'].values
    v2 = ad.obs.loc[m, 'CD44_expr'].values
    bp = ax.boxplot([v1, v2], positions=[0, 1], widths=.6, showfliers=False)
    for b in bp['boxes']:
        b.set_color(c)
ax.set_xticks([0, 1]); ax.set_xticklabels(['TAM SPP1', 'T cells CD44'])
ax.set_title('SPP1-CD44 axis (Neftel data)')

ax = axes[2]
for ad, plat, c in [(ad_ss2, 'Smart-seq2', '#4C72B0'), (ad_10x, '10x', '#DD8452')]:
    v1 = ad.obs.loc[ad.obs.gate == 'TAM', 'MIF_expr'].values
    v2 = ad.obs.loc[ad.obs.gate.isin(['conv_T', 'ab_DNT', 'gd_T']), 'CD74_expr'].values
    bp = ax.boxplot([v1, v2], positions=[0, 1], widths=.6, showfliers=False)
    for b in bp['boxes']:
        b.set_color(c)
ax.set_xticks([0, 1]); ax.set_xticklabels(['TAM MIF', 'T cells CD74'])
ax.set_title('MIF-CD74 axis (Neftel data)')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig20_cross_cohort_LR.png'), dpi=150); plt.close()

# per-cell tables
ad_ss2.obs[['gate', 'ab_DNT_sig', 'gd_T_sig', 'SPP1_expr', 'MIF_expr',
            'CD44_expr', 'CD74_expr']].to_csv(os.path.join(BASE, 'cell_level_smartseq2.csv'))
ad_10x.obs[['gate', 'ab_DNT_sig', 'gd_T_sig', 'SPP1_expr', 'MIF_expr',
            'CD44_expr', 'CD74_expr']].to_csv(os.path.join(BASE, 'cell_level_10x.csv'))
print('[done] module 10', flush=True)
