# -*- coding: utf-8 -*-
"""Module 3: trajectory - diffusion pseudotime on CD3+ T cells (conv_T + ab_DNT + gd_T),
root = naive_memory-high non-exhausted cells; validate conv_T -> DNT exhaustion direction."""
import scanpy as sc, numpy as np, pandas as pd, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE163108')
FIGS = os.path.join(BASE, 'figs'); os.makedirs(FIGS, exist_ok=True)
adata = sc.read_h5ad(os.path.join(BASE, 'gse163108_processed.h5ad'))
print('[load]', adata.shape)

# CD3+ only (drop non_T), keep raw gate for pseudotime colors
sub = adata[~adata.obs['gate'].isin(['non_T'])].copy()
sc.pp.highly_variable_genes(sub, n_top_genes=2000, batch_key='sample')
sc.pp.pca(sub, n_comps=30, use_highly_variable=True)
sc.pp.neighbors(sub, n_neighbors=15, use_rep='X_pca')
sc.tl.diffmap(sub, n_comps=15)

# root: cells with high naive_memory + low exhaustion + gate conv_T
nm = sub.obs['naive_memory'].values; ex = sub.obs['exhaustion'].values
cand = np.where((sub.obs['gate'] == 'conv_T').values)[0]
best = cand[np.argmax(nm[cand] - ex[cand])]
sub.uns['iroot'] = int(best)
sc.tl.dpt(sub, n_dcs=10)
sub.obs['dpt'] = sub.obs['dpt_pseudotime']

# mean signature + gate composition along pseudotime (binned)
bins = pd.qcut(sub.obs['dpt'], 20, labels=False)
sub.obs['pt_bin'] = bins.values
prof = sub.obs.groupby('pt_bin', observed=True)[['exhaustion', 'cytotoxicity', 'IFN_gamma',
     'naive_memory', 'tissue_residency', 'proliferation']].mean()
gate_frac = pd.crosstab(sub.obs['pt_bin'], sub.obs['gate'], normalize='index')

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
prof.plot(ax=axes[0], colormap='coolwarm', lw=2)
axes[0].set_title('Signature scores along pseudotime'); axes[0].set_ylabel('mean score')
gate_frac.plot.area(ax=axes[1], colormap='tab20', alpha=0.85)
axes[1].set_title('Gate composition along pseudotime'); axes[1].set_ylabel('fraction'); axes[1].set_xlabel('pseudotime bin (0->1, naive root)')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig8_pseudotime_profiles.png'), dpi=150); plt.close()

# UMAP colored by DPT
fig, ax = plt.subplots(figsize=(6, 5))
scv = ax.scatter(sub.obsm['X_umap'][:, 0], sub.obsm['X_umap'][:, 1], c=sub.obs['dpt'], s=1, cmap='viridis')
plt.colorbar(scv, label='DPT'); ax.set_title('Diffusion pseudotime (CD3+ T cells)'); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig9_umap_dpt.png'), dpi=150); plt.close()

# DNT cells: DPT distribution vs conv_T
dnt_dpt = sub.obs.loc[sub.obs['gate'] == 'ab_DNT', 'dpt']
conv_dpt = sub.obs.loc[sub.obs['gate'] == 'conv_T', 'dpt']
from scipy import stats
u, p = stats.mannwhitneyu(dnt_dpt, conv_dpt)
print(f'[dpt] ab_DNT median={dnt_dpt.median():.3f} | conv_T median={conv_dpt.median():.3f} | MWU p={p:.2e}')
sub.obs[['dpt', 'pt_bin']].to_csv(os.path.join(BASE, 'dpt_cell_level.csv'))
print('[done] module 3')
