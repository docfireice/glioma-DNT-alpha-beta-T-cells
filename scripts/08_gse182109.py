# -*- coding: utf-8 -*-
"""Module 1b: process GSE182109 (Abdelfattah 2022, 44 samples, mtx format)
QC -> Harmony -> gating (raw + strict) -> count gamma-delta T and ab_DNT.
Heavy run: ~250k cells."""
import scanpy as sc, anndata as ad, numpy as np, pandas as pd, os, glob, re, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import mmread

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'GSE182109')
BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE182109')
os.makedirs(BASE, exist_ok=True)
FIGS = os.path.join(BASE, 'figs'); os.makedirs(FIGS, exist_ok=True)

bcds = sorted(glob.glob(os.path.join(DATA, '*_barcodes.tsv.gz')))
print(f'[load] {len(bcds)} samples')
t0 = time.time()
adatas = []
for f in bcds:
    sm = re.sub(r'_barcodes\.tsv\.gz$', '', os.path.basename(f))
    X = mmread(os.path.join(DATA, sm + '_matrix.mtx.gz')).T.tocsr()  # cells x genes
    feats = pd.read_csv(os.path.join(DATA, sm + '_features.tsv.gz'), sep='\t', header=None)
    bcs = pd.read_csv(os.path.join(DATA, sm + '_barcodes.tsv.gz'), sep='\t', header=None)
    var = pd.DataFrame(index=feats.iloc[:, 1].astype(str).values)
    obs = pd.DataFrame(index=[sm + '_' + b for b in bcs.iloc[:, 0].astype(str)])
    a = ad.AnnData(X.astype(np.float32), obs=obs, var=var)
    a.var_names_make_unique()  # consistent '-1','-2' suffixes so samples align on concat
    a.obs['sample'] = sm
    adatas.append(a)
    print(f'  {sm}: {a.shape}', flush=True)
adata = ad.concat(adatas, join='outer')
del adatas
print(f'[concat] {adata.shape} in {time.time()-t0:.0f}s', flush=True)

# QC
adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'mt-'))
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs['pct_counts_mt'] < 10, :]
adata = adata[(adata.obs['n_genes_by_counts'] >= 200) & (adata.obs['n_genes_by_counts'] <= 8000), :]
print(f'[qc] {adata.shape}', flush=True)

sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key='sample')
sc.pp.pca(adata, n_comps=30, use_highly_variable=True)
print('[pca] done', flush=True)
sc.external.pp.harmony_integrate(adata, key='sample')
print('[harmony] done', flush=True)
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca_harmony')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=1.0, key_added='leiden', flavor='igraph', n_iterations=2)
print('[cluster]', adata.obs['leiden'].nunique(), 'clusters', flush=True)

# gating: raw gene-count gate
X = adata.X
def counts_for(genes):
    idx = [adata.var_names.get_loc(g) for g in genes if g in adata.var_names]
    return np.asarray(X[:, idx].sum(1)).ravel()
cd3 = counts_for(['CD3D', 'CD3E']) > 0
cd4 = counts_for(['CD4'])
cd8a = counts_for(['CD8A']); cd8b = counts_for(['CD8B'])
trdc = counts_for(['TRDC', 'TRDC1'])
gate = np.array(['other'] * adata.n_obs, dtype=object)
gate[cd3 & (cd4 == 0) & (cd8a == 0) & (cd8b == 0) & (trdc == 0)] = 'ab_DNT'
gate[cd3 & (trdc > 0) & (cd4 == 0) & (cd8a == 0) & (cd8b == 0)] = 'gd_T'
gate[cd3 & ((cd4 > 0) | (cd8a > 0) | (cd8b > 0))] = 'conv_T'
gate[~cd3] = 'non_T'
adata.obs['gate_raw'] = pd.Categorical(gate)

# strict: cluster-level support (cluster where >50% cells are raw-DNT/gdT)
for g, name in [('ab_DNT', 'strict_ab'), ('gd_T', 'strict_gd')]:
    mask = (gate == g)
    cf = pd.Series(mask).groupby(adata.obs['leiden'].values.astype(str)).transform('mean').values
    adata.obs[name] = mask & (cf > 0.5)
strict_dnt = adata.obs['strict_ab'].values; strict_gd = adata.obs['strict_gd'].values
gate2 = gate.copy()
gate2[strict_dnt & (gate != 'ab_DNT')] = 'ab_DNT'
gate2[strict_gd & (gate != 'gd_T')] = 'gd_T'
gate2[(gate == 'ab_DNT') & ~strict_dnt] = 'conv_T'
gate2[(gate == 'gd_T') & ~strict_gd] = 'conv_T'
adata.obs['gate'] = pd.Categorical(gate2)

cd3n = (adata.obs['gate'] != 'non_T').sum()
cnt = adata.obs['gate'].value_counts()
print('[gate STRICT]', dict(cnt), flush=True)
print(f'[gate STRICT] ab_DNT frac of CD3+ = {cnt.get("ab_DNT",0)/cd3n*100:.2f}% | gd_T = {cnt.get("gd_T",0)/cd3n*100:.2f}%', flush=True)

cnt.to_frame('n').to_csv(os.path.join(BASE, 'gate_counts.csv'))
adata.obs.to_csv(os.path.join(BASE, 'cell_metadata.csv'))
adata.write_h5ad(os.path.join(BASE, 'gse182109_processed.h5ad'), compression='gzip')

# UMAP figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, title in [(axes[0], 'gate', 'gate (strict)'), (axes[1], 'sample', 'sample')]:
    cats = adata.obs[col].astype(str)
    if col == 'gate':
        cols = {'conv_T': '#BBBBBB', 'ab_DNT': '#D65F5F', 'gd_T': '#2CA02C', 'non_T': '#EEEEEE'}
        order = [c for c in ['non_T', 'conv_T', 'ab_DNT', 'gd_T'] if c in set(cats)]
        for c in order:
            m = (cats == c).values
            ax.scatter(adata.obsm['X_umap'][m, 0], adata.obsm['X_umap'][m, 1], s=0.5, c=cols[c], label=f'{c} ({m.sum()})')
        ax.legend(markerscale=8, fontsize=8)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig11_gse182109_umap.png'), dpi=150); plt.close()
print('[done] module 1b', flush=True)
