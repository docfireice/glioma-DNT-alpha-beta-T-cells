# -*- coding: utf-8 -*-
"""GSE163108 (Mathewson 2021, 5 GBM, CD3+ T cells with TCR) full analysis.
Step 1: load raw 10X h5 + cell whitelist from metadata -> filter background barcodes
Step 2: QC (Hao 2026 thresholds) -> normalize -> HVG -> PCA -> Harmony/BBKNN -> Leiden
Step 3: CD3+ gating -> DNT (CD4-CD8A-CD8B-) / gamma-delta (TRDC/TRDV) classification
Step 4: DNT/gdT subclustering, markers, TCR clonal metrics
Usage: python 03_gse163108_full.py
"""
import os, glob, sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

sc.settings.verbosity = 1
RAW = 'data/raw'
OUT = 'results/GSE163108'
os.makedirs(OUT, exist_ok=True)

# ---------- Step 1: load ----------
meta = pd.read_csv(os.path.join(RAW, 'metadata_10x.csv.gz'))
meta = meta.rename(columns={'Unnamed: 0': 'barcode'})
meta['sample'] = meta['sampleid'].str.replace('_GEX', '', regex=False)
meta['bc'] = meta['barcode'].str.split('_').str[-1] + '-1'  # 10X h5 barcodes carry -1 suffix
print(f'[meta] annotated T cells = {len(meta)}; samples = {sorted(meta["sample"].unique())}')
print(meta['annotate_Tcelltype'].value_counts().to_string())

adatas = []
for f in sorted(glob.glob(os.path.join(RAW, 'GSM*_raw_feature_bc_matrix.h5'))):
    s = os.path.basename(f).split('_')[1]
    a = sc.read_10x_h5(f)
    a.var_names_make_unique()
    sub = meta[meta['sample'] == s]
    keep = a.obs_names.isin(set(sub['bc']))
    a = a[keep].copy()
    a.obs['sample'] = s
    a.obs = a.obs.merge(sub.set_index('bc')[['annotate_Tcelltype', 'clonotype_id',
                                             'clonotype_size', 'annotate_clonotype']],
                        left_index=True, right_index=True, how='left')
    adatas.append(a)
    print(f'[load] {s}: {a.n_obs} cells matched')

adata = ad.concat(adatas, join='outer')
adata.var_names_make_unique()
print(f'[concat] total cells = {adata.n_obs}, genes = {adata.n_vars}')

# ---------- Step 2: QC ----------
adata.var['mt'] = adata.var_names.str.startswith('MT-')
adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], inplace=True, percent_top=None)
qc = pd.DataFrame({'nGene': adata.obs.n_genes_by_counts, 'nUMI': adata.obs.total_counts,
                   'pct_mt': adata.obs.pct_counts_mt})
qc.describe().to_csv(os.path.join(OUT, 'qc_summary_before.csv'))
adata = adata[adata.obs.pct_counts_mt < 10, :].copy()
adata = adata[adata.obs.n_genes_by_counts.between(200, 4000), :].copy()
sc.pp.filter_genes(adata, min_cells=3)
print(f'[qc] after filtering: {adata.n_obs} cells')

adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='sample')
print(f'[hvg] {adata.var.highly_variable.sum()} HVGs')

# ---------- Step 3: PCA / integration / clustering ----------
sc.tl.pca(adata, n_comps=30)
try:
    sc.external.pp.harmony_integrate(adata, 'sample')
    rep = 'X_pca_harmony'
    print('[integrate] Harmony OK')
except Exception as e:
    print('[integrate] Harmony failed -> uncorrected PCA:', e)
    rep = 'X_pca'
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, use_rep=rep)
sc.tl.leiden(adata, resolution=1.0, key_added='leiden', flavor='igraph', n_iterations=2)
sc.tl.umap(adata)
print(f'[cluster] {adata.obs.leiden.nunique()} clusters')

# ---------- Step 4: DNT / gamma-delta gating ----------
def has(g): return g in adata.var_names
def vec(g):
    return np.asarray(adata[:, g].X.todense()).ravel() if has(g) else np.zeros(adata.n_obs)

cd3 = np.zeros(adata.n_obs, bool)
for g in ['CD3D', 'CD3E', 'CD3G']:
    if has(g): cd3 |= vec(g) > 0
cd4, cd8a, cd8b = vec('CD4'), vec('CD8A'), vec('CD8B')
gd_tr = np.zeros(adata.n_obs, bool)
for g in ['TRDC', 'TRDV1', 'TRDV2', 'TRGC1', 'TRGC2']:
    if has(g): gd_tr |= vec(g) > 0

gate = np.full(adata.n_obs, 'non_T', dtype=object)
gate[cd3 & ~((cd4 == 0) & (cd8a == 0) & (cd8b == 0))] = 'conv_T'
dnt_mask = cd3 & (cd4 == 0) & (cd8a == 0) & (cd8b == 0)
gate[dnt_mask & ~gd_tr] = 'ab_DNT'
gate[dnt_mask & gd_tr] = 'gd_T'

# ---- stringency filter: drop likely false-positive DNT/conv_T calls ----
# Author-annotated CD4+/CD8+ cells with zero CD4/CD8A/CD8B counts are dropout artefacts.
# Require either (a) author annotation is not CD4/CD8, or (b) cluster-level support:
# the cluster's mean CD4+CD8 score is near zero.
cd8_score = None  # (unused placeholder)
print('\n[gate] raw cell counts by gate:')
print(pd.Series(gate).value_counts().to_string())

author = adata.obs['annotate_Tcelltype'].fillna('NA').astype(str)
strict_dnt = dnt_mask.copy()
# keep DNT if author says CD4+/CD8+ ONLY when the whole cluster is DNT-like
cluster_frac_dnt = pd.Series(dnt_mask).groupby(adata.obs['leiden'].values).transform('mean').values
strict_dnt &= (~author.isin(['CD4+', 'CD8+', 'Treg'])) | (cluster_frac_dnt > 0.5)

gate2 = gate.copy()
gate2[(gate == 'ab_DNT') & ~strict_dnt] = 'conv_T'
gate2[(gate == 'gd_T') & ~strict_dnt] = 'conv_T'
adata.obs['gate_raw'] = gate
adata.obs['gate'] = gate2
print('\n[gate STRICT] counts after removing dropout-driven false positives:')
print(pd.Series(gate2).value_counts().to_string())
print(f"[gate STRICT] ab_DNT fraction of CD3+ = "
      f"{(gate2=='ab_DNT').sum()/max(cd3.sum(),1)*100:.2f}%  | gd_T = "
      f"{(gate2=='gd_T').sum()/max(cd3.sum(),1)*100:.2f}%  "
      f"(expected 1-3% / 1-5%)")
print('\n[gate] crosstab vs author annotation:')
print(pd.crosstab(adata.obs['gate'], author).to_string())

# ---------- Step 5: marker scoring ----------
marker_sets = {
    'cytotoxicity': ['GZMB', 'GZMA', 'GZMK', 'PRF1', 'NKG7', 'GNLY', 'KLRD1'],
    'IFN_gamma': ['IFNG', 'STAT1', 'CXCL13'],
    'exhaustion': ['PDCD1', 'HAVCR2', 'LAG3', 'TIGIT', 'TOX', 'CTLA4'],
    'naive_memory': ['CCR7', 'SELL', 'LEF1', 'TCF7', 'IL7R'],
    'Treg': ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2'],
    'MHCII_APC': ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'CD74', 'HLA-DMA'],
    'proliferation': ['MKI67', 'TOP2A', 'STMN1'],
    'tissue_residency': ['ITGA1', 'ITGAE', 'CD69', 'RUNX3'],
    'gd_signature': ['TRDC', 'TRDV1', 'TRDV2', 'TRGC1', 'KLRD1', 'ZBTB16'],
}
# use raw log-normalized values (X is log1p normalized; .raw holds it)
Ln = adata.raw.to_adata()
for name, gs in marker_sets.items():
    gs2 = [g for g in gs if g in Ln.var_names]
    if gs2:
        sc.tl.score_genes(Ln, gs2, score_name=name)
for name in marker_sets:
    if name in Ln.obs: adata.obs[name] = Ln.obs[name].values

score_cols = [c for c in marker_sets if c in adata.obs.columns]
score_tab = adata.obs.groupby('gate')[score_cols].mean().round(3)
score_tab.to_csv(os.path.join(OUT, 'gate_function_scores.csv'))
print('\n[score] mean signature score by gate:')
print(score_tab.to_string())

# ---------- Step 6: DNT/gdT subset analysis ----------
sub = adata[adata.obs.gate.isin(['ab_DNT', 'gd_T'])].copy()
print(f'\n[subset] DNT+gdT cells = {sub.n_obs} (ab_DNT={sum(sub.obs.gate=="ab_DNT")}, gd_T={sum(sub.obs.gate=="gd_T")})')
if sub.n_obs > 100:
    sc.pp.highly_variable_genes(sub, n_top_genes=1000, batch_key='sample')
    sc.tl.pca(sub, n_comps=min(20, sub.n_obs // 10), use_highly_variable=True)
    sc.pp.neighbors(sub, n_neighbors=15, n_pcs=min(15, sub.obsm['X_pca'].shape[1]))
    sc.tl.leiden(sub, resolution=0.8, key_added='dnt_sub', flavor='igraph', n_iterations=2)
    sc.tl.umap(sub)
    sc.tl.rank_genes_groups(sub, 'dnt_sub', method='wilcoxon')
    tops = {k: list(sc.get.rank_genes_groups_df(sub, group=k).head(15).names) for k in sub.obs.dnt_sub.cat.categories}
    pd.DataFrame(tops).to_csv(os.path.join(OUT, 'dnt_subcluster_markers.csv'), index=False)
    print(f'[subset] {sub.obs.dnt_sub.nunique()} DNT/gdT subclusters identified')
    print(pd.crosstab(sub.obs['dnt_sub'], sub.obs['gate']).to_string())
    print(pd.crosstab(sub.obs['dnt_sub'], sub.obs['sample']).to_string())
    sub.write(os.path.join(OUT, 'dnt_gdt_subset.h5ad'), compression='gzip')

# ---------- Step 7: TCR clonal metrics ----------
tcr = adata.obs[adata.obs.gate.isin(['ab_DNT', 'gd_T', 'conv_T'])]
cl = tcr.groupby('gate').agg(
    n_cells=('gate', 'size'),
    pct_with_clonotype=('clonotype_id', lambda x: x.notna().mean() * 100),
    median_clone_size=('clonotype_size', 'median'),
    mean_clone_size=('clonotype_size', 'mean'),
    max_clone_size=('clonotype_size', 'max'),
).round(2)
cl.to_csv(os.path.join(OUT, 'tcr_clonal_metrics.csv'))
print('\n[TCR] clonal metrics by gate:')
print(cl.to_string())

tcr.to_csv(os.path.join(OUT, 'tcr_cell_level.csv'))

# ---------- Step 8: save ----------
adata.obs.to_csv(os.path.join(OUT, 'cell_metadata_all.csv'))
adata.write(os.path.join(OUT, 'gse163108_processed.h5ad'), compression='gzip')
print('\n[done] outputs in', OUT)
