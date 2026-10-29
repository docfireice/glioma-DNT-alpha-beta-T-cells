# -*- coding: utf-8 -*-
"""Module 9b: cross-dataset replication of the differentiation-potential analysis
on GSE182109 (44 GBM/LGG samples, 219,908 cells; T cells via v2 gating).
Same CytoTRACE-style pipeline as module 9, applied to the log-normalised matrix
(counts layer not stored -> gene counts derived from X>0, entropy from expm1).
Outputs: fig21_*, results/GSE182109/diffpot_cell_level.csv
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE182109')
FIGS = os.path.join(BASE, 'figs')
os.makedirs(FIGS, exist_ok=True)
rng = np.random.default_rng(42)

print('[1] load h5ad + v2 metadata', flush=True)
a = sc.read_h5ad(os.path.join(BASE, 'gse182109_processed.h5ad'))
meta = pd.read_csv(os.path.join(BASE, 'cell_metadata_v2.csv'), index_col=0)
assert list(meta.index) == list(a.obs_names), 'metadata order mismatch'

gate_v2 = meta['gate_v2'].values
umap = a.obsm['X_umap']
tmask = gate_v2 != 'non_T'
print(f'   T cells (v2): {tmask.sum()}', flush=True)

X = a.X
X = sp.csr_matrix(X) if not sp.issparse(X) else X
Xt = X[tmask]                     # T-cell log-normalised matrix
gate_t = gate_v2[tmask]
umap_t = umap[tmask]
del a, X

# gene filter: expressed in >=1% of T cells
det = np.asarray((Xt > 0).sum(0)).ravel()
keep = det >= int(0.01 * Xt.shape[0])
Xk = Xt[:, keep]
gene_counts = np.asarray((Xk > 0).sum(1)).ravel().astype(float)
print(f'[2] genes kept {keep.sum()} | gene counts/cell median {np.median(gene_counts):.0f}', flush=True)

# GCS genes
n_sub = min(2000, Xk.shape[0])
idx = rng.choice(Xk.shape[0], n_sub, replace=False)
Cs = Xk[idx].toarray().astype(np.float32)
rg = stats.rankdata(Cs, axis=0)
rg -= rg.mean(0)
rgc = stats.rankdata(gene_counts[idx])
rgc -= rgc.mean()
num = (rg * rgc[:, None]).sum(0)
den = np.sqrt((rg ** 2).sum(0) * (rgc ** 2).sum()) + 1e-12
rho = num / den
rho[~np.isfinite(rho)] = 0
order = np.argsort(rho)[::-1]
n_gcs = int(min(200, (rho > 0).sum()))
gcs_idx = order[:n_gcs]
print(f'[3] GCS genes {n_gcs}, top rho={rho[gcs_idx[0]]:.3f}', flush=True)

size = np.asarray(Xk.sum(1)).ravel()          # log-norm lib size ~ monotone w/ depth
Xg = Xk[:, gcs_idx].toarray().astype(np.float32)
lin = np.expm1(Xg)                             # back to normalised linear
Xg2 = np.log1p(lin / np.maximum(size, 1)[:, None] * 1e4)
pcs = PCA(n_components=30, random_state=0).fit_transform(Xg2)
nn = NearestNeighbors(n_neighbors=31).fit(pcs)
_, nbr = nn.kneighbors(pcs)
sm = gene_counts.copy()
for _ in range(3):
    sm = 0.5 * sm + 0.5 * sm[nbr[:, 1:]].mean(1)
cyto = stats.rankdata(sm) / len(sm)

# transcriptional entropy on linear scale (per-cell normalised expm1)
L = Xt.tocsr().astype(np.float64).copy()
L.data = np.expm1(L.data)
tot = np.asarray(L.sum(1)).ravel()
L = sp.diags(1.0 / np.maximum(tot, 1e-9)) @ L
L = L.tocsr()
Ll = L.copy()
Ll.data = np.log2(Ll.data)
ent = -np.asarray(L.multiply(Ll).sum(1)).ravel()

out = pd.DataFrame({'gate': gate_t, 'cytotrace': cyto,
                    'transcriptional_entropy': ent,
                    'gene_counts': gene_counts,
                    'umap1': umap_t[:, 0], 'umap2': umap_t[:, 1]})
out.to_csv(os.path.join(BASE, 'diffpot_cell_level.csv'), index=False)

rows = []
for g in ['conv_T', 'ab_DNT', 'gd_T']:
    m = out.gate == g
    if m.sum() < 3:
        continue
    rows.append({'gate': g, 'n': int(m.sum()),
                 'cytotrace_median': round(float(out.loc[m, 'cytotrace'].median()), 4),
                 'entropy_median': round(float(out.loc[m, 'transcriptional_entropy'].median()), 4)})
gt = pd.DataFrame(rows)
print(gt.to_string(index=False), flush=True)
u, p = stats.mannwhitneyu(out.loc[out.gate == 'ab_DNT', 'cytotrace'],
                          out.loc[out.gate == 'conv_T', 'cytotrace'])
print(f'[4] ab_DNT vs conv_T CytoTRACE MWU p={p:.3e}', flush=True)
gt.to_csv(os.path.join(BASE, 'diffpot_by_gate.csv'), index=False)

cmap = {'conv_T': '#4C72B0', 'ab_DNT': '#C44E52', 'gd_T': '#55A868'}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
data, labs, cols = [], [], []
for g in ['conv_T', 'ab_DNT', 'gd_T']:
    v = out.loc[out.gate == g, 'cytotrace'].values
    data.append(v); labs.append(f'{g}\n(n={len(v)})'); cols.append(cmap[g])
parts = axes[0].violinplot(data, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(cols[i]); pc.set_alpha(0.75)
axes[0].set_xticks(range(1, len(labs) + 1)); axes[0].set_xticklabels(labs)
axes[0].set_ylabel('CytoTRACE differentiation potential')
axes[0].set_title('GSE182109 T cells')

for g in ['conv_T', 'ab_DNT', 'gd_T']:
    m = out.gate == g
    axes[1].scatter(out.loc[m, 'umap1'], out.loc[m, 'umap2'], s=1.5, alpha=0.25,
                    c=cmap[g], label=g)
axes[1].legend(markerscale=4, fontsize=8); axes[1].set_title('Gates (UMAP)')
axes[1].set_xticks([]); axes[1].set_yticks([])

s = axes[2].scatter(out['umap1'], out['umap2'], c=out['cytotrace'], s=1.5, cmap='magma')
plt.colorbar(s, ax=axes[2], label='CytoTRACE')
axes[2].set_title('Differentiation potential (UMAP)')
axes[2].set_xticks([]); axes[2].set_yticks([])
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig21_gse182109_diffpot.png'), dpi=150); plt.close()

with open(os.path.join(BASE, 'diffpot_stats.txt'), 'w') as f:
    f.write(f'ab_DNT_vs_convT_cytotrace_MWU p={p:.3e}\n')
    f.write(gt.to_string(index=False) + '\n')
print('[done] module 9b', flush=True)
