# -*- coding: utf-8 -*-
"""Module 9 (dry-lab alternative to RNA velocity):
independent inference of differentiation direction using
  1) CytoTRACE-style differentiation potential (gene-count smoothing over GCS genes)
  2) transcriptional entropy
  3) cross-check against the diffusion pseudotime (DPT) from module 3
Rationale: GSE163108/GSE182109 deposit only filtered spliced count matrices
(no BAM/loom), so unspliced counts are unavailable and spliced/unspliced RNA
velocity cannot be estimated. CytoTRACE differentiation potential is the standard
computational substitute and needs only the spliced matrix.
Outputs: fig18_*, results/GSE163108/diffpot_cell_level.csv
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

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE163108')
FIGS = os.path.join(BASE, 'figs')
os.makedirs(FIGS, exist_ok=True)
rng = np.random.default_rng(42)

print('[1] load', flush=True)
a = sc.read_h5ad(os.path.join(BASE, 'gse163108_processed.h5ad'))
sub = a[~a.obs['gate'].isin(['non_T'])].copy()
del a
print('   CD3+ cells:', sub.shape, flush=True)

counts = sub.layers['counts']
counts = sp.csr_matrix(counts) if not sp.issparse(counts) else counts.tocsr()

# ---------- gene filtering ----------
min_cells = int(0.01 * sub.n_obs)
det = np.asarray((counts > 0).sum(0)).ravel()
keep = det >= min_cells
print(f'[2] genes kept {keep.sum()}/{sub.n_vars}', flush=True)
C = counts[:, keep]
gene_counts = np.asarray((C > 0).sum(1)).ravel().astype(float)
print('[3] gene counts/cell median %.0f' % np.median(gene_counts), flush=True)

# ---------- 1) CytoTRACE-style potential ----------
n_sub = min(2000, C.shape[0])
idx = rng.choice(C.shape[0], n_sub, replace=False)
Cs = C[idx].toarray().astype(np.float32)
rg = stats.rankdata(Cs, axis=0)
rg = rg - rg.mean(0)
rgc = stats.rankdata(gene_counts[idx])
rgc = rgc - rgc.mean()
num = (rg * rgc[:, None]).sum(0)
den = np.sqrt((rg ** 2).sum(0) * (rgc ** 2).sum()) + 1e-12
rho = num / den
rho[~np.isfinite(rho)] = 0
order = np.argsort(rho)[::-1]
n_gcs = int(min(200, (rho > 0).sum()))
gcs_idx = order[:n_gcs]
print(f'[4] GCS genes {n_gcs}, top rho={rho[gcs_idx[0]]:.3f}', flush=True)

size = np.asarray(C.sum(1)).ravel()
Xg = C[:, gcs_idx].toarray().astype(np.float32)
Xg = np.log1p(Xg / np.maximum(size, 1)[:, None] * 1e4)
pcs = PCA(n_components=30, random_state=0).fit_transform(Xg)

nn = NearestNeighbors(n_neighbors=31).fit(pcs)
_, nbr = nn.kneighbors(pcs)
sm = gene_counts.copy()
for _ in range(3):
    sm = 0.5 * sm + 0.5 * sm[nbr[:, 1:]].mean(1)
sub.obs['cytotrace'] = stats.rankdata(sm) / len(sm)

# ---------- 2) transcriptional entropy (from counts) ----------
P = sp.csr_matrix(C).astype(np.float64)
tot = np.asarray(P.sum(1)).ravel()
P = sp.diags(1.0 / np.maximum(tot, 1e-9)) @ P
P = P.tocsr()
L = P.copy()
L.data = np.log2(L.data)
sub.obs['transcriptional_entropy'] = -np.asarray(P.multiply(L).sum(1)).ravel()

# ---------- 3) align with DPT ----------
dpt = pd.read_csv(os.path.join(BASE, 'dpt_cell_level.csv'), index_col=0)
assert len(dpt) == sub.n_obs, f'dpt rows {len(dpt)} != {sub.n_obs}'
sub.obs['dpt'] = dpt['dpt'].values
sub.obs['pt_bin'] = dpt['pt_bin'].values

r, p = stats.spearmanr(sub.obs['cytotrace'], sub.obs['dpt'])
r2, p2 = stats.spearmanr(sub.obs['transcriptional_entropy'], sub.obs['dpt'])
print(f'[5] CytoTRACE vs DPT rho={r:.3f} p={p:.3e}', flush=True)
print(f'    entropy   vs DPT rho={r2:.3f} p={p2:.3e}', flush=True)

rows = []
for g in ['conv_T', 'ab_DNT', 'gd_T']:
    m = (sub.obs['gate'] == g).values
    if m.sum() < 3:
        continue
    rows.append({'gate': g, 'n': int(m.sum()),
                 'cytotrace_median': round(float(sub.obs.loc[m, 'cytotrace'].median()), 4),
                 'entropy_median': round(float(sub.obs.loc[m, 'transcriptional_entropy'].median()), 4),
                 'dpt_median': round(float(sub.obs.loc[m, 'dpt'].median()), 4),
                 'naive_memory': round(float(sub.obs.loc[m, 'naive_memory'].median()), 4),
                 'exhaustion': round(float(sub.obs.loc[m, 'exhaustion'].median()), 4)})
gt = pd.DataFrame(rows)
print(gt.to_string(index=False), flush=True)
u1, p1 = stats.mannwhitneyu(sub.obs.loc[sub.obs.gate == 'ab_DNT', 'cytotrace'].values,
                            sub.obs.loc[sub.obs.gate == 'conv_T', 'cytotrace'].values)
print(f'[6] ab_DNT vs conv_T CytoTRACE MWU p={p1:.3e}', flush=True)

cs = sub.obs['clonotype_size'].astype(float).clip(lower=1).values
cs = np.where(np.isfinite(cs), cs, 1.0)
r3, p3 = stats.spearmanr(sub.obs['cytotrace'].values, np.log1p(cs))
print(f'[7] CytoTRACE vs log clone size rho={r3:.3f} p={p3:.3e}', flush=True)

# ---------- figures ----------
cmap_gate = {'conv_T': '#4C72B0', 'ab_DNT': '#C44E52', 'gd_T': '#55A868'}
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

ax = axes[0, 0]
hb = ax.hexbin(sub.obs['dpt'], sub.obs['cytotrace'], gridsize=60, cmap='viridis', bins='log')
plt.colorbar(hb, ax=ax, label='cells')
q = pd.qcut(sub.obs['dpt'], 25, labels=False, duplicates='drop')
gg = sub.obs.groupby(q, observed=True).agg(x=('dpt', 'mean'), y=('cytotrace', 'mean'))
ax.plot(gg['x'], gg['y'], color='red', lw=2)
ax.set_xlabel('diffusion pseudotime (DPT)')
ax.set_ylabel('CytoTRACE differentiation potential')
ax.set_title(f'CytoTRACE vs DPT (Spearman rho={r:.2f}, p={p:.1e})')

ax = axes[0, 1]
data, labs, cols = [], [], []
for g in ['conv_T', 'ab_DNT', 'gd_T']:
    v = sub.obs.loc[sub.obs.gate == g, 'cytotrace'].values
    if len(v) < 5:
        continue
    data.append(v); labs.append(f'{g}\n(n={len(v)})'); cols.append(cmap_gate[g])
parts = ax.violinplot(data, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(cols[i]); pc.set_alpha(0.7)
ax.set_xticks(range(1, len(labs) + 1)); ax.set_xticklabels(labs)
ax.set_ylabel('CytoTRACE differentiation potential')
ax.set_title('Differentiation potential by gate')

ax = axes[1, 0]
for g in ['conv_T', 'ab_DNT', 'gd_T']:
    m = (sub.obs['gate'] == g).values
    if m.sum() < 10:
        continue
    ax.scatter(sub.obs.loc[m, 'dpt'], sub.obs.loc[m, 'cytotrace'], s=2, alpha=0.22,
               color=cmap_gate[g], label=g)
ax.set_xlabel('DPT'); ax.set_ylabel('CytoTRACE')
ax.legend(markerscale=4, fontsize=8); ax.set_title('Gate overlay')

ax = axes[1, 1]
for g in ['conv_T', 'ab_DNT', 'gd_T']:
    m = (sub.obs['gate'] == g).values
    if m.sum() < 10:
        continue
    ax.scatter(sub.obs.loc[m, 'dpt'], sub.obs.loc[m, 'transcriptional_entropy'], s=2,
               alpha=0.2, color=cmap_gate[g], label=g)
ax.set_xlabel('DPT'); ax.set_ylabel('transcriptional entropy (bits)')
ax.set_title(f'Entropy vs DPT (rho={r2:.2f})')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig18_diffpot_vs_dpt.png'), dpi=150); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, key, lab in zip(axes, ['cytotrace', 'transcriptional_entropy'],
                        ['CytoTRACE differentiation potential', 'transcriptional entropy (bits)']):
    s = ax.scatter(sub.obsm['X_umap'][:, 0], sub.obsm['X_umap'][:, 1],
                   c=sub.obs[key], s=2, cmap='magma')
    plt.colorbar(s, ax=ax, label=lab)
    ax.set_title(lab); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig18b_umap_diffpot.png'), dpi=150); plt.close()

out = sub.obs[['sample', 'gate', 'clonotype_id', 'clonotype_size', 'cytotrace',
               'transcriptional_entropy', 'dpt', 'pt_bin', 'naive_memory', 'exhaustion']].copy()
out.to_csv(os.path.join(BASE, 'diffpot_cell_level.csv'))
gt.to_csv(os.path.join(BASE, 'diffpot_by_gate.csv'), index=False)
with open(os.path.join(BASE, 'diffpot_stats.txt'), 'w') as f:
    f.write(f'cytotrace_vs_dpt rho={r:.4f} p={p:.3e}\n')
    f.write(f'entropy_vs_dpt rho={r2:.4f} p={p2:.3e}\n')
    f.write(f'ab_DNT_vs_convT_cytotrace_MWU p={p1:.3e}\n')
    f.write(f'cytotrace_vs_logclonesize rho={r3:.4f} p={p3:.3e}\n')
print('[done] module 9', flush=True)
