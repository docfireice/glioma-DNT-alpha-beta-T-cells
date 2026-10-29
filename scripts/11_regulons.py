# -*- coding: utf-8 -*-
"""Module 7: SCENIC-style regulon analysis on GSE163108 CD3+ T cells.
Co-expression regulons: TF + top-50 positively correlated targets (Spearman on downsampled cells).
Activity: scanpy score_genes (mean-based AUCell surrogate).
Compare ab_DNT vs conv_T (Wilcoxon)."""
import scanpy as sc, anndata as ad, numpy as np, pandas as pd, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE163108')
FIGS = os.path.join(BASE, 'figs')
adata = sc.read_h5ad(os.path.join(BASE, 'gse163108_processed.h5ad'))
adata.obs_names_make_unique()
T = adata[adata.obs['gate'].isin(['ab_DNT', 'conv_T', 'gd_T'])].copy()
sc.pp.highly_variable_genes(T, n_top_genes=2000, batch_key='sample')
sc.pp.normalize_total(T, target_sum=1e4); sc.pp.log1p(T)
print('[T cells]', T.shape)

rng = np.random.default_rng(0)
idx = rng.choice(T.n_obs, min(8000, T.n_obs), replace=False)
S = T[idx, :]
sub = S[:, S.var['highly_variable'].values]
X = np.asarray(sub.X.toarray() if hasattr(sub.X, 'toarray') else sub.X)
df = pd.DataFrame(X, columns=sub.var_names)
print('[corr input]', df.shape)

TFS = ['FOXO1','FOXO3','TCF7','LEF1','EOMES','TBX21','GATA3','RUNX1','RUNX3','HIF1A','MYC',
       'NFKB1','RELA','STAT1','STAT3','STAT4','STAT5A','STAT5B','IRF1','IRF2','IRF4','IRF7','IRF8',
       'BATF','BHLHE40','PRDM1','TOX','NR4A1','NR4A2','NR4A3','FOS','JUN','ATF3','CREB1','EGR1',
       'EGR2','KLF2','KLF6','MAF','MAFB','IKZF1','IKZF2','IKZF3','SOX4','ETS1','SPI1','CEBPA','CEBPB']
tfs = [t for t in TFS if t in df.columns]
print(f'[TFs] {len(tfs)} in data')

# correlation regulons (on full T-cell matrix for target selection? use downsampled)
regulons = {}
for tf in tfs:
    corr = df.corrwith(df[tf], method='spearman')
    targets = corr.drop(tf).sort_values(ascending=False).head(50)
    targets = targets[targets > 0.2]
    if len(targets) >= 10:
        regulons[tf] = list(targets.index)
print(f'[regulons] {len(regulons)} regulons (>=10 targets, rho>0.2)')

# activity scoring on ALL T cells
acts = {}
for tf, tg in regulons.items():
    acts[tf] = sc.tl.score_genes(T, tg, score_name='A_' + tf, copy=False) if False else None
    sc.tl.score_genes(T, tg, score_name='A_' + tf)
act_cols = ['A_' + tf for tf in regulons]
A = T.obs[act_cols].copy()
A['gate'] = T.obs['gate'].values
A.to_csv(os.path.join(BASE, 'regulon_activity_cell_level.csv'))

# Wilcoxon ab_DNT vs conv_T
dnt = A[A.gate == 'ab_DNT']; conv = A[A.gate == 'conv_T']
rows = []
for tf in regulons:
    rho, p = stats.mannwhitneyu(dnt['A_' + tf], conv['A_' + tf])
    rows.append({'TF': tf, 'median_DNT': dnt['A_' + tf].median(),
                 'median_conv': conv['A_' + tf].median(),
                 'delta': dnt['A_' + tf].median() - conv['A_' + tf].median(), 'p': p})
res = pd.DataFrame(rows).sort_values('delta')
res['padj'] = res['p'] * len(res)  # BH-lite (bonferroni)
res.to_csv(os.path.join(BASE, 'regulon_DNT_vs_conv.csv'), index=False)
print(res.head(8).round(4).to_string())
print('...')
print(res.tail(8).round(4).to_string())

# heatmap: regulon activity per gate (z-scored)
z = A.groupby('gate')[act_cols].mean().T
z = (z - z.mean(1).values[:, None]) / z.std(1).values[:, None]
plt.figure(figsize=(4.5, max(6, len(z) * 0.18)))
sns.heatmap(z[[ 'conv_T', 'ab_DNT', 'gd_T']].dropna(), cmap='coolwarm', center=0)
plt.title('Regulon activity (z) by gate')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig15_regulon_heatmap.png'), dpi=150); plt.close()
print('[done] module 7')
