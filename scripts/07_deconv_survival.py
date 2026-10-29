# -*- coding: utf-8 -*-
"""Module 4: build gate signature matrix from GSE163108, NNLS deconvolve TCGA-GBM bulk,
Cox + KM for ab_DNT / gd_T fractions vs OS."""
import scanpy as sc, numpy as np, pandas as pd, os, gzip, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import nnls

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE163108')
TCGA = os.path.join(os.path.dirname(__file__), '..', 'data', 'TCGA')
FIGS = os.path.join(BASE, 'figs'); os.makedirs(FIGS, exist_ok=True)

# ---------- 1. signatures from scRNA ----------
adata = sc.read_h5ad(os.path.join(BASE, 'gse163108_processed.h5ad'))
adata.obs_names_make_unique()
sub = adata[adata.obs['gate'].isin(['ab_DNT', 'gd_T', 'conv_T'])].copy()
sc.pp.normalize_total(sub, target_sum=1e4); sc.pp.log1p(sub)
sc.pp.highly_variable_genes(sub, n_top_genes=3000, batch_key='sample')
sc.tl.rank_genes_groups(sub, 'gate', method='wilcoxon')
sig_genes = {}
for g in ['ab_DNT', 'gd_T', 'conv_T']:
    names = sc.get.rank_genes_groups_df(sub, group=g).head(100)['names']
    sig_genes[g] = [n for n in names if n in adata.var_names]
genes = sorted(set(sum(sig_genes.values(), [])))
print(f'[sig] genes={len(genes)}')

expr = sub[:, genes].to_df()
mat = np.column_stack([expr.loc[sub.obs['gate'] == g].mean().values for g in ['ab_DNT', 'gd_T', 'conv_T']])
pd.DataFrame(mat, index=genes, columns=['ab_DNT', 'gd_T', 'conv_T']).to_csv(
    os.path.join(BASE, 'signature_matrix.csv'))
print('[sig] matrix done:', mat.shape)

# ---------- 2. deconvolve TCGA-GBM ----------
with gzip.open(os.path.join(TCGA, 'GBM_HiSeqV2.gz'), 'rt') as f:
    bulk = pd.read_csv(f, sep='\t', index_col=0)
# drop duplicate gene rows (mean)
bulk = bulk.groupby(level=0).mean()
common = [g for g in genes if g in bulk.index]
print(f'[bulk] TCGA samples={bulk.shape[1]}, matched sig genes={len(common)}')
Xs = bulk.loc[common].T.values.astype(float)
# z-score per gene in bulk to be scale-compatible with scRNA-derived relative usage
Mb = np.column_stack([(bulk.loc[common].mean(axis=1) - bulk.loc[common].mean(axis=1).mean()) for _ in range(3)])
# Use NNLS on scaled matrices: scale both bulk and signature per gene (z-score)
bulk_z = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
sig_sc = mat[[genes.index(g) for g in common]]
sig_z = (sig_sc - sig_sc.mean(0)) / (sig_sc.std(0) + 1e-9)
fracs = np.apply_along_axis(lambda y: nnls(sig_z, y)[0], 1, bulk_z)
frac_df = pd.DataFrame(fracs, index=bulk.columns, columns=['ab_DNT', 'gd_T', 'conv_T'])
frac_df = frac_df.div(frac_df.sum(axis=1), axis=0)
frac_df.to_csv(os.path.join(BASE, 'tcga_deconv_fractions.csv'))
print(frac_df.describe().loc[['25%', '50%', '75%']])

# ---------- 3. clinical + Cox + KM ----------
clin = pd.read_csv(os.path.join(TCGA, 'GBM_clinical.tsv'), sep='\t', index_col=0)
frac_df['patient'] = [s[:12] for s in frac_df.index]
# Xena GBM clinical: CDE_vital_status (DECEASED/LIVING) + CDE_survival_time (days)
mm = clin[['CDE_vital_status', 'CDE_survival_time']].copy()
mm.columns = ['OS', 'OS.time']
mm = mm[~mm.index.duplicated(keep='first')]
# join: direct on sample id, fallback to 12-char patient prefix
frac_df2 = frac_df.copy(); frac_df2.index.name = 'sid'
m = frac_df2.join(mm, how='inner')
if len(m) == 0:
    frac_df2['patient'] = [s[:12] for s in frac_df2.index]
    mm2 = mm.copy(); mm2['patient'] = [s[:12] for s in mm.index]
    mm2 = mm2[~mm2.index.duplicated(keep='first')].set_index('patient') if False else mm2
    mm2 = mm2.reset_index().rename(columns={'index': 'clin_id'})
    mm2['patient'] = mm2['clin_id'].str[:12]
    mm2 = mm2.drop_duplicates('patient').set_index('patient')
    m = frac_df2.join(mm2[['OS', 'OS.time']], on='patient', how='inner')
m = m.dropna(subset=['OS', 'OS.time'])
m['OS.time'] = pd.to_numeric(m['OS.time'], errors='coerce')
m['event'] = (m['OS'] == 'DECEASED').astype(int)
m = m.dropna(subset=['OS.time'])
print('[clin] event distribution:', m['event'].value_counts().to_dict(), '| n =', len(m))

from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

results = {}
for cell in ['ab_DNT', 'gd_T']:
    med = m[cell].median()
    m[cell + '_high'] = (m[cell] > med).astype(int) if med > 0 else (m[cell] > 0).astype(int)
    cph = CoxPHFitter()
    cph.fit(m[[cell + '_high', 'OS.time', 'event']], duration_col='OS.time', event_col='event')
    hr = np.exp(cph.params_[cell + '_high'])
    p = cph.summary.loc[cell + '_high', 'p']
    results[cell] = (hr, p)
    print(f'[cox] {cell}: HR={hr:.3f}, p={p:.4f}')

    hi = m[m[cell + '_high'] == 1]; lo = m[m[cell + '_high'] == 0]
    lr = logrank_test(hi['OS.time'], lo['OS.time'], hi['event'], lo['event'])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    for grp, lbl, c in [(hi, f'{cell} high (n={len(hi)})', '#D65F5F'), (lo, f'{cell} low (n={len(lo)})', '#4878CF')]:
        kmf.fit(grp['OS.time'] / 30.44, grp['event'], label=lbl)
        kmf.plot_survival_function(ax=ax, color=c, ci_show=False)
    ax.set_title(f'{cell} fraction vs OS (TCGA-GBM) | logrank p={lr.p_value:.3f}')
    ax.set_xlabel('months'); ax.set_ylabel('OS probability')
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, f'fig10_km_{cell}.png'), dpi=150); plt.close()

pd.DataFrame(results, index=['HR', 'p']).T.to_csv(os.path.join(BASE, 'tcga_cox_results.csv'))
print('[done] module 4')
