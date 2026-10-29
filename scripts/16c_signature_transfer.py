# -*- coding: utf-8 -*-
"""Module 10c: signature transferability test.
Derive DNT/gd-T-specific DE genes from GSE163108 (ab_DNT vs conv_T, Wilcoxon),
then score them in independent Couturier 2020 cohorts (SL040/SL057) and report
AUROC of separating gated ab_DNT (resp. gd_T) from conv_T.
Also within-GSE182109 as a platform-matched positive control.
Outputs: fig22_signature_transfer.png, results/CELLxGENE/signature_transfer.csv
"""
import os, gc
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASEX = os.path.join(os.path.dirname(__file__), '..', 'results')
DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'CELLxGENE')
FIGS = os.path.join(BASEX, 'CELLxGENE', 'figs')
os.makedirs(FIGS, exist_ok=True)

# ---------- 1) derive specific DE genes from GSE163108 ----------
print('[1] GSE163108 DE (ab_DNT vs conv_T)', flush=True)
a = sc.read_h5ad(os.path.join(BASEX, 'GSE163108', 'gse163108_processed.h5ad'))
a = a[~a.obs['gate'].isin(['non_T'])].copy()
sc.tl.rank_genes_groups(a, 'gate', groups=['ab_DNT'], reference='conv_T', method='wilcoxon')
r = a.uns['rank_genes_groups']
df = pd.DataFrame({'gene': np.array(r['names']['ab_DNT']),
                   'logfc': np.array(r['logfoldchanges']['ab_DNT']),
                   'padj': np.array(r['pvals_adj']['ab_DNT'])})
dnt_genes = df[(df.padj < 0.05) & (df.logfc > 0.5)].gene.tolist()[:100]
print(f'   ab_DNT-specific genes: {len(dnt_genes)}', flush=True)
dnt_genes[:10] and print('   e.g.', dnt_genes[:10])

sc.tl.rank_genes_groups(a, 'gate', groups=['gd_T'], reference='conv_T', method='wilcoxon')
r = a.uns['rank_genes_groups']
dfg = pd.DataFrame({'gene': np.array(r['names']['gd_T']),
                    'logfc': np.array(r['logfoldchanges']['gd_T']),
                    'padj': np.array(r['pvals_adj']['gd_T'])})
gd_genes = dfg[(dfg.padj < 0.05) & (dfg.logfc > 0.5)].gene.tolist()[:100]
print(f'   gd_T-specific genes: {len(gd_genes)}', flush=True)
del a; gc.collect()

res_rows = []


def score_and_auc(path_or_adata, tag, gate_col='gate'):
    if isinstance(path_or_adata, str):
        ad = sc.read_h5ad(path_or_adata)
        if str(ad.var_names[0]).startswith('ENSG'):
            sym = ad.var['feature_name'].astype(str).values
            _, first = np.unique(sym, return_index=True)
            first = np.sort(first)
            ad = ad[:, first].copy()
            ad.var_names = pd.Index(sym[first])
        ad.X = sp.csr_matrix(ad.X) if not sp.issparse(ad.X) else ad.X.tocsr()
        if ad.X.data.max() > 200:
            sc.pp.normalize_total(ad, target_sum=1e4)
            sc.pp.log1p(ad)
        # v2 gating
        vidx = {g: i for i, g in enumerate(ad.var_names)}
        n = ad.n_obs
        def pos(g):
            if g not in vidx:
                return np.zeros(n, bool)
            return ad.X[:, vidx[g]].toarray().ravel() > 0
        t = pos('CD3D') | pos('CD3E')
        gdc = np.zeros(n, int)
        for g in ['TRDC', 'TRDV1', 'TRDV2', 'TRGC1']:
            gdc += pos(g).astype(int)
        gd = (gdc > 0) & t
        ab = t & ~gd & ~pos('CD4') & ~pos('CD8A') & ~pos('CD8B')
        conv = t & ~gd & (pos('CD4') | pos('CD8A') | pos('CD8B'))
        gate = np.array(['other'] * n, dtype=object)
        gate[conv] = 'conv_T'; gate[gd] = 'gd_T'; gate[ab] = 'ab_DNT'
        ad.obs['gate'] = gate
    else:
        ad = path_or_adata
    genes = set(ad.var_names)
    gl_d = [g for g in dnt_genes if g in genes]
    gl_g = [g for g in gd_genes if g in genes]
    sc.tl.score_genes(ad, gl_d, score_name='dnt_sig', use_raw=False)
    if gl_g:
        sc.tl.score_genes(ad, gl_g, score_name='gd_sig', use_raw=False)
    for cls, sigc in [('ab_DNT', 'dnt_sig'), ('gd_T', 'gd_sig')]:
        m1 = (ad.obs[gate_col] == cls).values
        m0 = (ad.obs[gate_col] == 'conv_T').values
        if m1.sum() < 10 or m0.sum() < 10 or sigc not in ad.obs:
            auc = np.nan
        else:
            y = np.r_[np.ones(m1.sum()), np.zeros(m0.sum())]
            x = np.r_[ad.obs.loc[m1, sigc].values, ad.obs.loc[m0, sigc].values]
            auc = float(roc_auc_score(y, x))
        res_rows.append({'cohort': tag, 'target': cls, 'n_target': int(m1.sum()),
                         'n_conv': int(m0.sum()), 'AUROC': round(auc, 3) if auc == auc else np.nan,
                         'genes_used': len(gl_d) if cls == 'ab_DNT' else len(gl_g)})
        print(f'   [{tag}] {cls} vs conv_T AUROC = {auc:.3f} (n={m1.sum()})', flush=True)
    return ad


print('[2] Couturier SL040', flush=True)
ad1 = score_and_auc(os.path.join(DATA, 'SL040_gbm.h5ad'), 'Couturier-SL040')
del ad1; gc.collect()
print('[3] Couturier SL057', flush=True)
ad2 = score_and_auc(os.path.join(DATA, 'SL057_gbm.h5ad'), 'Couturier-SL057')
del ad2; gc.collect()

print('[4] GSE182109 (platform-matched control)', flush=True)
ad3 = score_and_auc(os.path.join(BASEX, 'GSE182109', 'gse182109_processed.h5ad'), 'GSE182109')
ad3.obs[[c for c in ['gate', 'dnt_sig', 'gd_sig'] if c in ad3.obs]].to_csv(
    os.path.join(BASEX, 'GSE182109', 'sig_transfer_cell_level.csv'))
del ad3; gc.collect()

res = pd.DataFrame(res_rows)
res.to_csv(os.path.join(BASEX, 'CELLxGENE', 'signature_transfer.csv'), index=False)
print(res.to_string(index=False), flush=True)

# ---------- fig22 ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
cohorts = res['cohort'].unique()
x = np.arange(len(cohorts)); w = 0.35
for i, (tgt, c) in enumerate([('ab_DNT', '#C44E52'), ('gd_T', '#55A868')]):
    v = [res[(res.cohort == c_) & (res.target == tgt)]['AUROC'].values[0] for c_ in cohorts]
    v = [x if x == x else 0 for x in v]
    b = ax.bar(x + (i - 0.5) * w, v, w, color=c, label=tgt)
    ax.bar_label(b, fmt='%.2f', fontsize=8)
ax.axhline(0.5, color='grey', ls='--', lw=1)
ax.set_xticks(x); ax.set_xticklabels(cohorts, fontsize=9)
ax.set_ylim(0, 1); ax.set_ylabel('AUROC')
ax.set_title('Transferability of GSE163108-derived DNT/gd-T signatures')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig22_signature_transfer.png'), dpi=150); plt.close()
print('[done] module 10c', flush=True)
