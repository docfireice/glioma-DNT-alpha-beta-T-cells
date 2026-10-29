# -*- coding: utf-8 -*-
"""08b: diagnostics for GSE182109 gating"""
import anndata as ad, numpy as np, pandas as pd, os

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE182109')
a = ad.read_h5ad(os.path.join(BASE, 'gse182109_processed.h5ad'), backed='r')
obs = a.obs
genes = ['TRDC', 'TRDV1', 'TRDV2', 'TRGC1', 'TRGC2', 'CD3D', 'CD3E', 'TRAC', 'CD4', 'CD8A', 'CD8B']
idx = [a.var_names.get_loc(g) for g in genes if g in a.var_names]
names = [g for g in genes if g in a.var_names]
sub = a.X[:, idx]
cnt = np.asarray(sub.todense() if hasattr(sub, 'todense') else sub) if not hasattr(sub, 'toarray') else np.asarray(sub.toarray())
expr = pd.DataFrame(cnt, columns=names, index=obs.index)
expr['gate'] = obs['gate'].values.astype(str)
expr['leiden'] = obs['leiden'].values.astype(str)
expr['total_counts'] = obs['total_counts'].values
expr['n_genes'] = obs['n_genes_by_counts'].values

print('=== per-gate quality ===')
print(expr.groupby('gate')[['total_counts', 'n_genes']].median())

print('\n=== gamma-delta marker detection among CD3+ cells ===')
cd3 = expr['CD3D'] > 0
gdm = (expr[['TRDC', 'TRDV1', 'TRDV2', 'TRGC1']].sum(1) > 0)
print('CD3+ cells:', cd3.sum())
print('CD3+ & gd-marker+:', (cd3 & gdm).sum())
print('gd-marker+ & CD8A>0:', (gdm & (expr['CD8A'] > 0)).sum())
print('gd-marker+ & CD4>0:', (gdm & (expr['CD4'] > 0)).sum())
print('gd-marker+ & CD4=0 & CD8A=0 & CD8B=0:', (gdm & (expr['CD4'] == 0) & (expr['CD8A'] == 0) & (expr['CD8B'] == 0)).sum())

print('\n=== raw ab_DNT (CD3+ TRDC- CD4- CD8- ) composition ===')
ab = cd3 & (expr['TRDC'] == 0) & (expr['CD4'] == 0) & (expr['CD8A'] == 0) & (expr['CD8B'] == 0)
print('raw ab_DNT count:', ab.sum())
print('  with TRAC>0:', (ab & (expr['TRAC'] > 0)).sum())
print('  CD3D counts==1 (shallow):', (ab & (expr['CD3D'] == 1)).sum())
print('  median total_counts:', expr.loc[ab, 'total_counts'].median())
print('  median n_genes:', expr.loc[ab, 'n_genes'].median())
conv = expr['gate'] == 'conv_T'
print('  conv_T median total_counts:', expr.loc[conv, 'total_counts'].median())
print('  conv_T median n_genes:', expr.loc[conv, 'n_genes'].median())

print('\n=== per-cluster raw-DNT fraction (top 10) ===')
cf = expr[ab].groupby('leiden').size()
tot = expr.groupby('leiden').size()
frac = (cf / tot).dropna().sort_values(ascending=False)
print(pd.DataFrame({'rawDNT': cf, 'n': tot, 'frac': frac}).head(12))
expr.assign(raw_abDNT=ab).to_csv(os.path.join(BASE, 'gating_diag.csv'))
print('[saved] gating_diag.csv')
