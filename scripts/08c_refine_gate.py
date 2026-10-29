# -*- coding: utf-8 -*-
"""08c: refined gating for GSE182109
gd_T: CD3+ & (TRDC|TRDV1|TRDV2|TRGC1)+ , CD4/CD8 allowed (gamma-delta lineage definition)
ab_DNT: CD3+ & gd-markers==0 & CD4=0 & CD8A=0 & CD8B=0 ; TRAC+ subset = confirmed alpha-beta
"""
import anndata as ad, numpy as np, pandas as pd, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE182109')
FIGS = os.path.join(BASE, 'figs')
a = ad.read_h5ad(os.path.join(BASE, 'gse182109_processed.h5ad'))
genes = ['TRDC', 'TRDV1', 'TRDV2', 'TRGC1', 'CD3D', 'CD3E', 'TRAC', 'CD4', 'CD8A', 'CD8B']
idx = [a.var_names.get_loc(g) for g in genes]
X = a.X[:, idx]
cnt = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
e = pd.DataFrame(cnt, columns=genes, index=a.obs_names)

cd3 = (e['CD3D'] + e['CD3E']) > 0
gd = e[['TRDC', 'TRDV1', 'TRDV2', 'TRGC1']].sum(1) > 0
dn = (e['CD4'] == 0) & (e['CD8A'] == 0) & (e['CD8B'] == 0)

gate = np.array('non_T', dtype=object).repeat(a.n_obs) if False else np.full(a.n_obs, 'non_T', dtype=object)
gate[cd3.values] = 'conv_T'
gate[cd3.values & dn.values & ~gd.values] = 'ab_DNT'
gate[cd3.values & gd.values] = 'gd_T'
a.obs['gate_v2'] = pd.Categorical(gate)
a.obs['trac_pos'] = e['TRAC'] > 0

cd3n = cd3.sum()
cnts = a.obs['gate_v2'].value_counts()
print('[gate v2]', dict(cnts))
print(f"ab_DNT = {cnts.get('ab_DNT',0)/cd3n*100:.2f}% of CD3+ | gd_T = {cnts.get('gd_T',0)/cd3n*100:.2f}%")
print(f"ab_DNT TRAC+ (confirmed): {(a.obs['gate_v2']=='ab_DNT').sum() and ((a.obs['gate_v2']=='ab_DNT') & a.obs['trac_pos']).sum()}")

# by histology
samp = a.obs['sample'].astype(str)
hist = np.where(samp.str.startswith('LGG'), 'LGG', np.where(samp.str.contains('rGBM'), 'rGBM', 'ndGBM'))
a.obs['histology'] = pd.Categorical(hist)
ct = pd.crosstab(a.obs['histology'], a.obs['gate_v2'], normalize='index') * 100
print('\n[composition by histology %]')
print(ct.round(2))
ct.to_csv(os.path.join(BASE, 'gate_v2_by_histology.csv'))
cnts.to_frame('n').to_csv(os.path.join(BASE, 'gate_v2_counts.csv'))

# UMAP fig
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
um = a.obsm['X_umap']
cats = a.obs['gate_v2'].astype(str)
cols = {'non_T': '#F0F0F0', 'conv_T': '#BBBBBB', 'ab_DNT': '#D65F5F', 'gd_T': '#2CA02C'}
for c in ['non_T', 'conv_T', 'ab_DNT', 'gd_T']:
    m = (cats == c).values
    axes[0].scatter(um[m, 0], um[m, 1], s=0.5, c=cols[c], label=f'{c} ({m.sum():,})')
axes[0].legend(markerscale=10, fontsize=9); axes[0].set_title('GSE182109 gate v2'); axes[0].set_xticks([]); axes[0].set_yticks([])
hcats = a.obs['histology'].astype(str)
hcols = {'ndGBM': '#1f77b4', 'rGBM': '#d62728', 'LGG': '#2ca02c'}
for c in ['ndGBM', 'rGBM', 'LGG']:
    m = (hcats == c).values
    axes[1].scatter(um[m, 0], um[m, 1], s=0.5, c=hcols[c], label=f'{c} ({m.sum():,})')
axes[1].legend(markerscale=10, fontsize=9); axes[1].set_title('Histology'); axes[1].set_xticks([]); axes[1].set_yticks([])
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig11_gse182109_umap_v2.png'), dpi=150); plt.close()

# overwrite gate column and save metadata only (h5ad already big)
a.obs.to_csv(os.path.join(BASE, 'cell_metadata_v2.csv'))
print('[done] 08c')
