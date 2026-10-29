# -*- coding: utf-8 -*-
"""Module 6b: LR communication on GSE182109 (full tumors).
Annotate 32 clusters -> groups {Tumor, Macrophage, Endothelial, Oligodendrocyte, Astrocyte, NK, conv_T, ab_DNT, gd_T}
-> canonical LR pair scores (mean ligand(A) x mean receptor(B))."""
import scanpy as sc, anndata as ad, numpy as np, pandas as pd, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE182109')
FIGS = os.path.join(BASE, 'figs')
adata = sc.read_h5ad(os.path.join(BASE, 'gse182109_processed.h5ad'))
print('[load]', adata.shape)
md = pd.read_csv(os.path.join(BASE, 'cell_metadata_v2.csv'), index_col=0)
md = md.loc[adata.obs_names]
adata.obs['gate_v2'] = md['gate_v2'].values
adata.obs['histology'] = md['histology'].values

MARK = {
 'Macrophage': ['LYZ', 'C1QA', 'C1QB', 'CD68', 'AIF1', 'MSR1', 'APOE'],
 'Tumor': ['EGFR', 'SOX2', 'OLIG2', 'PDGFRA', 'PTN', 'MEST', 'SOX9'],
 'Endothelial': ['PECAM1', 'VWF', 'CLDN5'],
 'Oligodendrocyte': ['MBP', 'PLP1', 'MOG', 'MAG'],
 'Astrocyte': ['GFAP', 'AQP4', 'SLC1A3'],
 'Bcell': ['MS4A1', 'CD79A', 'MZB1'],
}
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
clusters = adata.obs['leiden'].astype(str)
means = {}
for ct, gl in MARK.items():
    gs = [g for g in gl if g in adata.var_names]
    sub = adata[:, gs].X
    cm = np.asarray(sub.toarray() if hasattr(sub, 'toarray') else sub).mean(1)
    means[ct] = pd.Series(cm).groupby(clusters.values).mean()
mk = pd.DataFrame(means)
annot = mk.idxmax(1)
for c in annot.index:
    sub = adata.obs['gate_v2'].astype(str)[clusters == c].value_counts(normalize=True)
    tfrac = sub.get('ab_DNT', 0) + sub.get('conv_T', 0) + sub.get('gd_T', 0)
    if tfrac > 0.4 and sub.get('ab_DNT', 0) > sub.get('conv_T', 0):
        annot[c] = 'ab_DNT'
    elif tfrac > 0.4 and sub.get('gd_T', 0) > 0.1:
        annot[c] = 'gd_T'
    elif tfrac > 0.4:
        annot[c] = 'conv_T'
adata.obs['celltype'] = annot[clusters.values].values
print(adata.obs['celltype'].value_counts())

LR = [
 ('CXCL13', 'CXCR5'), ('CCL5', 'CCR5'), ('CCL4', 'CCR5'), ('CCL3', 'CCR5'),
 ('CXCL9', 'CXCR3'), ('CXCL10', 'CXCR3'), ('CXCL16', 'CXCR6'),
 ('CCL2', 'CCR2'), ('CXCL12', 'CXCR4'),
 ('MIF', 'CD74'), ('SPP1', 'CD44'), ('SPP1', 'ITGAV'),
 ('LGALS9', 'HAVCR2'), ('CD274', 'PDCD1'), ('PDCD1LG2', 'PDCD1'),
 ('CD86', 'CTLA4'), ('CD80', 'CTLA4'), ('CD80', 'CD28'), ('CD86', 'CD28'),
 ('ICAM1', 'ITGAL'), ('VCAM1', 'ITGA4'), ('CD70', 'CD27'),
 ('VEGFA', 'KDR'), ('VEGFA', 'FLT1'), ('PDGFB', 'PDGFRB'),
 ('IL15', 'IL2RB'), ('IL6', 'IL6R'), ('IL10', 'IL10RA'),
 ('TGFB1', 'TGFBR1'), ('HMGB1', 'AGER'), ('TGM2', 'CD44'),
 ('CD48', 'CD2'), ('CD58', 'CD2'), ('NECTIN2', 'TIGIT'), ('PVR', 'TIGIT'),
 ('FASLG', 'FAS'), ('TNFSF10', 'TNFRSF10B'), ('CD226_R', 'CD226'),
 ('HLA-E_R', 'KLRD1'), ('ULBP1', 'KLRK1'),
]
pairs = [(l, r) for l, r in LR if l in adata.var_names and r in adata.var_names]
print(f'[LR] usable pairs: {len(pairs)}/{len(LR)}')

def mean_expr(g):
    col = adata[:, g].X
    v = np.asarray(col.toarray() if hasattr(col, 'toarray') else col).ravel()
    return pd.Series(v).groupby(adata.obs['celltype'].values).mean()

celltypes = sorted(adata.obs['celltype'].unique())
L = pd.DataFrame({g: mean_expr(g) for g, _ in pairs})
R = pd.DataFrame({g: mean_expr(g) for _, g in pairs})
inter = []
for (l, r) in pairs:
    for a in celltypes:
        for b in celltypes:
            inter.append({'ligand': l, 'receptor': r, 'source': a, 'target': b,
                          'score': L.loc[a, l] * R.loc[b, r]})
I = pd.DataFrame(inter)
I.to_csv(os.path.join(BASE, 'LR_all_pairs.csv'), index=False)

focus = ['ab_DNT', 'gd_T']
out = I[I.source.isin(focus) | I.target.isin(focus)].sort_values('score', ascending=False)
out.to_csv(os.path.join(BASE, 'LR_dnt_gdt_ranked.csv'), index=False)
print('[top outgoing]')
print(out[out.source.isin(focus)].head(12).round(3).to_string())
print('[top incoming]')
print(out[out.target.isin(focus)].head(12).round(3).to_string())

agg = I.groupby(['source', 'target'])['score'].mean().unstack()
plt.figure(figsize=(9, 7))
sns.heatmap(agg, cmap='viridis', annot=True, fmt='.2f')
plt.title('GSE182109: mean LR communication score')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig13_LR_heatmap.png'), dpi=150); plt.close()

top = out.head(40)
pv = top.pivot_table(index=[top.source, top.ligand], columns=[top.target, top.receptor], values='score')
plt.figure(figsize=(11, 8))
sns.heatmap(pv, cmap='magma', annot=True, fmt='.2f')
plt.title('Top DNT/gdT ligand-receptor interactions')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig14_LR_top_pairs.png'), dpi=150); plt.close()
print('[done] module 6b')
