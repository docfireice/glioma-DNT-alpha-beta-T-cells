# -*- coding: utf-8 -*-
"""Module 6: ligand-receptor communication (LIANA-style) on GSE163108.
1) annotate non-T clusters by canonical markers
2) canonical LR pairs -> aggregate score = mean_ligand(A) * mean_receptor(B)
3) rank interactions to/from ab_DNT & gd_T; heatmap + csv."""
import scanpy as sc, anndata as ad, numpy as np, pandas as pd, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE163108')
FIGS = os.path.join(BASE, 'figs')
adata = sc.read_h5ad(os.path.join(BASE, 'gse163108_processed.h5ad'))
adata.obs_names_make_unique()
print('[load]', adata.shape)

MARK = {
 'Tumor':   ['EGFR', 'SOX2', 'OLIG2', 'PDGFRA', 'PTN', 'MEST'],
 'Macrophage': ['LYZ', 'C1QA', 'C1QB', 'CD68', 'AIF1', 'FCGR3A', 'MSR1'],
 'Endothelial': ['PECAM1', 'VWF', 'CLDN5'],
 'Oligodendrocyte': ['MBP', 'PLP1', 'MOG', 'MAG'],
 'NK': ['GNLY', 'KLRD1', 'NKG7', 'NCAM1'],
}
genes_all = set(adata.var_names)
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
clusters = adata.obs['leiden'].astype(str)
# mean expression per cluster for marker genes
means = {}
for ct, gl in MARK.items():
    gs = [g for g in gl if g in genes_all]
    sub = adata[:, gs].X
    cm = np.asarray(sub.toarray() if hasattr(sub, 'toarray') else sub).mean(1)
    means[ct] = pd.Series(cm).groupby(clusters.values).mean()
mk = pd.DataFrame(means)
# gate-based T annotation per cluster: majority gate among T clusters
gatemap = pd.crosstab(clusters, adata.obs['gate'].astype(str), normalize='index')
annot = mk.idxmax(1).where(mk.max(1) > 0.5)
tfrac = gatemap[['ab_DNT', 'conv_T', 'gd_T']].sum(1)
for c in annot.index[annot.isna()]:
    if tfrac[c] > 0.3:
        annot[c] = 'Tcell_' + gatemap.loc[c, ['ab_DNT', 'conv_T', 'gd_T']].idxmax()
    else:
        annot[c] = 'Other'
adata.obs['celltype'] = annot[clusters.values].values
print(adata.obs['celltype'].value_counts())

# canonical LR pairs (ligand, receptor)
LR = [
 ('CXCL13', 'CXCR5'), ('CCL5', 'CCR5'), ('CCL4', 'CCR5'), ('CCL3', 'CCR5'),
 ('CXCL9', 'CXCR3'), ('CXCL10', 'CXCR3'), ('CXCL16', 'CXCR6'),
 ('CCL2', 'CCR2'), ('CCL22', 'CCR4'), ('CXCL12', 'CXCR4'),
 ('MIF', 'CD74'), ('SPP1', 'CD44'), ('SPP1', 'ITGAV'),
 ('LGALS9', 'HAVCR2'), ('CD274', 'PDCD1'), ('PDCD1LG2', 'PDCD1'),
 ('CD86', 'CTLA4'), ('CD80', 'CTLA4'), ('CD80', 'CD28'), ('CD86', 'CD28'),
 ('ICAM1', 'ITGAL'), ('VCAM1', 'ITGA4'), ('CD70', 'CD27'), ('TNFSF14', 'LTBR'),
 ('VEGFA', 'KDR'), ('VEGFA', 'FLT1'), ('PDGFB', 'PDGFRB'),
 ('IL16', 'CD4'), ('IL15', 'IL2RB'), ('IL6', 'IL6R'), ('IL10', 'IL10RA'),
 ('TGFB1', 'TGFBR1'), ('C5AR1_R', 'C5AR1'), ('HMGB1', 'AGER'),
 ('PFN1_R', 'CD74'), ('TGM2', 'CD44'), ('CD48', 'CD2'), ('CD58', 'CD2'),
 ('NECTIN2', 'TIGIT'), ('PVR', 'TIGIT'), ('PVR', 'CD226'),
 ('HLA-E_R', 'KLRD1'), ('B2M_R', 'KLRK1'), ('ULBP1', 'KLRK1'),
 ('FASLG', 'FAS'), ('TNFSF10', 'TNFRSF10B'), ('LAMP1_R', 'LAG3'),
 ('HLA-DRA_R', 'LAG3'), ('SIGLEC1_R', 'CD33'),
]
pairs = [(l, r) for l, r in LR if l in genes_all and r in genes_all]
print(f'[LR] usable pairs: {len(pairs)}/{len(LR)}')

def mean_expr(g):
    col = adata[:, g].X
    v = np.asarray(col.toarray() if hasattr(col, 'toarray') else col).ravel()
    return pd.Series(v).groupby(adata.obs['celltype'].values).mean()

celltypes = [c for c in adata.obs['celltype'].unique()]
L = pd.DataFrame({g: mean_expr(g) for g, _ in pairs})
R = pd.DataFrame({g: mean_expr(g) for _, g in pairs})
score = np.outer(0, 0)  # placeholder
S = pd.DataFrame(index=celltypes, columns=celltypes, dtype=float)
inter = []
for (l, r) in pairs:
    for a in celltypes:
        for b in celltypes:
            inter.append({'ligand': l, 'receptor': r, 'source': a, 'target': b,
                          'score': L.loc[a, l] * R.loc[b, r]})
I = pd.DataFrame(inter)
I.to_csv(os.path.join(BASE, 'LR_all_pairs.csv'), index=False)

# focus: sources/targets involving DNT/gdT
focus_t = ['ab_DNT', 'gd_T']
out = I[(I.source.isin(focus_t)) | (I.target.isin(focus_t))]
out = out.sort_values('score', ascending=False)
out.to_csv(os.path.join(BASE, 'LR_dnt_gdt_ranked.csv'), index=False)
print('[top outgoing]')
print(out[out.source.isin(focus_t)].head(12).round(3))
print('[top incoming]')
print(out[out.target.isin(focus_t)].head(12).round(3))

# heatmap: mean LR score between cell types (aggregate all pairs)
agg = I.groupby(['source', 'target'])['score'].mean().unstack()
plt.figure(figsize=(8, 6.5))
sns.heatmap(agg, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': 'mean LR score'})
plt.title('Ligand-receptor communication (all canonical pairs)')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig13_LR_heatmap.png'), dpi=150); plt.close()

# DNT/gdT-specific top pairs heatmap
top = out.head(40)
pv = top.pivot_table(index=['source', 'ligand'], columns=['target', 'receptor'], values='score')
plt.figure(figsize=(10, 8))
sns.heatmap(pv, cmap='magma', annot=True, fmt='.2f', cbar_kws={'label': 'LR score'})
plt.title('Top DNT/gdT interactions')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig14_LR_top_pairs.png'), dpi=150); plt.close()
print('[done] module 6')
