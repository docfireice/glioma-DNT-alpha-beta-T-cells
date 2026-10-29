# -*- coding: utf-8 -*-
"""Make figures for GSE163108 DNT analysis."""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'results/GSE163108'
FIG = 'results/GSE163108/figs'
os.makedirs(FIG, exist_ok=True)

sub = sc.read_h5ad(os.path.join(OUT, 'dnt_gdt_subset.h5ad'))
print(sub)

# Fig 1: UMAP by subcluster and gate
try:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sc.pl.umap(sub, color='dnt_sub', ax=axes[0], show=False, legend_loc='on data',
               title='DNT/gdT subclusters')
    sc.pl.umap(sub, color='gate', ax=axes[1], show=False, title='Gate')
    plt.tight_layout(); plt.savefig(f'{FIG}/fig1_umap_subset.png', dpi=200); plt.close()
    print('[fig] fig1 ok')
except Exception as e:
    print('[fig1] failed:', e)

# Fig 2: marker heatmap of subclusters
try:
    sc.tl.rank_genes_groups(sub, 'dnt_sub', method='wilcoxon')
    markers = [g for g in ['CD4','CD8A','FOXP3','IL2RA','PDCD1','CXCL13','CCR7','SELL','LEF1',
                           'GZMK','GZMB','NKG7','IFNG','HLA-DRA','HLA-DRB1','CD74','ITGA1','ITGAE',
                           'CX3CR1','MKI67','TRDC','TRDV1','TRDV2','TRGC1']
               if g in sub.var_names]
    sc.pl.dotplot(sub, markers, groupby='dnt_sub', standard_scale='var',
                  save='_dnt_markers.png', show=False)
    os.replace(f'figures/dotplot__dnt_markers.png', f'{FIG}/fig2_dotplot_markers.png')
    print('[fig] fig2 ok')
except Exception as e:
    print('[fig2] failed:', e)

# Fig 3: signature scores per subcluster
score_cols = ['cytotoxicity', 'IFN_gamma', 'exhaustion', 'naive_memory', 'Treg',
              'MHCII_APC', 'proliferation', 'tissue_residency', 'gd_signature']
score_cols = [c for c in score_cols if c in sub.obs.columns]
if score_cols:
    tab = sub.obs.groupby('dnt_sub')[score_cols].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(tab.values, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(score_cols))); ax.set_xticklabels(score_cols, rotation=45, ha='right')
    ax.set_yticks(range(len(tab))); ax.set_yticklabels(tab.index)
    ax.set_title('Signature scores by DNT subcluster (GSE163108)')
    plt.colorbar(im, ax=ax, label='mean score')
    plt.tight_layout(); plt.savefig(f'{FIG}/fig3_signature_heatmap.png', dpi=200); plt.close()
    tab.round(3).to_csv(f'{OUT}/subcluster_signature_scores.csv')
    print('[fig] fig3 ok')

# Fig 4: sample composition per subcluster
ct = pd.crosstab(sub.obs['dnt_sub'], sub.obs['sample'], normalize='index')
fig, ax = plt.subplots(figsize=(7, 4))
ct.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
ax.set_ylabel('fraction'); ax.set_title('Sample composition of DNT subclusters')
plt.tight_layout(); plt.savefig(f'{FIG}/fig4_sample_composition.png', dpi=200); plt.close()
print('[fig] fig4 ok')
