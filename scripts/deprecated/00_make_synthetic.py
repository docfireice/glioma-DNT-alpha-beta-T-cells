# -*- coding: utf-8 -*-
"""Generate synthetic GBM T-cell scRNA-seq data for pipeline smoke test.
Simulates CD4/CD8/gamma-delta/alpha-beta DNT populations with marker structure.
Output: data/raw/synthetic_gbm_tcells.csv (cells x genes) + true labels.
"""
import numpy as np, pandas as pd, os, sys

rng = np.random.default_rng(42)
out = sys.argv[1] if len(sys.argv) > 1 else 'data/raw'
os.makedirs(out, exist_ok=True)

genes = (['CD3D','CD3E','CD3G','CD4','CD8A','CD8B','TRDC','TRDV1','TRDV2',
          'FOXP3','IL2RA','PDCD1','CXCL13','CD40LG','CCR6','RORC','LEF1','CCR7',
          'RGS1','CREM','CXCL1','CXCL8','IFNG','GZMK','GZMB','HLA-DRA','HLA-DRB1',
          'ITGA1','CD69','ITGAE','RUNX3','CX3CR1','PTGDS','FGFBP2','TBX21',
          'NR4A1','MKI67','MS4A1','CD14','LYZ','EPCAM','MT-CO1','MT-ND2']
         + [f'G{i}' for i in range(200)])
n = {'CD4_T': 400, 'CD8_T': 400, 'gd_T': 300, 'ab_DNT': 250, 'B_like': 60, 'Myeloid_like': 50}

def base_cell():
    return rng.poisson(0.4, len(genes))

rows, labels = [], []
for ct, n_cells in n.items():
    markers = {
        'CD4_T': ['CD3D','CD3E','CD3G','CD4'], 'CD8_T': ['CD3D','CD3E','CD3G','CD8A'],
        'gd_T': ['CD3D','CD3E','CD3G','TRDC','TRDV1','TRDV2','IFNG','GZMK'],
        'ab_DNT': ['CD3D','CD3E','CD3G','FOXP3','IL2RA','RGS1','CREM'],
        'B_like': ['MS4A1'], 'Myeloid_like': ['CD14','LYZ'],
    }[ct]
    midx = [genes.index(g) for g in markers]
    for _ in range(n_cells):
        x = base_cell()
        for j in midx:
            x[j] += rng.poisson(8)
        rows.append(x); labels.append(ct)

X = pd.DataFrame(np.vstack(rows), columns=genes)
X.index = [f'cell_{i}' for i in range(len(X))]
# CD4/CD8 knockout for DNT/gdT realism (CD3+CD4-CD8-)
labels = np.array(labels)
knock = np.where(np.isin(labels, ['gd_T', 'ab_DNT']))[0]
X.iloc[knock, X.columns.get_indexer(['CD4', 'CD8A', 'CD8B'])] = 0
X.to_csv(os.path.join(out, 'synthetic_gbm_tcells.csv'))
pd.Series(list(labels), index=X.index, name='true_label').to_csv(os.path.join(out, 'synthetic_labels.csv'))
print(f'written: {X.shape[0]} cells x {X.shape[1]} genes ->', os.path.join(out,'synthetic_gbm_tcells.csv'))
