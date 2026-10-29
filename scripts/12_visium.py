# -*- coding: utf-8 -*-
"""Module 8: Visium spatial colocalization (GSE235672 GBM Visium).
Score spots with scRNA-derived signatures -> Spearman colocalization + spatial maps."""
import scanpy as sc, numpy as np, pandas as pd, os, glob, re, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE163108')
VIS = os.path.join(os.path.dirname(__file__), '..', 'data', 'Visium')
FIGS = os.path.join(BASE, 'figs')
SIG = pd.read_csv(os.path.join(BASE, 'signature_matrix.csv'), index_col=0)  # ab_DNT, gd_T, conv_T
# add macrophage/tumor signatures (canonical)
SIG['Macrophage'] = 0.0
for g in ['LYZ', 'C1QA', 'C1QB', 'CD68', 'APOE', 'MSR1']:
    if g not in SIG.index: SIG.loc[g] = 0.0
    SIG.loc[g, 'Macrophage'] = 1.0
SIG['Tumor'] = 0.0
for g in ['EGFR', 'SOX2', 'OLIG2', 'PDGFRA', 'PTN', 'MEST']:
    if g not in SIG.index: SIG.loc[g] = 0.0
    SIG.loc[g, 'Tumor'] = 1.0

h5_files = sorted(glob.glob(os.path.join(VIS, '*_filtered_feature_bc_matrix.h5')))
print(f'[visium] {len(h5_files)} samples')
all_scores = []
for mf in h5_files:
    # e.g. GSM7507311_ST_Pt1_filtered_feature_bc_matrix.h5 / ..._ST_Pt5_Pt7_...
    stem = re.sub(r'_filtered_feature_bc_matrix\.h5$', '', os.path.basename(mf))
    gsm = stem.split('_')[0]                       # GSM7507311
    pid = re.sub(r'^' + re.escape(gsm) + r'_ST_', '', stem)  # Pt1 / Pt5_Pt7
    X = sc.read_10x_h5(mf)
    X.var['gene_symbols'] = X.var_names
    X.var_names_make_unique()
    pos_file = os.path.join(VIS, f'{gsm}_{pid}_tissue_positions_list.csv.gz')
    pos = pd.read_csv(pos_file)
    if pos.columns[0] != 'barcode':                # legacy headerless csv
        pos.columns = ['barcode', 'in_tissue', 'array_row', 'array_col',
                       'pxl_row_in_fullres', 'pxl_col_in_fullres']
    pos = pos.set_index('barcode')
    common_bc = [b for b in X.obs_names if b in pos.index]
    X = X[common_bc, :]
    # normalize + score
    sc.pp.normalize_total(X, target_sum=1e4); sc.pp.log1p(X)
    scores = pd.DataFrame(index=X.obs_names)
    for sig in ['ab_DNT', 'gd_T', 'conv_T', 'Macrophage', 'Tumor']:
        gs = [g for g in SIG.index[SIG[sig] > 0] if g in X.var_names]
        sc.tl.score_genes(X, gs, score_name=sig)
        scores[sig] = X.obs[sig].values
    scores['sample'] = pid
    scores['row'] = pos.loc[X.obs_names, 'array_row'].values
    scores['col'] = pos.loc[X.obs_names, 'array_col'].values
    scores['imagecol'] = pos.loc[X.obs_names, 'pxl_col_in_fullres'].values
    scores['imagerow'] = pos.loc[X.obs_names, 'pxl_row_in_fullres'].values
    all_scores.append(scores)
    print(f'  {pid}: {X.shape[0]} spots', flush=True)
S = pd.concat(all_scores)
S.to_csv(os.path.join(BASE, 'visium_spot_scores.csv'), index=False)

# colocalization (Spearman per sample, then mean)
cors = {}
for sm, g in S.groupby('sample'):
    for a, b in [('ab_DNT', 'Macrophage'), ('ab_DNT', 'Tumor'), ('gd_T', 'Macrophage'),
                 ('gd_T', 'Tumor'), ('conv_T', 'Macrophage'), ('ab_DNT', 'gd_T')]:
        if g[a].std() > 0 and g[b].std() > 0:
            rho, p = stats.spearmanr(g[a], g[b])
            cors.setdefault(a + '|' + b, []).append((sm, rho, p))
rows = []
for k, v in cors.items():
    dfv = pd.DataFrame(v, columns=['sample', 'rho', 'p'])
    rows.append({'pair': k, 'mean_rho': dfv.rho.mean(), 'n_pos': (dfv.p < 0.05).sum(),
                 'n_samples': len(dfv)})
ctab = pd.DataFrame(rows).sort_values('mean_rho', ascending=False)
print(ctab.round(3).to_string())
ctab.to_csv(os.path.join(BASE, 'visium_colocalization.csv'), index=False)

# spatial maps for one sample (median ab_DNT score sample)
sm = S.groupby('sample')['ab_DNT'].median().idxmax()
g = S[S['sample'] == sm]
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
for ax, sig in zip(axes, ['ab_DNT', 'gd_T', 'conv_T', 'Macrophage', 'Tumor']):
    scv = ax.scatter(g.imagecol, -g.imagerow, c=g[sig], s=6, cmap='magma')
    plt.colorbar(scv, ax=ax, label=sig)
    ax.set_title(f'{sm}\n{sig}'); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig16_visium_maps.png'), dpi=150); plt.close()
print('[done] module 8')
