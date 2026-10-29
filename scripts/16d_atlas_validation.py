# -*- coding: utf-8 -*-
"""Module 10d: external validation on the 66-sample brain tumour single-cell atlas
(CELLxGENE; IDH-wt GBM + IDH-mut astrocytoma/oligodendroglioma, grades 2-4).
Immune-cell subset -> v2-aligned gating -> per-donor DNT/gd-T fractions by
grade/disease + SPP1/MIF-CD44/CD74 axis + signature transfer AUROC.
Outputs: fig19c/fig20c/fig22b, results/CELLxGENE/atlas_*.{csv,txt}
"""
import os, gc
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'CELLxGENE')
FIGS = os.path.join(BASE, 'figs')
os.makedirs(FIGS, exist_ok=True)
PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'CELLxGENE', 'glioma_atlas_66s.h5ad')

GD = ['TRDC', 'TRDV1', 'TRDV2', 'TRGC1']
TAM_MARKS = ['CD68', 'LYZ', 'CD14', 'C1QA', 'C1QB', 'C1QC', 'FCGR3A']

print('[1] load (backed) + subset immune cells', flush=True)
a = sc.read_h5ad(PATH, backed='r')
ct = a.obs['cell_type'].astype(str)
keep_ct = ct.str.contains('T cell|T lymphocyte|natural killer') | \
          ct.str.contains('macrophage|microglial|myeloid|dendritic|monocyte')
idx = np.where(keep_ct.values)[0]
print(f'   immune subset: {len(idx)} / {a.n_obs}', flush=True)
ad = a[idx].to_memory()
del a; gc.collect()

# gene symbols
if str(ad.var_names[0]).startswith('ENSG'):
    sym = ad.var['feature_name'].astype(str).values
    _, first = np.unique(sym, return_index=True)
    first = np.sort(first)
    ad = ad[:, first].copy()
    ad.var_names = pd.Index(sym[first])
ad.X = sp.csr_matrix(ad.X) if not sp.issparse(ad.X) else ad.X.tocsr()
print(f'   subset {ad.shape} | X max {ad.X.data.max():.1f}', flush=True)
if ad.X.data.max() > 200:
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
vidx = {g: i for i, g in enumerate(ad.var_names)}
n = ad.n_obs

def pos(g):
    if g not in vidx:
        return np.zeros(n, bool)
    return ad.X[:, vidx[g]].toarray().ravel() > 0

def vec(g):
    if g not in vidx:
        return np.zeros(n, np.float32)
    return ad.X[:, vidx[g]].toarray().ravel()

print('[2] gating (v2-aligned)', flush=True)
t = pos('CD3D') | pos('CD3E')
gdc = np.zeros(n, int)
for g in GD:
    gdc += pos(g).astype(int)
gd = (gdc > 0) & t
gd_strict = (gdc >= 2) & t & ~pos('TRAC')
cd4, cd8, cd8b, trac = pos('CD4'), pos('CD8A'), pos('CD8B'), pos('TRAC')
ab = t & ~gd & ~cd4 & ~cd8 & ~cd8b
ab_strict = ab & trac
conv = t & ~gd & (cd4 | cd8 | cd8b)
tam = np.zeros(n, bool)
for g in TAM_MARKS:
    tam |= pos(g)
tam = tam & ~t

gate = np.array(['other'] * n, dtype=object)
gate[tam] = 'TAM'; gate[conv] = 'conv_T'; gate[gd] = 'gd_T'; gate[ab] = 'ab_DNT'
ad.obs['gate'] = gate
ad.obs['is_T'] = t

print(f'   T {t.sum()} | ab_DNT {ab.sum()} (strict {ab_strict.sum()}) | gd_T {gd.sum()} '
      f'(strict {gd_strict.sum()}) | conv_T {conv.sum()} | TAM {tam.sum()}', flush=True)

# QC sensitivity: double-negative calls are inflated by shallow/dropout-heavy cells
qc_ok = np.ones(n, bool)
if 'nFeature_RNA' in ad.obs:
    qc_ok &= (ad.obs['nFeature_RNA'].values > 500)
if 'percent.mt' in ad.obs:
    qc_ok &= (ad.obs['percent.mt'].values < 20)
tq = t & qc_ok
abq = ab & qc_ok
gdq = gd & qc_ok
print(f'   [QC-filtered] T {tq.sum()} | ab_DNT {abq.sum()} ({100*abq.sum()/max(tq.sum(),1):.1f}%) '
      f'| gd_T {gdq.sum()} ({100*gdq.sum()/max(tq.sum(),1):.1f}%)', flush=True)

# per-donor table
meta = ad.obs.copy()
meta['donor'] = meta['donor_id'].astype(str)
rows = []
for dnr, gdf in meta[t].groupby('donor'):
    tn = len(gdf)
    row = {'donor': dnr, 'T_cells': tn,
           'ab_DNT': int((gdf.gate == 'ab_DNT').sum()),
           'gd_T': int((gdf.gate == 'gd_T').sum()),
           'conv_T': int((gdf.gate == 'conv_T').sum())}
    for col in ['grade', 'IDH_status', 'disease', 'Primary_or_Recurrent']:
        if col in gdf:
            row[col] = gdf[col].astype(str).mode()[0]
    rows.append(row)
per = pd.DataFrame(rows)
per['ab_DNT_pct'] = 100 * per['ab_DNT'] / per['T_cells']
per['gd_T_pct'] = 100 * per['gd_T'] / per['T_cells']
per.to_csv(os.path.join(BASE, 'atlas_per_donor_T_fractions.csv'), index=False)
print(per.groupby('grade')[['ab_DNT_pct', 'gd_T_pct']].agg(['count', 'median']).round(2).to_string(), flush=True)

# grade / disease comparisons (CD3+ fractions)
lab = per['grade'].astype(str)
kw1 = stats.kruskal(*[per.loc[lab == g, 'ab_DNT_pct'].values for g in sorted(per.grade.astype(str).unique())])
kw2 = stats.kruskal(*[per.loc[lab == g, 'gd_T_pct'].values for g in sorted(per.grade.astype(str).unique())])
print(f'[3] ab_DNT_pct across grades KW p={kw1.pvalue:.4f} | gd_T_pct KW p={kw2.pvalue:.4f}', flush=True)

# communication axis
tam_mask = gate == 'TAM'
t_mask = t
lr_stats = {'TAM_SPP1_pos': float((vec('SPP1')[tam_mask] > 0).mean()),
            'TAM_MIF_pos': float((vec('MIF')[tam_mask] > 0).mean()),
            'T_CD44_pos': float((vec('CD44')[t_mask] > 0).mean()),
            'T_CD74_pos': float((vec('CD74')[t_mask] > 0).mean())}
print('[4] LR axis:', {k: round(v, 3) for k, v in lr_stats.items()}, flush=True)

# signature transfer (ab_DNT-specific genes from module 10c)
print('[5] signature AUROC', flush=True)
auc = np.nan
try:
    axx = sc.read_h5ad(os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE163108',
                                    'gse163108_processed.h5ad'))
    axx = axx[~axx.obs['gate'].isin(['non_T'])].copy()
    sc.tl.rank_genes_groups(axx, 'gate', groups=['ab_DNT'], reference='conv_T', method='wilcoxon')
    r = axx.uns['rank_genes_groups']
    dfg = pd.DataFrame({'gene': np.array(r['names']['ab_DNT']).astype(str),
                        'logfc': np.asarray(r['logfoldchanges']['ab_DNT'], dtype=float),
                        'padj': np.asarray(r['pvals_adj']['ab_DNT'], dtype=float)})
    dnt_genes = dfg[(dfg.padj < 0.05) & (dfg.logfc > 0.5)].gene.tolist()[:100]
    del axx; gc.collect()
    gl = [str(g) for g in dnt_genes if str(g) in vidx]
    print(f'   DE genes {len(dnt_genes)}, in-atlas {len(gl)}', flush=True)
    if len(gl) >= 10:
        sc.tl.score_genes(ad, gl, score_name='dnt_sig', use_raw=False)
        m1 = gate == 'ab_DNT'; m0 = gate == 'conv_T'
        auc = roc_auc_score(np.r_[np.ones(m1.sum()), np.zeros(m0.sum())],
                            np.r_[ad.obs.loc[m1, 'dnt_sig'].values, ad.obs.loc[m0, 'dnt_sig'].values])
        print(f'   ab_DNT sig AUROC in atlas = {auc:.3f}', flush=True)
    else:
        print('   [skip] too few transferable genes', flush=True)
except Exception as e:
    print('   [skip] AUROC failed:', e, flush=True)

with open(os.path.join(BASE, 'atlas_validation_stats.txt'), 'w') as f:
    f.write(f'T_cells={int(t.sum())} ab_DNT={int(ab.sum())} gd_T={int(gd.sum())} '
            f'conv_T={int(conv.sum())} TAM={int(tam.sum())}\n')
    f.write(f'ab_DNT_strict={int(ab_strict.sum())} gd_T_strict={int(gd_strict.sum())}\n')
    f.write(f'QC: T={int(tq.sum())} ab_DNT={int(abq.sum())} gd_T={int(gdq.sum())}\n')
    f.write(f'ab_DNT_pct_grade_KW_p={kw1.pvalue:.5f}\ngd_T_pct_grade_KW_p={kw2.pvalue:.5f}\n')
    f.write('LR_axis: ' + str({k: round(v, 3) for k, v in lr_stats.items()}) + '\n')
    f.write(f'ab_DNT_sig_AUROC={auc:.4f}\n')

# ---------- fig19c: per-donor fractions ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, col, colr in [(axes[0], 'ab_DNT_pct', '#C44E52'), (axes[1], 'gd_T_pct', '#55A868')]:
    data = [per.loc[per.grade.astype(str) == g, col].values for g in ['2', '3', '4']]
    bp = ax.boxplot(data, tick_labels=['grade 2', 'grade 3', 'grade 4 (GBM)'])
    for i in range(len(data)):
        ax.scatter(np.random.normal(i + 1, 0.06, len(data[i])), data[i], s=10,
                   alpha=0.6, color=colr, zorder=3)
    ax.set_ylabel(f'{col} (% of CD3+)')
ax = axes[0]
p = stats.kruskal(*[per.loc[per.grade.astype(str) == g, 'ab_DNT_pct'].values for g in ['2', '3', '4']])
ax.set_title(f'ab_DNT fraction by WHO grade (KW p={p.pvalue:.3f})')
ax = axes[1]
p = stats.kruskal(*[per.loc[per.grade.astype(str) == g, 'gd_T_pct'].values for g in ['2', '3', '4']])
ax.set_title(f'gd_T fraction by WHO grade (KW p={p.pvalue:.3f})')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig19c_atlas_fractions.png'), dpi=150); plt.close()

# ---------- fig20c: LR axis in atlas ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, lig, rec, ttl in [(axes[0], 'SPP1', 'CD44', 'SPP1-CD44 axis'),
                          (axes[1], 'MIF', 'CD74', 'MIF-CD74 axis')]:
    vl, vr = vec(lig), vec(rec)
    ax.boxplot([vl[tam_mask], vr[t_mask]], tick_labels=[f'TAM {lig} (n={tam_mask.sum()})',
                                                        f'T {rec} (n={t_mask.sum()})'],
               showfliers=False)
    ax.set_title(ttl)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig20c_atlas_LR.png'), dpi=150); plt.close()

# per-donor LR expression summary
obs_pos = pd.Series(np.arange(n), index=ad.obs_names)
rows = []
for dnr, gdf in meta[tam_mask].groupby('donor'):
    pos_idx = obs_pos.loc[gdf.index].values
    rows.append({'donor': dnr, 'n_TAM': len(gdf),
                 'SPP1_pos': float((vec('SPP1')[pos_idx] > 0).mean()),
                 'MIF_pos': float((vec('MIF')[pos_idx] > 0).mean())})
pd.DataFrame(rows).to_csv(os.path.join(BASE, 'atlas_per_donor_TAM_LR.csv'), index=False)

print('[done] module 10d', flush=True)
