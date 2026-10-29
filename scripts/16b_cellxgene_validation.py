# -*- coding: utf-8 -*-
"""Module 10b: external validation on glioma scRNA cohorts hosted by CELLxGENE
(Couturier 2020 SL040/SL057 and the 66-sample brain tumour atlas).
Replicates DNT / gamma-delta T gating and the SPP1-CD44 / MIF-CD74
macrophage->T communication axis in independent patients.
Memory-lean: X is log-normalised in place; gating uses sparse column slices.
Outputs: fig19_cellxgene_*, fig20_cellxgene_*, results/CELLxGENE/*.csv
"""
import os, gc
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'CELLxGENE')
DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'CELLxGENE')
FIGS = os.path.join(BASE, 'figs')
os.makedirs(FIGS, exist_ok=True)

GD_MARKERS = ['TRDC', 'TRDV1', 'TRDV2', 'TRGC1', 'TRDV3']
TAM_MARKERS = ['CD68', 'LYZ', 'CD14', 'C1QA', 'C1QB', 'C1QC', 'FCGR3A']

sig = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE163108',
                               'signature_matrix.csv'), index_col=0)


def analyse(path, tag):
    print(f'[{tag}] load {os.path.basename(path)}', flush=True)
    ad = sc.read_h5ad(path)
    print(f'   {ad.shape} | X max {ad.X.max():.1f}', flush=True)
    if str(ad.var_names[0]).startswith('ENSG'):
        sym = ad.var['feature_name'].astype(str).values
        _, first = np.unique(sym, return_index=True)
        first = np.sort(first)
        ad = ad[:, first].copy()
        ad.var_names = pd.Index(sym[first])
        print(f'   mapped to symbols: {ad.shape}', flush=True)
    ad.X = sp.csr_matrix(ad.X) if not sp.issparse(ad.X) else ad.X.tocsr()
    if ad.X.data.max() > 200:                       # raw counts -> log-normalise in place
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    vidx = {g: i for i, g in enumerate(ad.var_names)}
    n = ad.n_obs

    def vec(g):
        if g not in vidx:
            return np.zeros(n, np.float32)
        return ad.X[:, vidx[g]].toarray().ravel()

    def pos(g):
        return vec(g) > 0

    t = pos('CD3D') | pos('CD3E')
    gd_count = np.zeros(n, int)
    for g in GD_MARKERS:
        gd_count += pos(g).astype(int)
    cd4, cd8, cd8b = pos('CD4'), pos('CD8A'), pos('CD8B')
    trac = pos('TRAC')
    gd = (gd_count > 0) & t                                  # permissive (v2 definition)
    gd_strict = (gd_count >= 2) & t & ~trac                  # sensitivity: >=2 gamma-delta TR
    ab_dnt = t & ~gd & ~cd4 & ~cd8 & ~cd8b                  # v2 gating: also CD8B-negative
    ab_dnt_strict = ab_dnt & trac                           # TRAC-confirmed alpha-beta DNT
    conv = t & ~gd & (cd4 | cd8 | cd8b)
    tam = np.zeros(n, bool)
    for g in TAM_MARKERS:
        tam |= pos(g)
    tam = tam & ~t

    gate = np.array(['other'] * n, dtype=object)
    gate[tam] = 'TAM'
    gate[conv] = 'conv_T'
    gate[gd] = 'gd_T'
    gate[ab_dnt] = 'ab_DNT'
    ad.obs['gate'] = gate
    ad.obs['trac_pos'] = trac

    for g in ['SPP1', 'MIF', 'CD44', 'CD74']:
        ad.obs[f'{g}_expr'] = vec(g)

    genes = set(ad.var_names)
    tm, tt = gate == 'TAM', t
    st = {
        'cohort': tag, 'T_cells': int(t.sum()),
        'ab_DNT': int(ab_dnt.sum()), 'gd_T': int(gd.sum()), 'conv_T': int(conv.sum()),
        'ab_DNT_strict': int(ab_dnt_strict.sum()), 'gd_T_strict': int(gd_strict.sum()),
        'TAM_n': int(tm.sum()),
        'TAM_SPP1_pos': float((ad.obs.loc[tm, 'SPP1_expr'] > 0).mean()) if tm.sum() else np.nan,
        'TAM_MIF_pos': float((ad.obs.loc[tm, 'MIF_expr'] > 0).mean()) if tm.sum() else np.nan,
        'T_CD44_pos': float((ad.obs.loc[tt, 'CD44_expr'] > 0).mean()) if tt.sum() else np.nan,
        'T_CD74_pos': float((ad.obs.loc[tt, 'CD74_expr'] > 0).mean()) if tt.sum() else np.nan,
    }
    print('   ', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()}, flush=True)

    # embed on T + TAM
    sub = ad[t | tam].copy()
    for cls in ['ab_DNT', 'gd_T']:
        gl = [g for g in sig[cls].nlargest(50).index if g in genes]
        if gl:
            sc.tl.score_genes(sub, gl, score_name=f'{cls}_sig', use_raw=False)
    # signature transferability: does the GSE163108-derived signature separate
    # the gated DNT/gd-T cells from conventional T cells in this cohort?
    from sklearn.metrics import roc_auc_score
    m_ab = (sub.obs.gate == 'ab_DNT').values
    m_cv = (sub.obs.gate == 'conv_T').values
    m_gd = (sub.obs.gate == 'gd_T').values
    if m_ab.sum() >= 10 and m_cv.sum() >= 10 and 'ab_DNT_sig' in sub.obs:
        y = np.r_[np.ones(m_ab.sum()), np.zeros(m_cv.sum())]
        x = np.r_[sub.obs.loc[m_ab, 'ab_DNT_sig'].values, sub.obs.loc[m_cv, 'ab_DNT_sig'].values]
        st['ab_DNT_sig_AUROC'] = float(roc_auc_score(y, x))
    if m_gd.sum() >= 10 and m_cv.sum() >= 10 and 'gd_T_sig' in sub.obs:
        y = np.r_[np.ones(m_gd.sum()), np.zeros(m_cv.sum())]
        x = np.r_[sub.obs.loc[m_gd, 'gd_T_sig'].values, sub.obs.loc[m_cv, 'gd_T_sig'].values]
        st['gd_T_sig_AUROC'] = float(roc_auc_score(y, x))
    if sub.n_obs > 5000:
        sc.pp.highly_variable_genes(sub, n_top_genes=3000)
        sc.pp.pca(sub, n_comps=20, use_highly_variable=True)
        sc.pp.neighbors(sub, n_neighbors=15)
        sc.tl.umap(sub)
    return ad, sub, st


res = {}
for path, tag in [(os.path.join(DATA, 'SL040_gbm.h5ad'), 'Couturier-SL040'),
                  (os.path.join(DATA, 'SL057_gbm.h5ad'), 'Couturier-SL057')]:
    if not os.path.exists(path):
        print(f'[skip] {path} not found', flush=True)
        continue
    res[tag] = analyse(path, tag)
    gc.collect()

if not res:
    raise SystemExit('no datasets processed')

# ---------- fig19 ----------
fig, axes = plt.subplots(1, len(res), figsize=(6.5 * len(res), 5.5))
if len(res) == 1:
    axes = [axes]
cmap = {'conv_T': '#4C72B0', 'ab_DNT': '#C44E52', 'gd_T': '#55A868',
        'TAM': '#DD8452', 'other': '#DDDDDD'}
for ax, (tag, (ad, sub, st)) in zip(axes, res.items()):
    if 'X_umap' not in sub.obsm:
        ax.text(.5, .5, 'too few cells', ha='center')
        continue
    for g in ['conv_T', 'ab_DNT', 'gd_T', 'TAM', 'other']:
        m = (sub.obs['gate'] == g).values
        if m.sum() == 0:
            continue
        ax.scatter(sub.obsm['X_umap'][m, 0], sub.obsm['X_umap'][m, 1], s=2,
                   c=cmap[g], label=f'{g} ({m.sum()})')
    ax.set_title(tag); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=3, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig19_cellxgene_umaps.png'), dpi=150); plt.close()

# ---------- cross-cohort table ----------
prev = [
    {'cohort': 'GSE163108', 'T_cells': 24756 - 419, 'ab_DNT': 801, 'gd_T': 34, 'conv_T': 23921},
    {'cohort': 'GSE182109', 'T_cells': 14536 + 11308 + 1769, 'ab_DNT': 11308, 'gd_T': 1769, 'conv_T': 14536},
]
summary = pd.DataFrame(prev + [{k: s[k] for k in ['cohort', 'T_cells', 'ab_DNT', 'gd_T', 'conv_T']}
                               for _, (ad, sub, s) in res.items()])
for c in ['ab_DNT', 'gd_T']:
    summary[c + '_pct'] = (100 * summary[c] / summary['T_cells'].replace(0, np.nan)).round(3)
summary.to_csv(os.path.join(BASE, 'cross_cohort_composition.csv'), index=False)
print(summary.to_string(index=False), flush=True)

lr = pd.DataFrame([s for _, (ad, sub, s) in res.items()])
lr.to_csv(os.path.join(BASE, 'lr_axis_summary.csv'), index=False)

# ---------- fig20 ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
x = np.arange(len(summary)); w = 0.38
ax.bar(x - w / 2, summary['ab_DNT_pct'], w, color='#C44E52', label='ab_DNT')
ax.bar(x + w / 2, summary['gd_T_pct'], w, color='#55A868', label='gd_T')
ax.set_xticks(x); ax.set_xticklabels(summary['cohort'], rotation=20, ha='right', fontsize=8)
ax.set_ylabel('% of CD3+ T cells'); ax.legend(); ax.set_title('Cross-cohort composition')

for ax, lig, rec, ttl in [(axes[1], 'SPP1', 'CD44', 'SPP1-CD44 axis'),
                          (axes[2], 'MIF', 'CD74', 'MIF-CD74 axis')]:
    vals, labels = [], []
    for tag, (ad, sub, s) in res.items():
        tm = ad.obs.gate == 'TAM'
        tt = ad.obs.gate.isin(['conv_T', 'ab_DNT', 'gd_T'])
        vals += [ad.obs.loc[tm, f'{lig}_expr'].values, ad.obs.loc[tt, f'{rec}_expr'].values]
        labels += [f'{tag.split("-")[-1]}\nTAM {lig}', f'{tag.split("-")[-1]}\nT {rec}']
    ax.boxplot(vals, showfliers=False)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(ttl)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig20_cellxgene_LR.png'), dpi=150); plt.close()

for tag, (ad, sub, s) in res.items():
    cols = ['gate', 'ab_DNT_sig', 'gd_T_sig', 'SPP1_expr', 'MIF_expr', 'CD44_expr', 'CD74_expr']
    ad.obs[[c for c in cols if c in ad.obs]].to_csv(os.path.join(BASE, f'cell_level_{tag}.csv'))

print('[done] module 10b', flush=True)
