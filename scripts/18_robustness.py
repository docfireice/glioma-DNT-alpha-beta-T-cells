# -*- coding: utf-8 -*-
"""Module 11 (robustness / pre-submission hardening)
1) Multivariable Cox: TCGA (age, sex, G-CIMP) and CGGA (age, grade, IDH, chemo, RT)
2) Continuous per-SD Cox + proportional-hazards assumption test
3) Dropout simulation: is the DN gate explained by transcript dropout?
4) TCR privacy sensitivity: drop singleton clonotypes
5) Atlas per-donor grade association: KW + Jonckheere-Terpstra trend
Outputs: results/GSE163108/robustness_*.csv, results/CELLxGENE/atlas_perdonor_grade_tests.csv,
         results/GSE163108/figs/fig23_forest_multivariable.png, robustness_summary.txt
"""
import os, warnings
import numpy as np, pandas as pd
import scanpy as sc
from scipy import stats, sparse as sp
from scipy.optimize import nnls
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', 'results', 'GSE163108')
CXG = os.path.join(HERE, '..', 'results', 'CELLxGENE')
DATA = os.path.join(HERE, '..', 'data')
FIGS = os.path.join(BASE, 'figs')
os.makedirs(FIGS, exist_ok=True)
LOG = []


def log(s):
    print(s, flush=True)
    LOG.append(s)


def cox(df, cols, label, penalizer=0.01):
    d = df[['time', 'event'] + cols].dropna().copy()
    if len(d) < 20 or d['event'].sum() < 10:
        log(f'  [{label}] SKIPPED: n={len(d)}, events={int(d["event"].sum())}')
        return pd.DataFrame(), None, np.nan
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(d, duration_col='time', event_col='event')
    out = []
    for c in cols:
        s = cph.summary.loc[c]
        out.append(dict(model=label, term=c, n=len(d), events=int(d.event.sum()),
                        HR=s['exp(coef)'], lo=s['exp(coef) lower 95%'], hi=s['exp(coef) upper 95%'],
                        p=s['p']))
    ph = proportional_hazard_test(cph, d, time_transform='rank')
    php = ph.summary['p'].min()
    log(f'  [{label}] n={len(d)}, events={int(d.event.sum())}, min PH-test p={php:.3f}')
    return pd.DataFrame(out), cph, php


# ===================== 1) TCGA multivariable =====================
log('=' * 70)
log('[1] TCGA-GBM multivariable Cox')
frac = pd.read_csv(os.path.join(BASE, 'tcga_deconv_fractions.csv'), index_col=0)
clin = pd.read_csv(os.path.join(DATA, 'TCGA', 'GBM_clinical.tsv'), sep='\t', index_col=0)
clin = clin[~clin.index.duplicated()]
tc = pd.DataFrame({
    'patient': [str(i)[:12] for i in clin.index],
    'age': pd.to_numeric(clin['CDE_DxAge'], errors='coerce')
            .fillna(pd.to_numeric(clin['age_at_initial_pathologic_diagnosis'], errors='coerce')).values,
    'male': (clin['gender'].astype(str).str.upper() == 'MALE').astype(float).values,
    'gcimp': ((clin['G_CIMP_STATUS'].astype(str).str.contains('G-CIMP'))
              & clin['G_CIMP_STATUS'].notna()).astype(float).values,
    'OS.time': pd.to_numeric(clin['CDE_survival_time'], errors='coerce').values,
    'event': (clin['CDE_vital_status'].astype(str).str.upper() == 'DECEASED').astype(float).values,
}, index=clin.index)
tc = tc.drop_duplicates('patient').set_index('patient')
frac['patient'] = [str(i)[:12] for i in frac.index]
fr = frac.dropna().drop_duplicates('patient').set_index('patient')
df = fr.join(tc, how='inner')
df['time'] = df['OS.time']
for g in ['gd_T', 'ab_DNT']:
    df[g + '_z'] = (df[g] - df[g].mean()) / df[g].std()
log(f'  n={len(df)}, events={int(df.event.sum())}, median OS={df.time.median():.0f} d')

r_t = []
m, _, php = cox(df, ['gd_T_z'], 'TCGA gd_T univariable'); r_t.append(m)
m, _, _ = cox(df, ['gd_T_z', 'age', 'male', 'gcimp'], 'TCGA gd_T adjusted'); r_t.append(m)
m, _, _ = cox(df, ['ab_DNT_z'], 'TCGA ab_DNT univariable'); r_t.append(m)
m, _, _ = cox(df, ['ab_DNT_z', 'age', 'male', 'gcimp'], 'TCGA ab_DNT adjusted'); r_t.append(m)
tcga_res = pd.concat(r_t, ignore_index=True)
tcga_res.to_csv(os.path.join(BASE, 'robustness_tcga_cox.csv'), index=False)

# ===================== 2) CGGA multivariable =====================
log('[2] CGGA-693 multivariable Cox')
SIG = pd.read_csv(os.path.join(BASE, 'signature_matrix.csv'), index_col=0)
genes = list(SIG.index)
expr = pd.read_csv(os.path.join(DATA, 'CGGA', 'CGGA.mRNAseq_693.RSEM-genes.20200506.txt'),
                   sep='\t', index_col=0)
common = [g for g in genes if g in expr.index]
Xl = np.log2(expr.loc[common].T.astype(float) + 1)
bulk_z = (Xl - Xl.mean(0)) / (Xl.std(0) + 1e-9)
sig_z = (SIG.loc[common] - SIG.loc[common].mean(0)) / (SIG.loc[common].std(0) + 1e-9)
fr_c = np.apply_along_axis(lambda y: nnls(sig_z, y)[0], 1, bulk_z.values)
cg = pd.DataFrame(fr_c, index=[str(i).replace('.', '-') for i in expr.columns],
                  columns=['ab_DNT', 'gd_T', 'conv_T'])
cg = cg.div(cg.sum(1), axis=0).fillna(0)

cl = pd.read_csv(os.path.join(DATA, 'CGGA', 'CGGA.mRNAseq_693_clinical.20200506.txt'), sep='\t')
cl.columns = [c.strip() for c in cl.columns]
cl = cl.set_index(cl.columns[0])
cl.index = [str(i).replace('.', '-') for i in cl.index]
cc = pd.DataFrame({
    'time': pd.to_numeric(cl['OS'], errors='coerce'),
    'event': pd.to_numeric(cl[[c for c in cl.columns if c.lower().startswith('censor')][0]],
                           errors='coerce'),
    'age': pd.to_numeric(cl['Age'], errors='coerce'),
    'male': (cl['Gender'].astype(str).str.lower().str.startswith('m')).astype(float),
    'grade': cl['Grade'].astype(str).map({'WHO II': 2, 'WHO III': 3, 'WHO IV': 4}).astype(float),
    'idh_mut': (cl['IDH_mutation_status'].astype(str).str.lower() == 'mutant').astype(float),
    'chemo': pd.to_numeric(cl[[c for c in cl.columns if c.startswith('Chemo')][0]],
                           errors='coerce'),
    'rt': pd.to_numeric(cl[[c for c in cl.columns if c.startswith('Radio')][0]],
                        errors='coerce'),
}, index=cl.index)
cc.loc[cl['IDH_mutation_status'].astype(str).str.contains('NA', na=True), 'idh_mut'] = np.nan
d = cg.join(cc, how='inner').dropna(subset=['time', 'event'])
d = d[d.time > 0]
for g in ['gd_T', 'ab_DNT']:
    d[g + '_z'] = (d[g] - d[g].mean()) / d[g].std()
log(f'  n={len(d)}, events={int(d.event.sum())}')
r_c = []
r_c.append(cox(d, ['gd_T_z'], 'CGGA all gd_T univariable')[0])
r_c.append(cox(d, ['gd_T_z', 'age', 'male', 'grade', 'idh_mut', 'chemo', 'rt'],
               'CGGA all gd_T adjusted')[0])
r_c.append(cox(d, ['ab_DNT_z'], 'CGGA all ab_DNT univariable')[0])
r_c.append(cox(d, ['ab_DNT_z', 'age', 'male', 'grade', 'idh_mut', 'chemo', 'rt'],
               'CGGA all ab_DNT adjusted')[0])
g4 = d[d.grade == 4].copy()
r_c.append(cox(g4, ['gd_T_z'], 'CGGA GBM gd_T univariable')[0])
r_c.append(cox(g4, ['gd_T_z', 'age', 'male', 'idh_mut', 'chemo'], 'CGGA GBM gd_T adjusted')[0])
r_c.append(cox(g4, ['ab_DNT_z'], 'CGGA GBM ab_DNT univariable')[0])
r_c.append(cox(g4, ['ab_DNT_z', 'age', 'male', 'idh_mut', 'chemo'],
               'CGGA GBM ab_DNT adjusted')[0])
cgga_res = pd.concat(r_c, ignore_index=True)
cgga_res.to_csv(os.path.join(BASE, 'robustness_cgga_cox.csv'), index=False)

# ===================== 3) Dropout simulation =====================
log('[3] Dropout simulation (GSE163108 raw counts)')
ad = sc.read_h5ad(os.path.join(BASE, 'gse163108_processed.h5ad'))
ad.obs_names_make_unique()
C = ad.layers['counts']
C = sp.csr_matrix(C) if not sp.issparse(C) else C.tocsr()
genes_all = list(ad.var_names)
idx = {g: i for i, g in enumerate(genes_all)}
need = ['CD4', 'CD8A', 'CD8B', 'TRAC', 'TRDC', 'TRDV1', 'TRDV2', 'TRGC1']
have = {g: idx[g] for g in need if g in idx}
log(f'  markers found: {sorted(have)}')
gate = ad.obs['gate'].astype(str).values
conv = np.where(gate == 'conv_T')[0]
sub = C[conv]
cd4 = np.asarray(sub[:, have['CD4']].todense()).ravel()
cd8a = np.asarray(sub[:, have['CD8A']].todense()).ravel()
cd8b = np.asarray(sub[:, have['CD8B']].todense()).ravel() if 'CD8B' in have else np.zeros(len(conv))
pos_cells = np.where((cd4 > 0) | (cd8a > 0) | (cd8b > 0))[0]
X0 = sub[pos_cells].astype(np.int64)
depth0 = np.asarray(X0.sum(1)).ravel()
log(f'  simulated cells: {len(pos_cells)}, median depth {np.median(depth0):.0f} UMI')

rows = []
rng = np.random.default_rng(0)
for p in [1.0, 0.5, 0.25, 0.10, 0.05]:
    Xp = X0.copy()
    if p < 1.0:
        Xp.data = rng.binomial(Xp.data.astype(np.int64), p)
    Xp.eliminate_zeros()
    c4 = np.asarray(Xp[:, have['CD4']].todense()).ravel()
    c8 = np.asarray(Xp[:, have['CD8A']].todense()).ravel()
    c8b = np.asarray(Xp[:, have['CD8B']].todense()).ravel() if 'CD8B' in have else np.zeros(len(pos_cells))
    trac = np.asarray(Xp[:, have['TRAC']].todense()).ravel() if 'TRAC' in have else np.ones(len(pos_cells))
    gd = np.zeros(len(pos_cells), bool)
    for g in ['TRDC', 'TRDV1', 'TRDV2', 'TRGC1']:
        if g in have:
            gd |= np.asarray(Xp[:, have[g]].todense()).ravel() > 0
    fake_dn = (c4 == 0) & (c8 == 0) & (c8b == 0) & (trac > 0) & (~gd)
    rows.append(dict(depth_scale=p, median_depth=float(np.median(np.asarray(Xp.sum(1)).ravel())),
                     false_DN_n=int(fake_dn.sum()), false_DN_rate=float(fake_dn.mean())))
    log(f'  scale {p:>4}: median depth {rows[-1]["median_depth"]:7.0f} | false-DN {fake_dn.mean()*100:5.2f}%')
sim = pd.DataFrame(rows)
sim.to_csv(os.path.join(BASE, 'robustness_dropout_simulation.csv'), index=False)
obs_dn_frac = float((gate == 'ab_DNT').sum() / len(gate))
log(f'  observed DN fraction (GSE163108, of all T cells) = {obs_dn_frac*100:.1f}%')

# ===================== 4) TCR privacy sensitivity =====================
log('[4] TCR privacy sensitivity (drop singletons)')
obs = ad.obs.copy()
obs['gate'] = gate
obs = obs[obs['clonotype_id'].notna() & (obs['clonotype_id'].astype(str) != 'nan')]
clono = obs.groupby('clonotype_id')
share = clono['gate'].agg(lambda s: set(s))
size = clono.size()
dn_clono = set(obs.loc[obs.gate == 'ab_DNT', 'clonotype_id'])


def privacy(mask_clono):
    s = share[[c for c in dn_clono if c in share.index and mask_clono.get(c, False)]]
    if len(s) == 0:
        return dict(n_clonotypes=0, private_clono_pct=np.nan, cells_with_private_pct=np.nan)
    priv = s.apply(lambda x: x == {'ab_DNT'})
    dn = obs[(obs.gate == 'ab_DNT') & obs.clonotype_id.isin(s.index)]
    return dict(n_clonotypes=int(len(s)), private_clono_pct=float(priv.mean() * 100),
                cells_with_private_pct=float(dn.clonotype_id.isin(priv[priv].index).mean() * 100))


all_ = privacy(pd.Series(True, index=share.index))
multi = privacy(size >= 2)
res_tcr = pd.DataFrame([dict(definition='all clonotypes', **all_),
                        dict(definition='clonotypes size>=2 only', **multi)])
res_tcr.to_csv(os.path.join(BASE, 'robustness_tcr_privacy.csv'), index=False)
for _, r in res_tcr.iterrows():
    log(f'  {r.definition}: n={r.n_clonotypes}, private {r.private_clono_pct:.1f}%, '
        f'cells {r.cells_with_private_pct:.1f}%')

# ===================== 5) Atlas per-donor grade tests =====================
log('[5] Atlas per-donor grade association')
at = pd.read_csv(os.path.join(CXG, 'atlas_per_donor_T_fractions.csv'))
res_g = []
for tag, dsub in [('all donors', at), ('donors with T>=30', at[at.T_cells >= 30])]:
    for col in ['gd_T_pct', 'ab_DNT_pct']:
        d2 = dsub.dropna(subset=['grade', col])
        groups = [g[col].values for _, g in d2.groupby('grade')]
        kw = stats.kruskal(*groups)
        tau = stats.kendalltau(d2['grade'], d2[col])
        res_g.append(dict(subset=tag, metric=col, n_donors=len(d2),
                          kw_p=kw.pvalue, jt_tau=tau.statistic, jt_p=tau.pvalue,
                          median_by_grade={int(k): round(float(v[col].median()), 2)
                                           for k, v in d2.groupby('grade')}))
        log(f'  {tag:18s} {col:10s}: n={len(d2)}, KW p={kw.pvalue:.4f}, '
            f'JT tau={tau.statistic:.3f} p={tau.pvalue:.4f}')
pd.DataFrame(res_g).to_csv(os.path.join(CXG, 'atlas_perdonor_grade_tests.csv'), index=False)

# ===================== 6) Forest plot =====================
log('[6] Forest plot')
plot_df = pd.concat([tcga_res, cgga_res], ignore_index=True)
plot_df = plot_df[plot_df.term.str.endswith('_z')].copy()
plot_df['lab'] = plot_df.model.str.replace(' gd_T', '\n').str.replace(' ab_DNT', ' ')
fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(plot_df) + 1.6))
y = np.arange(len(plot_df))[::-1]
col = ['#C44E52' if 'ab_DNT' in t else '#4C72B0' for t in plot_df.term]
for yi, hr, lo, hi, c in zip(y, plot_df.HR, plot_df.lo, plot_df.hi, col):
    ax.plot([lo, hi], [yi, yi], color=c, lw=1.8, solid_capstyle='butt', zorder=2)
    ax.scatter([hr], [yi], c=c, s=42, zorder=3)
ax.axvline(1, ls='--', c='grey', lw=1)
ax.set_yticks(y)
ax.set_yticklabels([f'{m}  (n={n})' for m, n in zip(plot_df.model, plot_df.n)], fontsize=8)
ax.set_xscale('log')
ax.set_xlabel('Hazard ratio per SD (95% CI, log scale)')
for yi, (hr, p) in zip(y, zip(plot_df.HR, plot_df.p)):
    ax.text(0.985, yi, f'{hr:.2f}  p={p:.2g}', transform=ax.get_yaxis_transform(),
            ha='right', va='center', fontsize=7.5,
            color='#B22222' if p < 0.05 else '#555555')
ax.set_title('Univariable vs adjusted Cox models', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig23_forest_multivariable.png'), dpi=300)
log('  saved fig23_forest_multivariable.png')

with open(os.path.join(BASE, 'robustness_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG))
log('DONE')
