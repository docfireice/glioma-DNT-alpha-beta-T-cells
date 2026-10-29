# -*- coding: utf-8 -*-
"""Plan B: replace the survival endpoint with cross-sectional clinical anchors.

Clinical endpoints re-examined:
  (1) WHO grade association  (CGGA 693 grade II-IV; GSE16011 grade I-IV + control)
  (2) IDH / G-CIMP association (CGGA, TCGA)
  (3) SPP1-CD44 and MIF-CD74 interface scores: association with grade and with gd_T content
  (4) Lineage-anchored gd_T quantification using TCR gamma/delta constant+variable
      transcripts (TRDC/TRGC1/TRGC2/TRDV2/TRDV3/TRGV9) -- definition-based, requires
      no learned signature, and is the same criterion used for the scRNA-seq gate.

Rationale: the survival claim failed because the infiltration estimate is
method-dependent. Here we ask whether the clinical gradient survives when the
endpoint is cross-sectional (grade/IDH) rather than time-to-event, i.e. whether
it is robust to the choice of quantification.
"""
import os, warnings
import numpy as np
import pandas as pd
import gzip
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(ROOT, 'results', 'GSE163108')
FIG = os.path.join(RES, 'figs')
DATA = os.path.join(ROOT, 'data')
os.makedirs(FIG, exist_ok=True)
OUT = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    OUT.append(s)


# ---------------------------------------------------------------- helpers
def spearman(x, y):
    d = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(d) < 10:
        return np.nan, np.nan, 0
    r, p = stats.spearmanr(d.x, d.y)
    return r, p, len(d)


def partial_spearman(x, y, covars):
    """Spearman on rank-residualised variables (grade-adjusted association)."""
    d = pd.DataFrame({'x': x, 'y': y})
    for k, v in covars.items():
        d[k] = v
    d = d.dropna()
    if len(d) < 20:
        return np.nan, np.nan, 0
    R = d.rank()
    def resid(col):
        X = sm.add_constant(R[[c for c in covars]])
        return sm.OLS(R[col], X).fit().resid
    return (*stats.spearmanr(resid('x'), resid('y')), len(d))


def ols_beta(df, ycol, pred, covars):
    d = df[[ycol, pred] + covars].dropna()
    if len(d) < 20 or d[pred].nunique() < 2:
        return None
    X = sm.add_constant(d[[pred] + covars].astype(float))
    m = sm.OLS(d[ycol].astype(float), X).fit()
    b, se, p = m.params[pred], m.bse[pred], m.pvalues[pred]
    return dict(n=len(d), beta=b, lo=b - 1.96 * se, hi=b + 1.96 * se, p=p)


def zscore(s):
    return (s - s.mean()) / s.std()


def bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = p[order[i]] * m / (i + 1)
        prev = min(prev, val)
        q[order[i]] = prev
    return q


# ---------------------------------------------------------------- load scores
gd_score = {
    'CGGA': pd.read_csv(os.path.join(RES, 'cgga_marker_scores.csv'), index_col=0),
    'TCGA': pd.read_csv(os.path.join(RES, 'tcga_marker_scores.csv'), index_col=0),
    'GSE16011': pd.read_csv(os.path.join(DATA, 'GSE16011', 'gse16011_marker_scores.csv'), index_col=0),
}
for k, v in gd_score.items():
    log(f'[score] {k}: {v.shape}')

# ---------------------------------------------------------------- CGGA clinical
cl = pd.read_csv(os.path.join(DATA, 'CGGA', 'CGGA.mRNAseq_693_clinical.20200506.txt'), sep='\t')
cl.columns = [c.strip() for c in cl.columns]
cl = cl.set_index('CGGA_ID')
cgga = gd_score['CGGA'].copy()
cgga.index = [str(i).replace('.', '-') for i in cgga.index]
cgga = cgga.join(pd.DataFrame({
    'grade': cl['Grade'].astype(str).map({'WHO II': 2, 'WHO III': 3, 'WHO IV': 4}).astype(float),
    'idh_mut': (cl['IDH_mutation_status'].astype(str).str.lower() == 'mutant').astype(float),
    'age': pd.to_numeric(cl['Age'], errors='coerce'),
    'male': (cl['Gender'].astype(str).str.lower().str.startswith('m')).astype(float),
    'histology': cl['Histology'].astype(str),
    'prs': cl['PRS_type'].astype(str),
}, index=cl.index), how='inner')
cgga = cgga.dropna(subset=['grade'])
log(f'[CGGA] n={len(cgga)} grades={cgga.grade.value_counts().sort_index().to_dict()} '
    f'IDH_mut={int(cgga.idh_mut.sum())}/{cgga.idh_mut.notna().sum()}')

# ---------------------------------------------------------------- GSE16011 annotation
sdrf = pd.read_csv(os.path.join(DATA, 'GSE16011', 'E-GEOD-16011.sdrf.txt'), sep='\t')
hcol = [c for c in sdrf.columns if 'histology' in c][0]
acol = [c for c in sdrf.columns if 'age at diagnosis' in c][0]
gcol = [c for c in sdrf.columns if 'gender' in c][0]
fcol = [c for c in sdrf.columns if 'Array Data File' in c][0]
sdrf['gsm'] = sdrf[fcol].astype(str).str.replace('.CEL', '', regex=False).str.strip()

GRADE_MAP = {'PA (grade I)': 1, 'A (grade II)': 2, 'OA (grade II)': 2, 'OD (grade II)': 2,
             'A (grade III)': 3, 'OA (grade III)': 3, 'OD (grade III)': 3,
             'GBM (grade IV)': 4, 'control': 0}
ann = pd.DataFrame({
    'gsm': sdrf['gsm'].values,
    'grade': sdrf[hcol].astype(str).str.strip().map(GRADE_MAP).values,
    'age': pd.to_numeric(sdrf[acol], errors='coerce').values,
    'male': (sdrf[gcol].astype(str).str.lower().str[0] == 'm').astype(float).values,
    'histology_raw': sdrf[hcol].astype(str).str.strip().values,
}).drop_duplicates('gsm').set_index('gsm')
g11 = gd_score['GSE16011'].copy()
g11.index = [str(i).replace('.CEL.gz', '').replace('(', '').replace(')', '').strip() for i in g11.index]
g11 = g11.join(ann, how='inner')
log(f'[GSE16011] matched={len(g11)} grades={g11.grade.value_counts().sort_values().to_dict()}')

# ---------------------------------------------------------------- TCGA clinical
clin_t = pd.read_csv(os.path.join(DATA, 'TCGA', 'GBM_clinical.tsv'), sep='\t', index_col=0)
tcga = gd_score['TCGA'].copy()
tc = pd.DataFrame({
    'patient': [str(i)[:12] for i in clin_t.index],
    'gcimp': ((clin_t['G_CIMP_STATUS'].astype(str).str.upper() == 'G-CIMP')).astype(float).values,
    'age': pd.to_numeric(clin_t['CDE_DxAge'], errors='coerce').values,
    'male': (clin_t['gender'].astype(str).str.upper() == 'MALE').astype(float).values,
    'subtype': clin_t['GeneExp_Subtype'].astype(str).values,
}, index=clin_t.index).drop_duplicates('patient').set_index('patient')
tcga['pat'] = [str(i)[:12] for i in tcga.index]
tcga = tcga.drop_duplicates('pat').set_index('pat').join(tc, how='inner')
log(f'[TCGA] n={len(tcga)} G-CIMP={int(tcga.gcimp.sum())}')

# ---------------------------------------------------------------- expression-derived scores
def load_expr(name):
    if name == 'CGGA':
        e = pd.read_csv(os.path.join(DATA, 'CGGA', 'CGGA.mRNAseq_693.RSEM-genes.20200506.txt'),
                        sep='\t', index_col=0)
        e = np.log2(e + 1)
        e.columns = [str(c).replace('.', '-') for c in e.columns]
    elif name == 'TCGA':
        with gzip.open(os.path.join(DATA, 'TCGA', 'GBM_HiSeqV2.gz'), 'rt') as f:
            e = pd.read_csv(f, sep='\t', index_col=0)
    else:
        e = pd.read_csv(os.path.join(DATA, 'GSE16011', 'GSE16011_rma_gene.tsv'), sep='\t', index_col=0)
        if 'PROBEID' in e.columns:
            e = e.set_index('PROBEID')
        e = e.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any')
        e.columns = [str(c).replace('.CEL.gz', '').replace('(', '').replace(')', '').strip() for c in e.columns]
    return e.groupby(level=0).mean()


LINEAGE = {'CGGA': ['TRDC', 'TRGC1', 'TRGC2', 'TRGV9'],
           'GSE16011': ['TRDC', 'TRGC2', 'TRDV2', 'TRDV3', 'TRGV9'],
           'TCGA': []}
LINEAGE_PRIMARY = {'CGGA': 'TRDC', 'GSE16011': 'TRDC', 'TCGA': None}
AXIS = {'SPP1_CD44': ['SPP1', 'CD44'], 'MIF_CD74': ['MIF', 'CD74']}
TCELL = ['CD3D', 'CD3E', 'CD2']

expr = {k: load_expr(k) for k in ['CGGA', 'TCGA', 'GSE16011']}
for k, e in expr.items():
    log(f'[expr] {k}: {e.shape}')

derived = {}
for k, e in expr.items():
    d = pd.DataFrame(index=e.columns)
    lg = [g for g in LINEAGE[k] if g in e.index]
    if lg:
        Z = e.loc[lg].apply(zscore, axis=1)
        d['gd_lineage'] = Z.mean(axis=0)
        d['n_lineage_genes'] = len(lg)
        for g in lg:
            d['tr_' + g] = e.loc[g]
    if LINEAGE_PRIMARY[k] and LINEAGE_PRIMARY[k] in e.index:
        d['gd_anchor'] = zscore(e.loc[LINEAGE_PRIMARY[k]])
    for ax, gl in AXIS.items():
        hit = [g for g in gl if g in e.index]
        if len(hit) == len(gl):
            d[ax] = e.loc[hit].apply(zscore, axis=1).mean(axis=0)
    hit_t = [g for g in TCELL if g in e.index]
    if hit_t:
        d['tcell'] = e.loc[hit_t].apply(zscore, axis=1).mean(axis=0)
    derived[k] = d
    log(f'[derived] {k}: cols={list(d.columns)} lineage_genes={lg if lg else "-"}')

# normalise index keys for join
derived['CGGA'].index = [str(i).replace('.', '-') for i in derived['CGGA'].index]
derived['GSE16011'].index = [str(i).replace('.CEL.gz', '').replace('(', '').replace(')', '').strip()
                             for i in derived['GSE16011'].index]
derived['TCGA'].index = [str(i)[:12] for i in derived['TCGA'].index]
derived['TCGA'] = derived['TCGA'][~derived['TCGA'].index.duplicated()]

# ================================================================ 1. GRADE
log('\n' + '=' * 70)
log('1) WHO grade association (cross-sectional)')
log('=' * 70)
grade_rows = []


def grade_block(tag, df, pops, grade_col='grade', min_grade=1):
    sub = df[df[grade_col] >= min_grade]
    for pop in pops:
        if pop not in sub.columns:
            continue
        gr = [g for g in sorted(sub[grade_col].dropna().unique())]
        groups = [sub.loc[sub[grade_col] == g, pop].dropna() for g in gr]
        groups = [x for x in groups if len(x) >= 3]
        if len(groups) < 2:
            continue
        kw = stats.kruskal(*groups)
        rho, p_rho, n = spearman(sub[grade_col], sub[pop])
        row = dict(cohort=tag, population=pop, n=n, n_grades=len(groups),
                   kw_p=kw.pvalue, spearman_rho=rho, trend_p=p_rho,
                   mean_by_grade={int(g): round(float(sub.loc[sub[grade_col] == g, pop].mean()), 3)
                                  for g in gr})
        grade_rows.append(row)
        log(f"  {tag:9s} {pop:10s} n={n:4d} KW p={kw.pvalue:.3g}  trend rho={rho:+.3f} p={p_rho:.3g}"
            f"  means={row['mean_by_grade']}")


POP = ['gd_T', 'ab_DNT', 'conv_T']
grade_block('CGGA', cgga, POP, min_grade=2)
grade_block('GSE16011', g11, POP, min_grade=1)

# adjusted per-grade-step effect
log('\n  -- adjusted per-grade-step effect (OLS, score ~ grade + age + sex [+ IDH]) --')
adj_rows = []
for tag, df, extra in [('CGGA', cgga, ['idh_mut']), ('GSE16011', g11, [])]:
    d = df[df.grade >= (2 if tag == 'CGGA' else 1)].copy()
    for pop in POP:
        for cov in [[], ['age', 'male'] + extra]:
            r = ols_beta(d, pop, 'grade', cov)
            if r is None:
                continue
            adj_rows.append(dict(cohort=tag, population=pop, model='adjusted' if cov else 'unadjusted',
                                 n=r['n'], beta_per_grade=r['beta'], lo=r['lo'], hi=r['hi'], p=r['p']))
            log(f"  {tag:9s} {pop:9s} {'adj' if cov else 'raw':11s} n={r['n']:4d} "
                f"beta={r['beta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] p={r['p']:.3g}")

# ================================================================ 2. IDH
log('\n' + '=' * 70)
log('2) IDH / G-CIMP association')
log('=' * 70)
idh_rows = []
for pop in POP:
    a = cgga.loc[cgga.idh_mut == 1, pop].dropna()
    b = cgga.loc[cgga.idh_mut == 0, pop].dropna()
    u = stats.mannwhitneyu(a, b)
    # logistic: IDH ~ score + grade
    d = cgga.dropna(subset=[pop, 'idh_mut', 'grade'])
    X = sm.add_constant(sm.add_constant(d[[pop, 'grade']].astype(float)))
    lg = sm.Logit(d.idh_mut.astype(float), X).fit(disp=0)
    idh_rows.append(dict(cohort='CGGA', population=pop, mut_mean=round(a.mean(), 3), wt_mean=round(b.mean(), 3),
                         n_mut=len(a), n_wt=len(b), mwu_p=u.pvalue,
                         or_per_sd=np.exp(lg.params[pop]).round(3), logit_p=lg.pvalues[pop]))
    log(f"  CGGA {pop:9s} mutant={a.mean():.3f} (n={len(a)}) vs wildtype={b.mean():.3f} (n={len(b)}) "
        f"MWU p={u.pvalue:.3g} | grade-adjusted OR/SD={np.exp(lg.params[pop]):.2f} p={lg.pvalues[pop]:.3g}")

for pop in POP:
    a = tcga.loc[tcga.gcimp == 1, pop].dropna()
    b = tcga.loc[tcga.gcimp == 0, pop].dropna()
    if len(a) < 5:
        continue
    u = stats.mannwhitneyu(a, b)
    idh_rows.append(dict(cohort='TCGA(G-CIMP)', population=pop, mut_mean=round(a.mean(), 3),
                         wt_mean=round(b.mean(), 3), n_mut=len(a), n_wt=len(b), mwu_p=u.pvalue,
                         or_per_sd=np.nan, logit_p=np.nan))
    log(f"  TCGA G-CIMP+ {pop:9s} {a.mean():.3f} (n={len(a)}) vs G-CIMP- {b.mean():.3f} (n={len(b)}) p={u.pvalue:.3g}")

# ================================================================ 3. axis scores
log('\n' + '=' * 70)
log('3) SPP1-CD44 / MIF-CD74 interface scores')
log('=' * 70)
axis_rows = []
DATASETS = {'CGGA': (cgga, 2), 'GSE16011': (g11, 1), 'TCGA': (tcga, None)}
for tag, (df, mg) in DATASETS.items():
    d = df.join(derived[tag], how='inner')
    if 'SPP1_CD44' not in d.columns:
        log(f'  {tag}: axis score unavailable')
        continue
    has_grade = ('grade' in d.columns) and (d['grade'].notna().sum() > 30)
    for ax in ['SPP1_CD44', 'MIF_CD74']:
        for pop in POP:
            r, p, n = spearman(d[ax], d[pop])
            pr, pp, pn = partial_spearman(d[ax], d[pop], {'grade': d['grade']}) \
                if has_grade else (np.nan, np.nan, 0)
            axis_rows.append(dict(cohort=tag, axis=ax, population=pop, n=n,
                                  rho=r, p=p, rho_grade_adj=pr, p_grade_adj=pp))
            log(f"  {tag:9s} {ax:10s} ~ {pop:9s} rho={r:+.3f} p={p:.3g} | grade-adj rho={pr:+.3f} p={pp:.3g}")
    if has_grade:
        sub = d[d.grade >= mg]
        for ax in ['SPP1_CD44', 'MIF_CD74']:
            rho, p, n = spearman(sub['grade'], sub[ax])
            log(f"  {tag:9s} {ax:10s} ~ grade   rho={rho:+.3f} p={p:.3g} n={n}")

# ================================================================ 4. lineage anchoring
log('\n' + '=' * 70)
log('4) Lineage-transcript anchoring of the gd_T estimate')
log('=' * 70)
anchor_rows = []
for tag in ['CGGA', 'GSE16011']:
    d = DATASETS[tag][0].join(derived[tag], how='inner')
    if 'gd_anchor' not in d.columns:
        continue
    mg = DATASETS[tag][1]
    sub = d[d.grade >= mg]
    for label, col in [('TRDC anchor', 'gd_anchor'), ('multi-gene lineage', 'gd_lineage')]:
        r, p, n = spearman(d['gd_T'], d[col])
        r2, _, _ = spearman(d[col], d['conv_T'])
        r3, _, _ = spearman(d[col], d['tcell'])
        rho_g, pg, ng = spearman(sub['grade'], sub[col])
        anchor_rows.append(dict(cohort=tag, metric=label, n=n, rho_sig_vs_metric=r, p=p,
                                rho_metric_vs_convT=r2, rho_metric_vs_tcell=r3,
                                grade_rho=rho_g, grade_p=pg, n_grade=ng))
        log(f"  {tag:9s} {label:18s} vs gd_T score rho={r:+.3f} p={p:.3g} | vs conv_T rho={r2:+.3f} "
            f"| vs grade rho={rho_g:+.3f} p={pg:.3g}")
    # per-gene direction check
    for g in [c for c in d.columns if c.startswith('tr_')]:
        rho_g, pg, ng = spearman(sub['grade'], sub[g])
        log(f"      {g[3:]:6s} vs grade rho={rho_g:+.3f} p={pg:.3g}")

# ================================================================ 4b specificity
log('\n' + '=' * 70)
log('4b) Is the grade gradient gd-specific, or general T-cell infiltration?')
log('=' * 70)
spec_rows = []
for tag in ['CGGA', 'GSE16011']:
    df, mg = DATASETS[tag]
    d = df.join(derived[tag], how='inner')
    d = d[d.grade >= mg]
    d['gd_rel'] = d['gd_T'] - d['conv_T']
    for pop in ['gd_T', 'ab_DNT']:
        for cov in [[], ['CD3D'], ['tcell'], ['conv_T']]:
            cov = [c for c in cov if c in d.columns]
            r = ols_beta(d, pop, 'grade', cov)
            if r is None:
                continue
            spec_rows.append(dict(cohort=tag, population=pop, covariate=','.join(cov) if cov else 'none',
                                  n=r['n'], beta=r['beta'], lo=r['lo'], hi=r['hi'], p=r['p']))
            log(f"  {tag:9s} {pop:7s} adj=[{','.join(cov) if cov else 'none':8s}] "
                f"beta={r['beta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] p={r['p']:.3g}")
    for col in ['gd_rel', 'gd_anchor']:
        if col not in d.columns:
            continue
        rho, p, n = spearman(d['grade'], d[col])
        log(f"  {tag:9s} {col:10s} ~ grade rho={rho:+.3f} p={p:.3g} n={n}")

# ================================================================ 5. relative enrichment
log('\n' + '=' * 70)
log('5) Relative gd enrichment (gd_T - conv_T, log scale) by grade')
log('=' * 70)
rel_rows = []
for tag in ['CGGA', 'GSE16011']:
    df, mg = DATASETS[tag]
    d = df.copy()
    d['gd_rel'] = d['gd_T'] - d['conv_T']
    sub = d[d.grade >= mg]
    rho, p, n = spearman(sub['grade'], sub['gd_rel'])
    r = ols_beta(sub, 'gd_rel', 'grade', ['age', 'male'])
    rel_rows.append(dict(cohort=tag, n=n, rho=rho, p=p,
                         beta=(r['beta'] if r else np.nan), beta_lo=(r['lo'] if r else np.nan),
                         beta_hi=(r['hi'] if r else np.nan), beta_p=(r['p'] if r else np.nan)))
    log(f"  {tag:9s} gd_rel ~ grade rho={rho:+.3f} p={p:.3g} n={n} | adj beta/grade="
        f"{r['beta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] p={r['p']:.3g}")

# ================================================================ save tables
grade_df = pd.DataFrame(grade_rows)
grade_df['kw_q'] = bh(grade_df.kw_p)
grade_df['trend_q'] = bh(grade_df.trend_p)
adj_df = pd.DataFrame(adj_rows)
idh_df = pd.DataFrame(idh_rows)
idh_df['mwu_q'] = bh(idh_df.mwu_p)
axis_df = pd.DataFrame(axis_rows)
if len(axis_df):
    axis_df['p_q'] = bh(axis_df.p)
anchor_df = pd.DataFrame(anchor_rows)
rel_df = pd.DataFrame(rel_rows)

for name, t in [('clinical_grade_tests', grade_df), ('clinical_grade_adjusted', adj_df),
                ('clinical_idh_tests', idh_df), ('clinical_axis_tests', axis_df),
                ('clinical_lineage_anchor', anchor_df), ('clinical_relative_enrichment', rel_df),
                ('clinical_specificity_models', pd.DataFrame(spec_rows))]:
    p = os.path.join(RES, f'{name}.csv')
    t.to_csv(p, index=False)
    log(f'[written] {p}')

# ================================================================ figure
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)

# A: CGGA grade boxplots
ax = fig.add_subplot(gs[0, 0])
for i, pop in enumerate(['gd_T', 'ab_DNT', 'conv_T']):
    data = [cgga.loc[cgga.grade == g, pop].dropna().values for g in [2, 3, 4]]
    pos = np.arange(3) + i * 0.26 - 0.26
    bp = ax.boxplot(data, positions=pos, widths=0.22, showfliers=False, patch_artist=True)
    for b in bp['boxes']:
        b.set_facecolor(['#d62728', '#1f77b4', '#7f7f7f'][i])
        b.set_alpha(0.75)
    for med in bp['medians']:
        med.set_color('k')
ax.set_xticks(range(3))
ax.set_xticklabels(['II', 'III', 'IV'])
ax.set_xlabel('WHO grade')
ax.set_ylabel('marker score (log2)')
ax.set_title(f'A  CGGA (n={len(cgga)})', loc='left', fontsize=11, fontweight='bold')
ax.legend(['$\\gamma\\delta$ T', 'DN $\\alpha\\beta$ T', 'conv. T'], fontsize=7, frameon=False)

# B: GSE16011 grade boxplots
ax = fig.add_subplot(gs[0, 1])
grades11 = [1, 2, 3, 4]
for i, pop in enumerate(['gd_T', 'ab_DNT', 'conv_T']):
    data = [g11.loc[g11.grade == g, pop].dropna().values for g in grades11]
    pos = np.arange(len(grades11)) + i * 0.26 - 0.26
    bp = ax.boxplot(data, positions=pos, widths=0.22, showfliers=False, patch_artist=True)
    for b in bp['boxes']:
        b.set_facecolor(['#d62728', '#1f77b4', '#7f7f7f'][i])
        b.set_alpha(0.75)
    for med in bp['medians']:
        med.set_color('k')
ax.set_xticks(range(len(grades11)))
ax.set_xticklabels(['I', 'II', 'III', 'IV'])
ax.set_xlabel('WHO grade')
ax.set_title(f'B  GSE16011 (n={len(g11)})', loc='left', fontsize=11, fontweight='bold')

# C: lineage anchor scatter
ax = fig.add_subplot(gs[0, 2])
d = DATASETS['CGGA'][0].join(derived['CGGA'], how='inner').dropna(subset=['gd_T', 'gd_anchor'])
sc = ax.scatter(d['gd_anchor'], d['gd_T'], c=d['grade'], cmap='viridis', s=12, alpha=0.75)
r, p, n = spearman(d['gd_T'], d['gd_anchor'])
ax.set_xlabel('TRDC (z)')
ax.set_ylabel('$\\gamma\\delta$ signature score')
ax.set_title(f'C  Lineage anchoring, CGGA\nSpearman $\\rho$={r:.2f}, P={p:.1e}', loc='left',
             fontsize=11, fontweight='bold')
plt.colorbar(sc, ax=ax, label='grade', shrink=0.85)

# D: axis vs gd_T
ax = fig.add_subplot(gs[1, 0])
d = DATASETS['CGGA'][0].join(derived['CGGA'], how='inner').dropna(subset=['gd_T', 'SPP1_CD44'])
r, p, n = spearman(d['SPP1_CD44'], d['gd_T'])
ax.scatter(d['SPP1_CD44'], d['gd_T'], c=d['grade'], cmap='viridis', s=12, alpha=0.75)
ax.set_xlabel('SPP1-CD44 score (z)')
ax.set_ylabel('$\\gamma\\delta$ T score')
ax.set_title(f'D  Interface axis, CGGA\n$\\rho$={r:.2f}, P={p:.1e}', loc='left', fontsize=11,
             fontweight='bold')

# E: per-grade-step effect forest
ax = fig.add_subplot(gs[1, 1])
sub = adj_df[adj_df.model == 'adjusted']
pops = ['gd_T', 'ab_DNT', 'conv_T']
ylab, yval, ylo, yhi, col = [], [], [], [], []
for i, (_, r) in enumerate(sub.iterrows()):
    ylab.append(f"{r.cohort} {r.population}")
    yval.append(r.beta_per_grade); ylo.append(r.lo); yhi.append(r.hi)
    col.append(['#d62728', '#1f77b4', '#7f7f7f'][pops.index(r.population)])
y = np.arange(len(yval))[::-1]
ax.errorbar(yval, y, xerr=[np.array(yval) - np.array(ylo), np.array(yhi) - np.array(yval)],
            fmt='o', ecolor='k', ms=5, lw=1.2, mfc='w', mec='k')
for xi, yi, ci in zip(yval, y, col):
    ax.plot(xi, yi, 'o', color=ci, ms=6)
ax.axvline(0, ls='--', c='grey', lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(ylab, fontsize=7)
ax.set_xlabel('$\\beta$ per grade step (log2 units)')
ax.set_title('E  Adjusted grade effect', loc='left', fontsize=11, fontweight='bold')

# F: relative enrichment
ax = fig.add_subplot(gs[1, 2])
for tag, c in [('CGGA', '#d62728'), ('GSE16011', '#ff7f0e')]:
    df, mg = DATASETS[tag]
    d = df[df.grade >= mg].copy()
    d['gd_rel'] = d['gd_T'] - d['conv_T']
    gs_ = sorted(d.grade.unique())
    m = [d.loc[d.grade == g, 'gd_rel'].mean() for g in gs_]
    se = [d.loc[d.grade == g, 'gd_rel'].std() / np.sqrt((d.grade == g).sum()) for g in gs_]
    ax.errorbar(gs_, m, yerr=[1.96 * s for s in se], marker='o', color=c, label=tag, capsize=3)
ax.axhline(0, ls='--', c='grey', lw=0.8)
ax.set_xlabel('WHO grade')
ax.set_ylabel('$\\gamma\\delta$ - conv. T (log2)')
ax.set_title('F  Relative enrichment', loc='left', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, frameon=False)

plt.savefig(os.path.join(FIG, 'fig24_clinical_endpoints.png'), dpi=200, bbox_inches='tight')
log(f'[figure] {os.path.join(FIG, "fig24_clinical_endpoints.png")}')

with open(os.path.join(RES, 'clinical_endpoints_log.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT))
print('\nDONE')
