# -*- coding: utf-8 -*-
"""Unified, method-consistent infiltration scoring for all three bulk cohorts.

Rationale: NNLS on z-scored bulk vectors has negative observations, which is invalid
for non-negative least squares and produced all-zero fits in 44% of CGGA samples.
We therefore score every cohort with the same class-specific marker approach used for
GSE16011 (top-50 genes with maximal own-class minus max(other-class) expression),
and re-run Cox models identically across cohorts.
"""
import os, gzip, warnings
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', 'results', 'GSE163108')
DATA = os.path.join(HERE, '..', 'data')
rows = []

SIG = pd.read_csv(os.path.join(BASE, 'signature_matrix.csv'), index_col=0)
weights = {}
for c in SIG.columns:
    others = [o for o in SIG.columns if o != c]
    weights[c] = (SIG[c] - SIG[others].max(axis=1)).clip(lower=0)
markers = {c: w.nlargest(50).index.tolist() for c, w in weights.items()}
print('[markers]', {c: len(v) for c, v in markers.items()})


def score(expr, tag):
    """expr: genes x samples, log-scale. Returns samples x gates marker score."""
    out = pd.DataFrame(index=expr.columns)
    for c, gl in markers.items():
        g = [x for x in gl if x in expr.index]
        sub = expr.loc[g]
        out[c] = sub.mean(axis=0)
        print(f'  [{tag}] {c}: {len(g)}/{len(gl)} markers matched')
    return out


def fit(d, cols, label, penalizer=0.01):
    dd = d[['time', 'event'] + cols].dropna()
    cols = [c for c in cols if dd[c].nunique() > 1]
    if len(dd) < 20 or dd.event.sum() < 10 or not cols:
        print(f'  [{label}] skipped n={len(dd)}'); return
    try:
        cph = CoxPHFitter(penalizer=penalizer).fit(dd, 'time', 'event')
    except Exception as e:
        print(f'  [{label}] fit failed {type(e).__name__}'); return
    s = cph.summary.loc[cols[0]]
    r = dict(model=label, n=len(dd), events=int(dd.event.sum()), HR=s['exp(coef)'],
             lo=s['exp(coef) lower 95%'], hi=s['exp(coef) upper 95%'], p=s['p'])
    try:
        php = proportional_hazard_test(cph, dd, time_transform='rank').summary['p'].min()
        r['PH_p_min'] = php
    except Exception:
        pass
    print(f"  {label:40s} n={r['n']:4d} HR={r['HR']:.2f} ({r['lo']:.2f}-{r['hi']:.2f}) p={r['p']:.3g}")
    rows.append(r)


# ---------------- TCGA ----------------
print('[TCGA]')
with gzip.open(os.path.join(DATA, 'TCGA', 'GBM_HiSeqV2.gz'), 'rt') as f:
    bulk = pd.read_csv(f, sep='\t', index_col=0)
bulk = bulk.groupby(level=0).mean()
sc = score(bulk, 'TCGA')
sc.to_csv(os.path.join(BASE, 'tcga_marker_scores.csv'))
clin = pd.read_csv(os.path.join(DATA, 'TCGA', 'GBM_clinical.tsv'), sep='\t', index_col=0)
tc = pd.DataFrame({
    'patient': [str(i)[:12] for i in clin.index],
    'time': pd.to_numeric(clin['CDE_survival_time'], errors='coerce').values,
    'event': (clin['CDE_vital_status'].astype(str).str.upper() == 'DECEASED').astype(float).values,
    'age': pd.to_numeric(clin['CDE_DxAge'], errors='coerce')
            .fillna(pd.to_numeric(clin['age_at_initial_pathologic_diagnosis'], errors='coerce')).values,
    'male': (clin['gender'].astype(str).str.upper() == 'MALE').astype(float).values,
    'gcimp': ((clin['G_CIMP_STATUS'].astype(str).str.contains('G-CIMP'))
              & clin['G_CIMP_STATUS'].notna()).astype(float).values,
}, index=clin.index).drop_duplicates('patient').set_index('patient')
sc['patient'] = [str(i)[:12] for i in sc.index]
m = sc.drop_duplicates('patient').set_index('patient').join(tc, how='inner')
m = m[(m.time > 0)].dropna(subset=['gd_T', 'ab_DNT', 'time', 'event'])
print(f'  analytic n={len(m)}, events={int(m.event.sum())}')
for g in ['gd_T', 'ab_DNT']:
    m[g + '_z'] = (m[g] - m[g].mean()) / m[g].std()
    m[g + '_hi'] = (m[g] > m[g].median()).astype(int)
    fit(m, [g + '_hi'], f'TCGA {g} median split (marker)')
    fit(m, [g + '_z'], f'TCGA {g} per SD (marker)')
    fit(m, [g + '_z', 'age', 'male', 'gcimp'], f'TCGA {g} per SD adjusted (marker)')

# ---------------- CGGA ----------------
print('[CGGA]')
expr = pd.read_csv(os.path.join(DATA, 'CGGA', 'CGGA.mRNAseq_693.RSEM-genes.20200506.txt'),
                   sep='\t', index_col=0)
expr = np.log2(expr + 1)
sc2 = score(expr, 'CGGA')
sc2.to_csv(os.path.join(BASE, 'cgga_marker_scores.csv'))
cl = pd.read_csv(os.path.join(DATA, 'CGGA', 'CGGA.mRNAseq_693_clinical.20200506.txt'), sep='\t')
cl.columns = [c.strip() for c in cl.columns]
cl = cl.set_index('CGGA_ID')
sc2.index = [str(i).replace('.', '-') for i in sc2.index]
cc = pd.DataFrame({
    'time': pd.to_numeric(cl['OS'], errors='coerce'),
    'event': pd.to_numeric(cl['Censor (alive=0; dead=1)'], errors='coerce'),
    'age': pd.to_numeric(cl['Age'], errors='coerce'),
    'male': (cl['Gender'].astype(str).str.lower().str.startswith('m')).astype(float),
    'grade': cl['Grade'].astype(str).map({'WHO II': 2, 'WHO III': 3, 'WHO IV': 4}).astype(float),
    'idh_mut': (cl['IDH_mutation_status'].astype(str).str.lower() == 'mutant').astype(float),
    'chemo': pd.to_numeric(cl['Chemo_status (TMZ treated=1;un-treated=0)'], errors='coerce'),
}, index=cl.index)
d = sc2.join(cc, how='inner')
d = d[(d.time > 0)].dropna(subset=['gd_T', 'ab_DNT', 'time', 'event'])
print(f'  all-glioma n={len(d)}, events={int(d.event.sum())}')
for g in ['gd_T', 'ab_DNT']:
    d[g + '_z'] = (d[g] - d[g].mean()) / d[g].std()
    d[g + '_hi'] = (d[g] > d[g].median()).astype(int)
    fit(d, [g + '_hi'], f'CGGA all {g} median split (marker)')
    fit(d, [g + '_z'], f'CGGA all {g} per SD (marker)')
    fit(d, [g + '_z', 'age', 'male', 'grade', 'idh_mut', 'chemo'], f'CGGA all {g} adjusted (marker)')
g4 = d[d.grade == 4]
print(f'  GBM n={len(g4)}, events={int(g4.event.sum())}')
for g in ['gd_T', 'ab_DNT']:
    fit(g4, [g + '_hi'], f'CGGA GBM {g} median split (marker)')
    fit(g4, [g + '_z'], f'CGGA GBM {g} per SD (marker)')
    fit(g4, [g + '_z', 'age', 'male', 'idh_mut', 'chemo'], f'CGGA GBM {g} adjusted (marker)')

out = pd.DataFrame(rows)
out.to_csv(os.path.join(BASE, 'robustness_marker_survival.csv'), index=False)
print('\nwritten robustness_marker_survival.csv')
