# -*- coding: utf-8 -*-
"""Final survival table with an explicit, defensible policy:
degenerate NNLS fits (all-zero solution) are treated as MISSING, not as zero infiltration.
Reports: median-split Cox, continuous per-SD Cox, and multivariable-adjusted per-SD Cox
for TCGA-GBM, CGGA all-glioma and CGGA GBM.
"""
import os, warnings
import numpy as np, pandas as pd
from scipy.optimize import nnls
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', 'results', 'GSE163108')
DATA = os.path.join(HERE, '..', 'data')
rows = []


def fit(d, cols, label, penalizer=0.01):
    dd = d[['time', 'event'] + cols].dropna()
    keep = [c for c in cols if dd[c].nunique() > 1]
    if len(keep) < len(cols):
        print(f'  [{label}] dropped constant covariate(s): {set(cols) - set(keep)}')
        cols = keep
    if len(dd) < 20 or dd.event.sum() < 10 or not cols:
        print(f'  [{label}] skipped n={len(dd)}'); return None
    try:
        cph = CoxPHFitter(penalizer=penalizer).fit(dd, 'time', 'event')
    except Exception as e:
        print(f'  [{label}] fit failed: {type(e).__name__}')
        return None
    s = cph.summary.loc[cols[0]]
    r = dict(model=label, n=len(dd), events=int(dd.event.sum()),
             HR=s['exp(coef)'], lo=s['exp(coef) lower 95%'], hi=s['exp(coef) upper 95%'], p=s['p'])
    print(f"  {label:42s} n={r['n']:4d} HR={r['HR']:.2f} ({r['lo']:.2f}-{r['hi']:.2f}) p={r['p']:.3g}")
    rows.append(r)
    return r


# ---------------- TCGA ----------------
print('[TCGA]')
frac = pd.read_csv(os.path.join(BASE, 'tcga_deconv_fractions.csv'), index_col=0)
frac = frac.dropna()                      # degenerate fits are NaN already
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
frac['patient'] = [str(i)[:12] for i in frac.index]
m = frac.drop_duplicates('patient').set_index('patient').join(tc, how='inner')
m = m[(m.time > 0)].dropna(subset=['gd_T', 'ab_DNT', 'time', 'event'])
print(f'  analytic n={len(m)}, events={int(m.event.sum())}')
for g in ['gd_T', 'ab_DNT']:
    gz = (m[g] - m[g].mean()) / m[g].std()
    m[g + '_z'] = gz
    grp = (m[g] > m[g].median()).astype(int)
    m[g + '_hi'] = grp
    fit(m, [g + '_hi'], f'TCGA {g} median split')
    fit(m, [g + '_z'], f'TCGA {g} per SD univariable')
    fit(m, [g + '_z', 'age', 'male', 'gcimp'], f'TCGA {g} per SD adjusted')

# ---------------- CGGA ----------------
print('[CGGA]')
SIG = pd.read_csv(os.path.join(BASE, 'signature_matrix.csv'), index_col=0)
genes = list(SIG.index)
expr = pd.read_csv(os.path.join(DATA, 'CGGA', 'CGGA.mRNAseq_693.RSEM-genes.20200506.txt'),
                   sep='\t', index_col=0)
common = [g for g in genes if g in expr.index]
Xl = np.log2(expr.loc[common].T.astype(float) + 1)
bz = (Xl - Xl.mean(0)) / (Xl.std(0) + 1e-9)
sz = (SIG.loc[common] - SIG.loc[common].mean(0)) / (SIG.loc[common].std(0) + 1e-9)
fr = np.apply_along_axis(lambda y: nnls(sz, y)[0], 1, bz.values)
cg = pd.DataFrame(fr, index=[str(i).replace('.', '-') for i in expr.columns],
                  columns=['ab_DNT', 'gd_T', 'conv_T'])
cg = cg.div(cg.sum(1), axis=0)            # do NOT fillna(0): degenerate fits stay missing
print(f'  degenerate NNLS fits treated as missing: {int(cg.isna().any(axis=1).sum())}/{len(cg)}')
cl = pd.read_csv(os.path.join(DATA, 'CGGA', 'CGGA.mRNAseq_693_clinical.20200506.txt'), sep='\t')
cl.columns = [c.strip() for c in cl.columns]
cl = cl.set_index('CGGA_ID'); cl.index = [str(i).replace('.', '-') for i in cl.index]
cc = pd.DataFrame({
    'time': pd.to_numeric(cl['OS'], errors='coerce'),
    'event': pd.to_numeric(cl['Censor (alive=0; dead=1)'], errors='coerce'),
    'age': pd.to_numeric(cl['Age'], errors='coerce'),
    'male': (cl['Gender'].astype(str).str.lower().str.startswith('m')).astype(float),
    'grade': cl['Grade'].astype(str).map({'WHO II': 2, 'WHO III': 3, 'WHO IV': 4}).astype(float),
    'idh_mut': (cl['IDH_mutation_status'].astype(str).str.lower() == 'mutant').astype(float),
    'chemo': pd.to_numeric(cl['Chemo_status (TMZ treated=1;un-treated=0)'], errors='coerce'),
}, index=cl.index)
d = cg.join(cc, how='inner')
d = d[(d.time > 0)].dropna(subset=['gd_T', 'ab_DNT', 'time', 'event'])
print(f'  all-glioma analytic n={len(d)}, events={int(d.event.sum())}')
for g in ['gd_T', 'ab_DNT']:
    d[g + '_z'] = (d[g] - d[g].mean()) / d[g].std()
    d[g + '_hi'] = (d[g] > d[g].median()).astype(int)
    fit(d, [g + '_hi'], f'CGGA all {g} median split')
    fit(d, [g + '_z'], f'CGGA all {g} per SD univariable')
    fit(d, [g + '_z', 'age', 'male', 'grade', 'idh_mut', 'chemo'], f'CGGA all {g} per SD adjusted')
g4 = d[d.grade == 4]
print(f'  GBM analytic n={len(g4)}, events={int(g4.event.sum())}')
for g in ['gd_T', 'ab_DNT']:
    fit(g4, [g + '_hi'], f'CGGA GBM {g} median split')
    fit(g4, [g + '_z'], f'CGGA GBM {g} per SD univariable')
    fit(g4, [g + '_z', 'age', 'male', 'idh_mut', 'chemo'], f'CGGA GBM {g} per SD adjusted')

out = pd.DataFrame(rows)
out.to_csv(os.path.join(BASE, 'robustness_final_survival.csv'), index=False)
print(f'\nwritten {os.path.join(BASE, "robustness_final_survival.csv")}')
