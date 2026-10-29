# -*- coding: utf-8 -*-
"""Addendum: PH assumption per term + grade-stratified Cox (CGGA) + median-split reconciliation."""
import os, warnings
import numpy as np, pandas as pd
from scipy.optimize import nnls
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', 'results', 'GSE163108')
DATA = os.path.join(HERE, '..', 'data')
LOG = []


def log(s):
    print(s, flush=True); LOG.append(s)


SIG = pd.read_csv(os.path.join(BASE, 'signature_matrix.csv'), index_col=0)
genes = list(SIG.index)
expr = pd.read_csv(os.path.join(DATA, 'CGGA', 'CGGA.mRNAseq_693.RSEM-genes.20200506.txt'),
                   sep='\t', index_col=0)
common = [g for g in genes if g in expr.index]
Xl = np.log2(expr.loc[common].T.astype(float) + 1)
bulk_z = (Xl - Xl.mean(0)) / (Xl.std(0) + 1e-9)
sig_z = (SIG.loc[common] - SIG.loc[common].mean(0)) / (SIG.loc[common].std(0) + 1e-9)
fr = np.apply_along_axis(lambda y: nnls(sig_z, y)[0], 1, bulk_z.values)
cg = pd.DataFrame(fr, index=[str(i).replace('.', '-') for i in expr.columns],
                  columns=['ab_DNT', 'gd_T', 'conv_T'])
cg = cg.div(cg.sum(1), axis=0).fillna(0)
cg.to_csv(os.path.join(BASE, 'cgga_deconv_fractions.csv'))

cl = pd.read_csv(os.path.join(DATA, 'CGGA', 'CGGA.mRNAseq_693_clinical.20200506.txt'), sep='\t')
cl.columns = [c.strip() for c in cl.columns]
cl = cl.set_index('CGGA_ID')
cl.index = [str(i).replace('.', '-') for i in cl.index]
cc = pd.DataFrame({
    'time': pd.to_numeric(cl['OS'], errors='coerce'),
    'event': pd.to_numeric(cl['Censor (alive=0; dead=1)'], errors='coerce'),
    'age': pd.to_numeric(cl['Age'], errors='coerce'),
    'male': (cl['Gender'].astype(str).str.lower().str.startswith('m')).astype(float),
    'grade': cl['Grade'].astype(str).map({'WHO II': 2, 'WHO III': 3, 'WHO IV': 4}).astype(float),
    'idh_mut': (cl['IDH_mutation_status'].astype(str).str.lower() == 'mutant').astype(float),
    'chemo': pd.to_numeric(cl['Chemo_status (TMZ treated=1;un-treated=0)'], errors='coerce'),
    'rt': pd.to_numeric(cl['Radio_status (treated=1;un-treated=0)'], errors='coerce'),
}, index=cl.index)
d = cg.join(cc, how='inner').dropna(subset=['time', 'event'])
d = d[d.time > 0]
for g in ['gd_T', 'ab_DNT']:
    d[g + '_z'] = (d[g] - d[g].mean()) / d[g].std()
d.to_csv(os.path.join(BASE, 'cgga_merged_clinical.csv'))

log('[PH] per-term proportional-hazards test, CGGA adjusted models')
for g in ['gd_T', 'ab_DNT']:
    dd = d[['time', 'event', g + '_z', 'age', 'male', 'grade', 'idh_mut', 'chemo', 'rt']].dropna()
    cph = CoxPHFitter(penalizer=0.01).fit(dd, 'time', 'event')
    ph = proportional_hazard_test(cph, dd, time_transform='rank').summary
    log(f'  -- {g} --')
    for t, r in ph.iterrows():
        log(f'     {t:10s} p={r["p"]:.4f}')

log('[strat] CGGA all-glioma, Cox stratified by WHO grade')
for g in ['gd_T', 'ab_DNT']:
    dd = d[['time', 'event', g + '_z', 'age', 'male', 'idh_mut', 'chemo', 'rt', 'grade']].dropna()
    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(dd, 'time', 'event', strata=['grade'])
        s = cph.summary.loc[g + '_z']
        log(f'  {g}: HR={s["exp(coef)"]:.2f} ({s["exp(coef) lower 95%"]:.2f}-'
            f'{s["exp(coef) upper 95%"]:.2f}), p={s["p"]:.3g}, n={len(dd)}')
    except Exception as e:
        log(f'  {g}: stratified fit failed ({e})')

log('[median-split] reconciliation with the originally reported HRs')
for g in ['gd_T', 'ab_DNT']:
    dd = d[['time', 'event', g]].dropna()
    hi = dd[g] > dd[g].median()
    dd2 = dd[['time', 'event']].copy(); dd2['grp'] = hi.astype(int)
    cph = CoxPHFitter(penalizer=0.01).fit(dd2, 'time', 'event')
    s = cph.summary.loc['grp']
    log(f'  CGGA all {g}: median-split HR={s["exp(coef)"]:.2f} '
        f'({s["exp(coef) lower 95%"]:.2f}-{s["exp(coef) upper 95%"]:.2f}), p={s["p"]:.3g}')

with open(os.path.join(BASE, 'robustness_addendum.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG))
log('DONE')
