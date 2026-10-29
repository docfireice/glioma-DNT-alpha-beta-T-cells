# -*- coding: utf-8 -*-
"""Module 5: CGGA mRNAseq_693 validation - same NNLS deconvolution + Cox/KM as TCGA."""
import numpy as np, pandas as pd, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE163108')
CGGA = os.path.join(os.path.dirname(__file__), '..', 'data', 'CGGA')
FIGS = os.path.join(BASE, 'figs')
SIG = pd.read_csv(os.path.join(BASE, 'signature_matrix.csv'), index_col=0)
genes = list(SIG.index)

expr = pd.read_csv(os.path.join(CGGA, 'CGGA.mRNAseq_693.RSEM-genes.20200506.txt'), sep='\t', index_col=0)
print('[expr]', expr.shape)
common = [g for g in genes if g in expr.index]
print(f'[sig] matched {len(common)}/{len(genes)}')
Xs = expr.loc[common].T.astype(float)
# log2(RSEM+1), z-score per gene
Xl = np.log2(Xs + 1)
bulk_z = (Xl - Xl.mean(0)) / (Xl.std(0) + 1e-9)
sig_sc = SIG.loc[common]
sig_z = (sig_sc - sig_sc.mean(0)) / (sig_sc.std(0) + 1e-9)
fracs = np.apply_along_axis(lambda y: nnls(sig_z, y)[0], 1, bulk_z.values)
frac = pd.DataFrame(fracs, index=expr.columns, columns=['ab_DNT', 'gd_T', 'conv_T'])
frac = frac.div(frac.sum(1), axis=0).fillna(0.0)
frac.index = frac.index.astype(str).str.replace('.', '-', regex=False)  # CGGA ids use dots
print('[frac]', frac.describe().loc[['50%']].round(3).to_dict())

clin = pd.read_csv(os.path.join(CGGA, 'CGGA.mRNAseq_693_clinical.20200506.txt'), sep='\t')
print('[clin] cols:', [c for c in clin.columns][:20])
clin.columns = [c.strip() for c in clin.columns]
# standard CGGA columns: Censor (alive=0; dead=1), OS
osc = [c for c in clin.columns if c.lower().startswith('censor')]
ost = [c for c in clin.columns if c == 'OS' or c.lower().startswith('survival')]
c_col, t_col = osc[0], ost[0]
clin = clin.set_index(clin.columns[0])
clin['event'] = (clin[c_col] == 1).astype(int)
clin['time'] = pd.to_numeric(clin[t_col], errors='coerce')
m = frac.join(clin[['event', 'time', 'Grade']], how='inner')
m = m.dropna(subset=['event', 'time'])
m['event'] = m['event'].astype(int)
m['time'] = pd.to_numeric(m['time'], errors='coerce')
m = m[m['time'] > 0]
gbm = m[m['Grade'].astype(str).str.contains('IV')] if 'Grade' in m else m
print(f'[cohort] n={len(m)}, events={m.event.sum()}; GBM n={len(gbm)}, events={gbm.event.sum()}')

res = {}
for cohort, df in [('CGGA-all-glioma', m), ('CGGA-GBM', gbm)]:
    for cell in ['gd_T', 'ab_DNT']:
        df = df.copy()
        df[cell] = pd.to_numeric(df[cell], errors='coerce').fillna(0.0)
        med = df[cell].median()
        df = df.copy(); df[cell + '_high'] = (df[cell] > med).astype(int) if med > 0 else (df[cell] > 0).astype(int)
        hi = df[df[cell + '_high'] == 1]; lo = df[df[cell + '_high'] == 0]
        lr = logrank_test(hi['time'], lo['time'], hi['event'], lo['event'])
        cph = CoxPHFitter()
        cph.fit(df[[cell + '_high', 'time', 'event']], duration_col='time', event_col='event')
        hr = np.exp(cph.params_[cell + '_high']); p = cph.summary.loc[cell + '_high', 'p']
        res[f'{cohort}|{cell}'] = (hr, p, lr.p_value, len(df))
        print(f'[cox] {cohort} {cell}: HR={hr:.3f} cox_p={p:.4f} logrank_p={lr.p_value:.4f} n={len(df)}')
        if cohort == 'CGGA-GBM' or (cohort == 'CGGA-all-glioma' and cell == 'gd_T'):
            fig, ax = plt.subplots(figsize=(6, 4.5))
            kmf = KaplanMeierFitter()
            for grp, lbl, c in [(hi, f'{cell} high (n={len(hi)})', '#D65F5F'), (lo, f'{cell} low (n={len(lo)})', '#4878CF')]:
                kmf.fit(grp['time'] / 30.44, grp['event'], label=lbl)
                kmf.plot_survival_function(ax=ax, color=c, ci_show=False)
            ax.set_title(f'{cohort}: {cell} vs OS | logrank p={lr.p_value:.3f}')
            ax.set_xlabel('months'); ax.set_ylabel('OS')
            plt.tight_layout(); plt.savefig(os.path.join(FIGS, f'fig12_km_{cohort}_{cell}.png'), dpi=150); plt.close()

pd.DataFrame(res, index=['HR', 'cox_p', 'logrank_p', 'n']).T.to_csv(os.path.join(BASE, 'cgga_cox_results.csv'))
print('[done] module 5')
