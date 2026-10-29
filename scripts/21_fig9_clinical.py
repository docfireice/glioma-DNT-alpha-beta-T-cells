# -*- coding: utf-8 -*-
"""Figure 9 (replacement for the withdrawn survival figure): cross-sectional
grade association of gamma-delta T-cell content, anchored on the lineage-defining
transcript TRDC, with explicit specificity (T-cell-load adjusted) models.

Panels
  A  CGGA: gd_T / ab_DNT / conv_T signature score by WHO grade
  B  TRDC (z) by WHO grade in both cohorts (mean +/- 95% CI)
  C  Signature score vs TRDC in CGGA (concordance of two quantifications)
  D  Forest: per-grade-step beta, unadjusted and T-cell-load adjusted
  E  Relative enrichment (gd_T - conv_T) by grade
"""
import os, warnings
import numpy as np, pandas as pd, gzip
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(ROOT, 'results', 'GSE163108')
DATA = os.path.join(ROOT, 'data')
FIG = os.path.join(RES, 'figs')

# ---------------- data ----------------
cg = pd.read_csv(os.path.join(RES, 'cgga_marker_scores.csv'), index_col=0)
cg.index = [str(i).replace('.', '-') for i in cg.index]
cl = pd.read_csv(os.path.join(DATA, 'CGGA', 'CGGA.mRNAseq_693_clinical.20200506.txt'), sep='\t')
cl.columns = [c.strip() for c in cl.columns]
cl = cl.set_index('CGGA_ID')
cgga = cg.join(pd.DataFrame({
    'grade': cl['Grade'].astype(str).map({'WHO II': 2, 'WHO III': 3, 'WHO IV': 4}).astype(float),
    'idh_mut': (cl['IDH_mutation_status'].astype(str).str.lower() == 'mutant').astype(float),
    'age': pd.to_numeric(cl['Age'], errors='coerce'),
    'male': cl['Gender'].astype(str).str.lower().str.startswith('m').astype(float),
}, index=cl.index), how='inner').dropna(subset=['grade'])

e = pd.read_csv(os.path.join(DATA, 'CGGA', 'CGGA.mRNAseq_693.RSEM-genes.20200506.txt'), sep='\t', index_col=0)
e = np.log2(e + 1)
e.columns = [str(c).replace('.', '-') for c in e.columns]
cgga['TRDC'] = e.loc['TRDC']
cgga['CD3D'] = e.loc['CD3D']
cgga['SPP1_CD44'] = (e.loc['SPP1'] - e.loc['SPP1'].mean()) / e.loc['SPP1'].std() \
    + (e.loc['CD44'] - e.loc['CD44'].mean()) / e.loc['CD44'].std()

g11 = pd.read_csv(os.path.join(DATA, 'GSE16011', 'gse16011_marker_scores.csv'), index_col=0)
g11.index = [str(i).replace('.CEL.gz', '').replace('(', '').replace(')', '').strip() for i in g11.index]
sdrf = pd.read_csv(os.path.join(DATA, 'GSE16011', 'E-GEOD-16011.sdrf.txt'), sep='\t')
hcol = [c for c in sdrf.columns if 'histology' in c][0]
fcol = [c for c in sdrf.columns if 'Array Data File' in c][0]
acol = [c for c in sdrf.columns if 'age at diagnosis' in c][0]
gcol = [c for c in sdrf.columns if 'gender' in c][0]
GM = {'PA (grade I)': 1, 'A (grade II)': 2, 'OA (grade II)': 2, 'OD (grade II)': 2,
      'A (grade III)': 3, 'OA (grade III)': 3, 'OD (grade III)': 3, 'GBM (grade IV)': 4, 'control': 0}
ann = pd.DataFrame({'gsm': sdrf[fcol].astype(str).str.replace('.CEL', '', regex=False).str.strip(),
                    'grade': sdrf[hcol].astype(str).str.strip().map(GM),
                    'age': pd.to_numeric(sdrf[acol], errors='coerce'),
                    'male': (sdrf[gcol].astype(str).str.lower().str[0] == 'm').astype(float)}
                   ).drop_duplicates('gsm').set_index('gsm')
g11 = g11.join(ann, how='inner')
e2 = pd.read_csv(os.path.join(DATA, 'GSE16011', 'GSE16011_rma_gene.tsv'), sep='\t', index_col=0)
if 'PROBEID' in e2.columns:
    e2 = e2.set_index('PROBEID')
e2 = e2.apply(pd.to_numeric, errors='coerce').dropna(how='any')
e2.columns = [str(c).replace('.CEL.gz', '').replace('(', '').replace(')', '').strip() for c in e2.columns]
e2 = e2.groupby(level=0).mean()
g11['TRDC'] = e2.loc['TRDC']
g11['CD3D'] = e2.loc['CD3D']


def z(s):
    return (s - s.mean()) / s.std()


# ---------------- statistics ----------------
def beta(df, y, cov):
    d = df[['grade', y] + cov].dropna()
    X = sm.add_constant(d[['grade'] + cov].astype(float))
    m = sm.OLS(d[y].astype(float), X).fit()
    ci = m.conf_int().loc['grade']
    return m.params['grade'], ci[0], ci[1], m.pvalues['grade'], len(d)


rows = []
# IDH status is a molecular correlate of grade and a potential mediator of the
# grade-immune relationship, so it is NOT part of the adjustment set here;
# it is analysed separately as an association (section 2 of 20_clinical_endpoints).
for tag, df, base in [('CGGA', cgga, ['age', 'male']), ('GSE16011', g11, ['age', 'male'])]:
    for met, col in [('signature', 'gd_T'), ('TRDC', 'TRDC')]:
        for mod, cov in [('unadjusted', []), ('adj. age/sex', base),
                         ('adj. + T-cell load', base + ['CD3D'])]:
            cov = [c for c in cov if c in df.columns]
            b, lo, hi, p, n = beta(df, col, cov)
            rows.append(dict(cohort=tag, metric=met, model=mod, beta=b, lo=lo, hi=hi, p=p, n=n))
fdf = pd.DataFrame(rows)
fdf.to_csv(os.path.join(RES, 'fig9_forest_source.csv'), index=False)

# ---------------- figure ----------------
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.8, 'xtick.major.width': 0.8,
                     'ytick.major.width': 0.8})
fig = plt.figure(figsize=(11.5, 7.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.55, wspace=0.42,
                      left=0.075, right=0.985, top=0.93, bottom=0.09)

COL = {'$\\gamma\\delta$ T': '#d62728', 'DN $\\alpha\\beta$ T': '#1f77b4', 'conv. T': '#7f7f7f'}

# A -------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
pops = [('gd_T', '$\\gamma\\delta$ T'), ('ab_DNT', 'DN $\\alpha\\beta$ T'), ('conv_T', 'conv. T')]
for i, (pop, lab) in enumerate(pops):
    data = [cgga.loc[cgga.grade == g, pop].dropna().values for g in [2, 3, 4]]
    pos = np.arange(3) + (i - 1) * 0.25
    bp = ax.boxplot(data, positions=pos, widths=0.21, showfliers=False, patch_artist=True,
                    medianprops=dict(color='k', lw=0.9),
                    whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
    for b in bp['boxes']:
        b.set_facecolor(COL[lab]); b.set_alpha(0.8); b.set_edgecolor('k'); b.set_linewidth(0.6)
ax.set_xticks(range(3)); ax.set_xticklabels(['II', 'III', 'IV'])
ax.set_xlabel('WHO grade'); ax.set_ylabel('signature score (log$_2$)')
ax.set_title('CGGA, n = 692', loc='left', fontsize=9, fontweight='bold')
ax.legend([plt.Rectangle((0, 0), 1, 1, fc=COL[l], ec='k', lw=0.5, alpha=0.8) for _, l in pops],
          [l for _, l in pops], fontsize=6.5, frameon=False, loc='lower right', handlelength=1.1)

# B -------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
for tag, df, c, mk in [('CGGA', cgga, '#d62728', 'o'), ('GSE16011', g11[g11.grade >= 1], '#ff7f0e', 's')]:
    gr = sorted(df.grade.unique())
    zz = z(df['TRDC'])
    m = [zz[df.grade == g].mean() for g in gr]
    se = [zz[df.grade == g].std(ddof=1) / np.sqrt((df.grade == g).sum()) for g in gr]
    ax.errorbar(gr, m, yerr=[1.96 * s for s in se], marker=mk, color=c, label=f'{tag} (n={len(df)})',
                capsize=3, lw=1.2, ms=4)
ax.axhline(0, ls=':', c='grey', lw=0.7)
ax.set_xlabel('WHO grade'); ax.set_ylabel('TRDC (z-score)')
ax.set_xticks([1, 2, 3, 4])
ax.set_title('Lineage transcript TRDC', loc='left', fontsize=9, fontweight='bold')
ax.legend(fontsize=6.5, frameon=False, loc='upper left')

# C -------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 2])
sc = ax.scatter(z(cgga['TRDC']), cgga['gd_T'], c=cgga['grade'], cmap='viridis', s=8, alpha=0.7,
                linewidths=0)
r, p = stats.spearmanr(cgga['TRDC'], cgga['gd_T'])
ax.set_xlabel('TRDC (z)'); ax.set_ylabel('$\\gamma\\delta$ signature score')
ax.set_title(f'CGGA concordance\n$\\rho$ = {r:.2f}, P = {p:.0e}', loc='left', fontsize=9,
             fontweight='bold')
cb = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02); cb.set_label('grade', fontsize=7)
cb.ax.tick_params(labelsize=6)

# D -------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, :2])
lab_map = {'unadjusted': 'unadj.', 'adj. age/sex': 'adj. age/sex', 'adj. + T-cell load': 'adj. + T-cell'}
yl, yv, ylo, yhi, col = [], [], [], [], []
order = [('CGGA', 'signature'), ('CGGA', 'TRDC'), ('GSE16011', 'signature'), ('GSE16011', 'TRDC')]
show = list(fdf[(fdf.model == 'adj. age/sex')].index)
sel = fdf[(fdf.model != 'unadjusted')]
for _, r_ in sel.iterrows():
    nm = f"{r_.cohort}  {'signature' if r_.metric == 'signature' else 'TRDC'}"
    yl.append(f"{nm}  ({lab_map[r_.model]})")
    yv.append(r_.beta); ylo.append(r_.lo); yhi.append(r_.hi)
    col.append('#d62728' if r_.metric == 'signature' else '#2ca02c')
y = np.arange(len(yv))[::-1]
ax.axvline(0, ls='--', c='grey', lw=0.8)
for xi, yi, lo_, hi_, c_ in zip(yv, y, ylo, yhi, col):
    ax.plot([lo_, hi_], [yi, yi], '-', color=c_, lw=1.1)
    ax.plot(xi, yi, 'o', color=c_, ms=4.5, mec='k', mew=0.5)
ax.set_yticks(y); ax.set_yticklabels(yl, fontsize=6.5)
ax.set_xlabel('$\\beta$ per WHO grade step (log$_2$ units, 95% CI)')
ax.set_title('Per-grade-step effect (OLS)', loc='left', fontsize=9, fontweight='bold')
h1 = plt.Line2D([], [], marker='o', ls='-', color='#d62728', label='signature score', ms=4)
h2 = plt.Line2D([], [], marker='o', ls='-', color='#2ca02c', label='TRDC', ms=4)
ax.legend(handles=[h1, h2], fontsize=6.5, frameon=False, loc='upper right')

# E -------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 2])
for tag, df, c, mk in [('CGGA', cgga, '#d62728', 'o'), ('GSE16011', g11[g11.grade >= 1], '#ff7f0e', 's')]:
    d = df.copy(); d['rel'] = d['gd_T'] - d['conv_T']
    gr = sorted(d.grade.unique())
    m = [d.loc[d.grade == g, 'rel'].mean() for g in gr]
    se = [d.loc[d.grade == g, 'rel'].std(ddof=1) / np.sqrt((d.grade == g).sum()) for g in gr]
    ax.errorbar(gr, m, yerr=[1.96 * s for s in se], marker=mk, color=c, capsize=3, lw=1.2, ms=4,
                label=tag)
ax.axhline(0, ls='--', c='grey', lw=0.8)
ax.set_xlabel('WHO grade'); ax.set_ylabel('$\\gamma\\delta$ $-$ conv. T (log$_2$)')
ax.set_xticks([1, 2, 3, 4])
ax.set_title('Relative enrichment', loc='left', fontsize=9, fontweight='bold')
ax.legend(fontsize=6.5, frameon=False, loc='upper left')

out = os.path.join(FIG, 'fig9_clinical_clean.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
print('written', out)
print(fdf.round(4).to_string(index=False))
