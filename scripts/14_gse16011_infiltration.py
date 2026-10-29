# -*- coding: utf-8 -*-
"""GSE16011 (Murat 2008) RMA matrix -> NNLS deconvolution -> infiltration validation."""
import pandas as pd, numpy as np, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE163108')
DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'GSE16011')

# ---- load signature matrix (genes x 3, built in 07) ----
SIG = pd.read_csv(os.path.join(BASE, 'signature_matrix.csv'), index_col=0)
sig_genes = SIG.index.astype(str).tolist()

# ---- load RMA expression ----
expr = pd.read_csv(os.path.join(DATA, 'GSE16011_rma_gene.tsv'), sep='\t', index_col=0)
if 'PROBEID' in expr.columns:                       # R write.table row-names artifact
    expr = expr.set_index('PROBEID')
expr = expr.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any')
expr.columns = [c.replace('.CEL.gz', '').replace('(', '').replace(')', '') for c in expr.columns]
print('[expr]', expr.shape)

# signature matrix is highly collinear on bulk (r 0.90-0.97) -> NNLS degenerate.
# Use specificity-weighted marker scoring instead (CIBERSORT-lite):
# gene weight = max(0, its class value - max(other class values))
common = [g for g in sig_genes if g in expr.index]
E = expr.loc[common]
sub = SIG.loc[common]
weights = {}
for ci, c in enumerate(SIG.columns):
    others = sub.drop(columns=c).max(axis=1)
    w = (sub[c] - others).clip(lower=0)
    weights[c] = w
scores = pd.DataFrame(index=expr.columns)
for c, w in weights.items():
    top = w.nlargest(50).index
    scores[c] = E.loc[top].mean(axis=0)
print('[marker genes per class]:', {c: int((w > 0).sum()) for c, w in weights.items()})
print(scores.describe().T[['mean', '50%', 'max']].round(3))
scores.to_csv(os.path.join(DATA, 'gse16011_marker_scores.csv'))


# ---- sample histology from GSM titles ----
import re, urllib.request, ssl, gzip, io
ctx = ssl.create_default_context()
url = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE16011&targ=gsm&form=text&view=brief'
with urllib.request.urlopen(url, context=ctx, timeout=300) as r:
    txt = r.read().decode()
txt = txt.replace('\r\n', '\n')
hist = {}
for block in txt.split('^SAMPLE = ')[1:]:
    gsm = block.split('\n')[0].strip()
    m_t = re.search(r'!Sample_title = (.+)', block)
    m_h = re.findall(r'!Sample_characteristics_ch1 = histology: (.+)', block)
    t = (m_t.group(1) if m_t else '').lower()
    h = (m_h[0] if m_h else '').lower()
    tag = t + ' | ' + h
    if 'control' in tag:
        hist[gsm] = 'Control'
    elif ('glioblastoma' in tag or 'gbm' in tag or 'grade iv' in tag):
        hist[gsm] = 'GBM'
    elif 'grade iii' in tag:
        hist[gsm] = 'Grade III'
    elif 'grade ii' in tag:
        hist[gsm] = 'Grade II'
    else:
        hist[gsm] = 'Glioma'
hser = pd.Series(hist)
print(hser.value_counts().to_dict())

# GSM ids appear in expr columns
scores['histology'] = [hist.get(c.replace('.CEL.gz',''), 'Unknown') for c in scores.index]
print('[matched]', (scores['histology'] != 'Unknown').sum(), '/', len(scores))

# ---- stats + plot ----
from scipy import stats
groups = [g for g in ['Control', 'Grade II', 'Grade III', 'GBM'] if (scores['histology'] == g).sum() > 3]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, cell in zip(axes, ['ab_DNT', 'gd_T', 'conv_T']):
    data = [scores.loc[scores['histology'] == g, cell].values for g in groups]
    bp = ax.boxplot(data, tick_labels=groups, showfliers=False, patch_artist=True)
    for p in bp['boxes']:
        p.set_facecolor('#aec7e8')
    for i, d in enumerate(data):
        ax.scatter(np.random.normal(i + 1, 0.06, len(d)), d, s=8, c='k', alpha=0.4, zorder=3)
    # trend test (GBM vs Control)
    if 'GBM' in groups and 'Control' in groups:
        u, p = stats.mannwhitneyu(
            scores.loc[scores['histology'] == 'GBM', cell],
            scores.loc[scores['histology'] == 'Control', cell])
        ax.set_title(f'{cell}\nGBM vs Control p={p:.2g}')
    else:
        ax.set_title(cell)
    ax.set_ylabel('marker score (log2 expr)')
    ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'figs', 'fig17_gse16011_infiltration.png'), dpi=150)
print('[done] fig17 saved')
