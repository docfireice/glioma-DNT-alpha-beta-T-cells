# -*- coding: utf-8 -*-
"""Module 2: TCR deep analysis - clonotype sharing between DNT and conv_T,
clone size distribution, expansion vs functional signatures."""
import pandas as pd, numpy as np, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'GSE163108')
FIGS = os.path.join(BASE, 'figs'); os.makedirs(FIGS, exist_ok=True)
df = pd.read_csv(os.path.join(BASE, 'tcr_cell_level.csv'), index_col=0)

# keep cells with clonotype info
tc = df[df['clonotype_id'].notna() & (df['clonotype_id'].astype(str) != '')].copy()
print(f'[tcr] cells with clonotype: {len(tc)} / {len(df)}')

gates = ['ab_DNT', 'conv_T', 'gd_T']
sets = {g: set(tc.loc[tc['gate'] == g, 'clonotype_id']) for g in gates}
print({g: len(s) for g, s in sets.items()})

# ---- clone size distribution per gate (log scale) ----
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, g in zip(axes, gates):
    cs = tc.loc[tc['gate'] == g, 'clonotype_size']
    ax.hist(np.log10(cs), bins=40, color='#4878CF', edgecolor='white')
    ax.axvline(np.log10(10), color='r', ls='--', lw=1)
    ax.set_title(f'{g} (n={len(cs)})'); ax.set_xlabel('log10 clone size'); ax.set_ylabel('cells')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig5_tcr_clone_size.png'), dpi=150); plt.close()

# ---- clonotype sharing (Venn-style bars) ----
dnt_ids, conv_ids, gd_ids = sets['ab_DNT'], sets['conv_T'], sets['gd_T']
shared = dnt_ids & conv_ids
private = dnt_ids - conv_ids
# cells of ab_DNT in shared vs private clones
dnt_cells = tc[tc['gate'] == 'ab_DNT']
frac_shared_cells = dnt_cells['clonotype_id'].isin(shared).mean()
n_shared, n_priv = len(shared), len(private)

# expansion: clone size >= 10 considered expanded (on any gate)
tc['expanded'] = tc['clonotype_size'] >= 10
exp_rate = tc.groupby('gate')['expanded'].mean() * 100

# per-gate: fraction of cells in expanded clones
tab = pd.DataFrame({
    'gate': gates,
    'n_clonotypes': [len(sets[g]) for g in gates],
    'pct_cells_in_expanded_clones': [tc.loc[tc.gate == g, 'expanded'].mean() * 100 for g in gates],
})
tab.to_csv(os.path.join(BASE, 'tcr_expansion_summary.csv'), index=False)
print(tab)

# ---- signature score vs clone size (DNT) ----
sig_cols = ['exhaustion', 'cytotoxicity', 'tissue_residency', 'proliferation']
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, c in zip(axes, sig_cols):
    x = np.log10(dnt_cells['clonotype_size'].clip(lower=1))
    y = dnt_cells[c]
    mask = x.notna() & y.notna()
    rho, p = stats.spearmanr(x[mask], y[mask])
    ax.scatter(x[mask], y[mask], s=4, alpha=0.4, color='#4878CF')
    ax.set_title(f'{c} (rho={rho:.2f}, p={p:.1e})')
    ax.set_xlabel('log10 clone size'); ax.set_ylabel(c)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig6_tcr_sig_vs_clonesize.png'), dpi=150); plt.close()

# ---- sharing bar figure ----
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['shared\nwith conv_T', 'DNT-private'], [n_shared, n_priv], color=['#4878CF', '#D65F5F'])
for i, v in enumerate([n_shared, n_priv]):
    ax.text(i, v + max(n_shared, n_priv) * 0.02, str(v), ha='center')
ax.set_ylabel('clonotypes'); ax.set_title(f'ab_DNT clonotypes (cells: {frac_shared_cells*100:.1f}% in shared clones)')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig7_tcr_sharing.png'), dpi=150); plt.close()

# save per-clonotype sharing table
share_df = pd.DataFrame({'clonotype_id': list(dnt_ids),
                         'in_conv_T': [cid in conv_ids for cid in dnt_ids],
                         'in_gdT': [cid in gd_ids for cid in dnt_ids]})
share_df.to_csv(os.path.join(BASE, 'tcr_dnt_sharing.csv'), index=False)
print(f'[done] shared={n_shared} private={n_priv} frac_shared_cells={frac_shared_cells:.3f}')
