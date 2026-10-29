# -*- coding: utf-8 -*-
"""Lightweight end-to-end pipeline (numpy/pandas only; no scanpy needed).
Steps: QC -> log-normalize -> HVG -> PCA -> KMeans clustering ->
CD3+ gating -> DNT / gamma-delta gate -> marker DEG (Mann-Whitney U)
Usage: python 02b_pipeline_numpy.py <csv> <labels_csv> <outdir>
"""
import sys, os
import numpy as np, pandas as pd

def mannwhitney_u(x, y):
    n1, n2 = len(x), len(y)
    allv = np.concatenate([x, y])
    ranks = pd.Series(allv).rank().values
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u1 - mu) / (sigma if sigma > 0 else 1e-9)
    return u1, z

def main(csv, labels_csv, outdir):
    os.makedirs(outdir, exist_ok=True)
    X = pd.read_csv(csv, index_col=0)
    genes = X.columns.str.upper()
    print(f'[load] cells={X.shape[0]} genes={X.shape[1]}')

    # ---- QC (Hao 2026 thresholds) ----
    mt = genes.str.startswith('MT-')
    pct_mt = X.loc[:, mt].sum(1) / X.sum(1) * 100
    n_genes = (X > 0).sum(1)
    keep = (pct_mt < 10) & (n_genes >= 3)
    X = X.loc[keep]
    print(f'[qc] cells={X.shape[0]} (mito<10%, min_genes)')

    # ---- normalize & log1p ----
    lib = X.sum(1)
    Ln = np.log1p(X.div(lib, axis=0) * 1e4)
    Ln = Ln.loc[:, Ln.var(0) > 0]

    # ---- HVG (top variance) + PCA via SVD ----
    hvg = Ln.var(0).nlargest(min(1000, Ln.shape[1])).index
    M = Ln[hvg].sub(Ln[hvg].mean(0), axis=1).values
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    P = M @ Vt.T[:, :15]  # PCs

    # ---- KMeans clustering (numpy) ----
    k = 8
    rng = np.random.default_rng(0)
    C = P[rng.choice(len(P), k, replace=False)]
    for _ in range(50):
        lab = ((P[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([P[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    print(f'[cluster] clusters={len(set(lab))}')

    # ---- gating: CD3+ -> DNT / gamma-delta ----
    gi = {g: i for i, g in enumerate(genes)}
    def col(g): return X.iloc[:, gi[g]].values if g in gi else None
    cd3 = np.zeros(len(X), bool)
    for g in ('CD3D', 'CD3E', 'CD3G'):
        c = col(g)
        if c is not None: cd3 |= c > 0
    cd4, cd8a, cd8b = col('CD4'), col('CD8A'), col('CD8B')
    trdc, trdv1, trdv2 = col('TRDC'), col('TRDV1'), col('TRDV2')
    dnt = cd3.copy()
    if cd4 is not None: dnt &= (cd4 == 0)
    if cd8a is not None: dnt &= (cd8a == 0)
    if cd8b is not None: dnt &= (cd8b == 0)
    gd = np.zeros(len(X), bool)
    for c in (trdc, trdv1, trdv2):
        if c is not None: gd |= c > 0

    gate = np.full(len(X), 'non_T', object)
    gate[cd3 & ~dnt] = 'other_T'
    gate[dnt & ~gd] = 'ab_DNT'
    gate[dnt & gd] = 'gamma_delta_T'
    meta = pd.DataFrame({'leiden_kmeans': lab, 'celltype_gate': gate}, index=X.index)
    print('[gate]\n', meta.celltype_gate.value_counts().to_string())

    # ---- DEG per cluster vs rest (top10) ----
    Lg = Ln.values
    rows = {}
    for cl in sorted(set(lab)):
        a = Lg[lab == cl]; b = Lg[lab != cl]
        scores = []
        for j in range(Lg.shape[1]):
            _, z = mannwhitney_u(a[:, j], b[:, j])
            scores.append(z)
        top = np.argsort(scores)[::-1][:10]
        rows[f'cluster_{cl}'] = [Ln.columns[j] for j in top]
    pd.DataFrame(rows).to_csv(os.path.join(outdir, 'top10_markers_per_cluster.csv'), index=False)

    # ---- accuracy vs true labels ----
    if os.path.exists(labels_csv):
        t = pd.read_csv(labels_csv, index_col=0).iloc[:, 0]
        common = meta.index.intersection(t.index)
        tm = t.loc[common].astype(str); lm = meta.loc[common, 'leiden_kmeans'].astype(str)
        # permutation best-match accuracy
        acc = max(
            (lm[tm == lab_true] == cl).mean()
            for lab_true in tm.unique() for cl in lm.unique()
        )
        purity = np.mean([ (lm[tm == u].value_counts().iloc[0] / max((tm == u).sum(),1)) for u in tm.unique() ])
        print(f'[eval] best-match accuracy≈{acc:.2f}  mean cluster purity={purity:.2f}')

    meta.to_csv(os.path.join(outdir, 'cell_metadata.csv'))
    print('[done]', outdir)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
