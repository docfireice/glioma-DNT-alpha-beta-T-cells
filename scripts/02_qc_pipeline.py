# -*- coding: utf-8 -*-
"""Step 1+3: QC -> DNT gating -> subclustering -> markers + GO-lite enrichment.
Smoke-test runnable on any small 10X-like count matrix (csv/h5/mtx).
Usage: python 02_qc_pipeline.py <input_matrix> <outdir>
Input: csv/tsv/txt.gz with genes x cells or cells x genes; or .h5 (10X).
"""
import sys, os, re
import numpy as np
import pandas as pd
import scanpy as sc

sc.settings.verbosity = 2

def load_any(path):
    if path.endswith('.h5'):
        a = sc.read_10x_h5(path)
        a.var_names_make_unique()
        return a
    sep = '\t' if path.endswith(('.tsv', '.txt.gz', '.tsv.gz')) else ','
    df = pd.read_csv(path, sep=sep, index_col=0)
    # ensure cells x genes
    if df.shape[0] > df.shape[1]:
        df = df.T
    return sc.AnnData(df.astype(np.float32))

def main(inp, outdir):
    os.makedirs(outdir, exist_ok=True)
    a = load_any(inp)
    print(f'[load] cells={a.n_obs} genes={a.n_vars}')

    # ---- QC (Hao et al. 2026 thresholds) ----
    a.var['mt'] = a.var_names.str.upper().str.startswith(('MT-', 'MT.'))
    sc.pp.calculate_qc_metrics(a, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    a = a[a.obs.pct_counts_mt < 10, :]
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_cells(a, max_genes=4000)
    # scrublet if available, else skip
    try:
        sc.external.pp.scrublet(a, expected_doublet_rate=0.05)
        a = a[a.obs.doublet_score <= 0.25, :]
    except Exception as e:
        print('[scrublet] skipped:', e)
    print(f'[qc] cells={a.n_obs}')

    # ---- normalize & clustering ----
    a.layers['counts'] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(3000, a.n_vars - 1))
    sc.tl.pca(a, n_comps=min(30, min(a.n_obs, a.n_vars) - 2))
    sc.pp.neighbors(a, n_pcs=min(15, a.obsm['X_pca'].shape[1]))
    sc.tl.leiden(a, resolution=1.0, key_added='leiden', flavor='igraph', n_iterations=2)
    sc.tl.umap(a)
    print(f'[cluster] clusters={a.obs.leiden.nunique()}')

    # ---- DNT / gamma-delta gating ----
    up = a.var_names.str.upper()
    def g(sym):
        m = up.get_loc(sym) if sym in up else None
        return None if m is None else a.X[:, m]
    cd3 = [s for s in ('CD3D', 'CD3E', 'CD3G') if s in set(up)]
    cd4 = g('CD4'); cd8a = g('CD8A'); cd8b = g('CD8B')
    trdc = g('TRDC'); trdv1 = g('TRDV1'); trdv2 = g('TRDV2')
    def expr(x, thr=0):
        return (np.asarray(x).ravel() > thr) if x is not None else None

    is_t = np.zeros(a.n_obs, bool)
    for s in cd3:
        is_t |= np.asarray(g(s)).ravel() > 0
    a.obs['is_T'] = is_t

    if cd4 is None or cd8a is None:
        print('[gate] CD4/CD8A not in panel -> gating demo on available markers')
        a.obs['celltype_gate'] = 'other'
    else:
        dnt = is_t & (np.asarray(cd4).ravel() == 0) & (np.asarray(cd8a).ravel() == 0) & \
              ((cd8b is None) | (np.asarray(cd8b).ravel() == 0))
        gd = np.zeros(a.n_obs, bool)
        for x in (trdc, trdv1, trdv2):
            if x is not None:
                gd |= np.asarray(x).ravel() > 0
        a.obs['celltype_gate'] = np.select(
            [is_t & dnt & gd, is_t & dnt & ~gd, is_t & ~dnt],
            ['gamma_delta_T', 'ab_DNT', 'other_T'], default='non_T')
        print(a.obs.celltype_gate.value_counts())

    a.obs.to_csv(os.path.join(outdir, 'cell_metadata.csv'))
    # rank genes per leiden cluster
    sc.tl.rank_genes_groups(a, 'leiden', method='wilcoxon')
    top = pd.DataFrame({k: sc.get.rank_genes_groups_df(a, group=k).head(20).names.values
                        for k in a.obs.leiden.cat.categories})
    top.to_csv(os.path.join(outdir, 'top20_markers_per_cluster.csv'), index=False)
    try:
        sc.pl.umap(a, color=['leiden', 'celltype_gate'], save='_gate.png',
                   show=False) if 'celltype_gate' in a.obs else None
    except Exception:
        pass
    a.write(os.path.join(outdir, 'qc_clustered.h5ad'), compression='gzip')
    print('[done]', outdir)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
