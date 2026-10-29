import scanpy as sc, pandas as pd
a = sc.read_10x_h5('data/raw/GSM4972162_E10_raw_feature_bc_matrix.h5')
m = pd.read_csv('data/raw/metadata_10x.csv.gz')
bc = m[m['sampleid'] == 'E10_GEX'].iloc[:, 0].str.replace('E10_GEX_', '', regex=False).values
print('meta bc sample:', bc[:5], 'n=', len(bc))
obs = set(a.obs_names)
for suffix in ['', '-1']:
    hit = sum((b + suffix) in obs for b in bc)
    print('suffix', repr(suffix), 'matched', hit, '/', len(bc))
# also try hash comparison
print('h5 first:', list(a.obs_names[:3]))
