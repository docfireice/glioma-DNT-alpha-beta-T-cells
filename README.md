# Analysis code — double-negative αβ T cells in glioblastoma

Code accompanying the manuscript:

> **Clonally distinct, terminally differentiated CD4⁻CD8⁻ double-negative αβ T cells are
> recruited through macrophage SPP1–CD44 and MIF–CD74 axes in glioblastoma**
> Yanqing He, Jian Liu (corresponding). Department of Neurosurgery, Xiangya Hospital,
> Central South University; National Clinical Research Center for Geriatric Disorders.

This repository contains every script needed to reproduce the gating, T-cell receptor (TCR)
repertoire, trajectory/differentiation, regulon, ligand–receptor, spatial, bulk-quantification
and clinical-association analyses reported in the paper, together with the figure-assembly
scripts for Figures 1–7 and Supplementary Figures S1–S6.

It does **not** contain data. All datasets are public; see [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)
for accessions, download routes and where each file is expected on disk.

---

## 1. Environment

Analyses were run on Python 3.12 with the following pinned versions (see `requirements.txt`):

| package | version | used for |
|---|---|---|
| scanpy | 1.12.4 | single-cell loading, QC, clustering, UMAP, scoring |
| anndata | 0.13.4 | h5ad handling |
| numpy | 2.5.4 | array operations |
| scipy | 1.18.1 | sparse matrices, statistical tests |
| pandas | 2.3.3 | tabular output |
| scikit-learn | 1.9.1 | AUROC, scaling, metrics |
| lifelines | 0.30.3 | Cox models, Kaplan–Meier (robustness analyses only) |
| statsmodels | 0.15.0 | OLS with covariates, ordinal models |
| matplotlib / seaborn | 3.11.2 / 0.13.2 | figures |
| pillow | — | figure composition |

One step (`13_gse16011_rma.R`) requires R with `oligo` and the `pd.hg.u133.plus.2` annotation
package for RMA normalisation of the GSE16011 CEL files.

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Layout and how to run

Scripts use **relative paths** and must be executed from the repository root:

```
<repo>/
├── scripts/            # analysis scripts, numbered by module
│   └── deprecated/     # probes and superseded routes, retained for transparency
├── docs/               # data sources, figure→script map
├── data/               # you create this and download the public datasets into it
└── results/            # created by the scripts
```

```bash
python scripts/03_gse163108_full.py          # example
```

## 3. Cell-state definitions (the parameters a reviewer will want to check)

Gate definitions are stated verbatim at the top of `scripts/08c_refine_gate.py` and are applied
identically to every cohort (`scripts/08_gse182109.py`, `scripts/16b_cellxgene_validation.py`,
`scripts/16d_atlas_validation.py`):

```
gd_T   : CD3+  & (TRDC | TRDV1 | TRDV2 | TRGC1)+            # gamma-delta lineage definition
ab_DNT : CD3+  & gd-markers == 0 & CD4 == 0 & CD8A == 0 & CD8B == 0
         TRAC+ subset = confirmed alpha-beta lineage
CD3+   : CD3D + CD3E > 0        (detection threshold: raw count > 0; no imputation)
```

Notes on the definition:

- `CD8B` is required to be negative. Dropping it inflates the apparent DN fraction; the
  stringency of this requirement is the reason the GSE163108 (3.3%) and GSE182109 (41.0%)
  fractions differ so widely — deeper 10× coverage resolves CD8B, shallower data does not.
- No imputation, no smoothing and no signature score enters the gate itself. Only detection of
  raw transcripts, so the compartment is defined by lineage, not by a learned classifier.
- Tissue-level (bulk) quantification cannot use this gate. It uses the constant-region transcript
  `TRDC` instead (see §5).

## 4. Analysis order

Run in this order; each step writes into `results/<dataset>/`.

| Step | Script | Purpose |
|---|---|---|
| 1 | `02_qc_pipeline.py`, `02b_pipeline_numpy.py` | QC, normalisation, clustering, UMAP |
| 2 | `03_gse163108_full.py` | full GSE163108 processing, marker detection, gate assignment |
| 3 | `04_figures.py` | UMAP/dot-plot/heat-map/composition panels |
| 4 | `05_tcr_deep.py` | TCR clonotype sharing, privacy, clone-size analysis |
| 5 | `06_trajectory.py` | diffusion pseudotime, signature profiles along pseudotime |
| 6 | `07_deconv_survival.py` | NNLS deconvolution and Cox models (**superseded — see §6**) |
| 7 | `08_gse182109.py`, `08b_gate_diag.py`, `08c_refine_gate.py` | external cohort, gate diagnostics, stringent v2 gate |
| 8 | `09_cgga_validation.py`, `14_gse16011_infiltration.py`, `13_gse16011_rma.R` | bulk cohorts |
| 9 | `10_lr_communication.py`, `10_lr_gse182109.py` | ligand–receptor inference (CellChat-style) |
| 10 | `11_regulons.py` | SCENIC-style regulon activity |
| 11 | `12_visium.py` | Visium deconvolution and spatial co-localisation |
| 12 | `15_differentiation_potential.py`, `17_gse182109_diffpot.py` | CytoTRACE-style potential and transcriptional entropy |
| 13 | `16b_cellxgene_validation.py`, `16c_signature_transfer.py`, `16d_atlas_validation.py` | CELLxGENE cohorts, signature transfer, 66-sample atlas |
| 14 | `18_robustness.py`, `18b_robustness_addendum.py` | multivariable Cox, PH diagnostics, dropout simulation, TCR sensitivity |
| 15 | `19_survival_markerscore.py`, `20_clinical_endpoints.py` | marker-score quantification and the cross-sectional clinical analysis |
| 16 | `21_fig9_clinical.py`, `22_renumber_figs.py`, `23_build_main_figures.py` | clinical figure, figure renumbering, final plate assembly |

`dl_gse16011.py`, `dl_gse131928.py`, `download_cellxgene.sh`, `download_atlas.sh`, `retry_download.sh`
and `pardl.py` handle data retrieval only.

## 5. Why bulk quantification uses TRDC rather than a signature

A 287-gene signature derived from the single-cell reference and a 34-cell γδ reference both fail
as bulk estimators for this compartment: the two strategies applied to the same cohorts gave
**opposite** survival directions (HR 0.48 vs 2.24 in CGGA), with sample-level correlation between
them of only r = 0.25–0.35. NNLS deconvolution additionally returned all-zero solutions for
44% (CGGA) and 49% (TCGA) of samples, because z-scored observations contain negative values that
violate the non-negativity constraint. For these reasons all prognostic claims were withdrawn from
the manuscript, and clinical inference is restricted to a cross-sectional association with WHO
grade anchored on `TRDC` — the constant-region transcript that defines the γδ lineage and is the
same criterion used in the single-cell gate. `scripts/20_clinical_endpoints.py` implements this
and prints the failure diagnostics of the alternative strategies.

## 6. Reproducibility caveats

- `07_deconv_survival.py` and `18c_final_survival.py` are retained because their negative result
  is reported in the manuscript (Supplementary Note S1). They are **not** part of the current
  Results.
- Random seeds are fixed inside each script where stochastic steps occur (UMAP, Leiden, bootstraps).
- Figures 1–7 are assembled from per-panel files produced by earlier steps; `23_build_main_figures.py`
  regenerates the eight panels that exist only as tables and then composes the plates at 300 dpi.
- `22_renumber_figs.py` maps the analysis-stage figure numbering onto the 7-figure submission
  numbering; run it only if you are regenerating the manuscript.

## 7. Citation, licence and contact

If you use this code, please cite the manuscript above. Released under the MIT licence
(see `LICENSE`).

Correspondence: Jian Liu, M.D., Ph.D. — doclj2015@csu.edu.cn
