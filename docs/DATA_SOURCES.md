# Data sources

No data are redistributed in this repository. Download each dataset and place it at the path
shown; scripts use these relative paths.

| Dataset | Role | Accession / route | Expected location |
|---|---|---|---|
| GSE163108 | discovery cohort, 31 patients, CD3-sorted scRNA-seq + paired TCR-seq | GEO `GSE163108` | `data/GSE163108/` (10x matrix + `filtered_contig_annotations.csv`) |
| GSE182109 | validation cohort, 13 donors, pan-glioma | GEO `GSE182109` | `data/GSE182109/` (filtered matrix; no BAM/loom deposited) |
| GSE16011 | 284-sample Affymetrix HG-U133 Plus 2.0 cohort (WHO grade I–IV + controls) | GEO `GSE16011` / ArrayExpress mirror `E-GEOD-16011` (`E-GEOD-16011.sdrf.txt` for histology) | `data/GSE16011/` (`cel/`, `series_matrix.txt.gz`, `GSE16011_rma_gene.tsv` after step 13) |
| TCGA-GBM | bulk RNA-seq, n = 156 with survival | UCSC Xena / GDC (`GBM_HiSeqV2.gz`, `GBM_clinical.tsv`) | `data/TCGA/` |
| CGGA-693 | bulk RNA-seq, Chinese glioma cohort | cgga.org.cn (`CGGA.mRNAseq_693.RSEM-genes.20200506.txt` + clinical) | `data/CGGA/` |
| Couturier SL040 / SL057 | external single-cell GBM cohorts | CELLxGENE collection (Couturier 2020, Nat Commun 11:3406) | fetched by `download_cellxgene.sh` |
| 66-sample glioma atlas | external single-cell + spatial atlas (639,409 cells, 56 donors) | CELLxGENE (Zaman et al., MedComm 2026;7:e71022) | fetched by `download_atlas.sh` |

## Notes

- **GSE182109** deposits only filtered spliced matrices. Unspliced counts are therefore unavailable
  and spliced/unspliced RNA velocity cannot be estimated; differentiation direction is instead
  inferred with CytoTRACE-style potential plus transcriptional entropy (`15_`, `17_`).
- **NCBI rate limiting.** Direct GEO downloads were throttled or reset in our environment during
  this work; the ArrayExpress/BioStudies mirror was used for the GSE16011 sample annotation
  (`E-GEOD-16011.sdrf.txt`, histology field) and CELLxGENE for the two external single-cell
  cohorts. The scripts record which route each file came from.
- **GSE16011 CEL files.** `13_gse16011_rma.R` drops CEL files smaller than 100 kB (failed
  downloads) and aborts if fewer than 275 remain, so a partial download cannot silently propagate.
- All datasets are de-identified and were generated under the ethics approvals of the original
  studies; no new human or animal material was collected for this work.
