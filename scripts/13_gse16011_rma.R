# GSE16011 Murat 2008: RMA normalization of 284 CEL.gz -> gene-level matrix
suppressMessages({
  library(oligo)
})
cat("R", R.version.string, "| oligo", as.character(packageVersion("oligo")), "\n")
cel_dir <- "data/GSE16011/cel"
out_tsv <- "data/GSE16011/GSE16011_rma_gene.tsv"
all_files <- list.files(cel_dir, pattern = "\\.CEL\\.gz$", full.names = TRUE)
files <- all_files[file.info(all_files)$size > 100000]   # drop empty/failed downloads
cat("CEL files:", length(files), "(of", length(all_files), "found)\n")
if (length(files) < 275) stop("download incomplete")

ab <- read.celfiles(files)
cat("read.celfiles done:", nrow(ab), "probesets\n")

eset <- rma(ab)
cat("RMA done\n")

expr <- exprs(eset)
cat("expr dim:", dim(expr), "\n")

# probe -> gene symbol mapping (hgu133plus2.db)
suppressMessages(library(hgu133plus2.db))
probe_ids <- rownames(expr)
sym <- AnnotationDbi::select(hgu133plus2.db, keys = probe_ids,
                             columns = "SYMBOL", keytype = "PROBEID")
cat("mapped:", sum(!is.na(sym$SYMBOL)), "/", length(probe_ids), "\n")

sym <- sym[!is.na(sym$SYMBOL) & !duplicated(sym$PROBEID), ]
expr <- expr[sym$PROBEID, , drop = FALSE]
rownames(expr) <- sym$SYMBOL
# collapse multiple probes per gene -> max mean expression
if (any(duplicated(rownames(expr)))) {
  gmean <- rowMeans(expr)
  keep <- tapply(seq_along(rownames(expr)), rownames(expr),
                 function(i) i[which.max(gmean[i])])
  keep <- unlist(keep, use.names = FALSE)
  expr <- expr[keep, , drop = FALSE]
  rownames(expr) <- sym$SYMBOL[keep]
}
cat("gene-level dim:", dim(expr), "\n")

out <- cbind(PROBEID = rownames(expr), expr)
write.table(out, out_tsv, sep = "\t", quote = FALSE)
cat("saved:", out_tsv, "\n")
