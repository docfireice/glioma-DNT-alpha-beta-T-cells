#!/bin/bash
# Download Brain tumour single-cell atlas (glioma cohort, 66 samples) via CELLxGENE S3.
set -u
cd "$(dirname "$0")/../data/CELLxGENE" || exit 1

URL="https://datasets.cellxgene.cziscience.com/b840a9b5-0dbf-4760-969f-8bb73c19b87b.h5ad"
OUT="glioma_atlas_66s.h5ad"
WANT=7253526062

for i in $(seq 1 600); do
  sz=0
  [ -f "$OUT" ] && sz=$(stat -c %s "$OUT")
  if [ "$sz" -ge "$WANT" ]; then
    echo "[atlas][ok] $sz bytes after $i round(s)"
    exit 0
  fi
  echo "[atlas] round $i have $sz/$WANT"
  curl -sL -C - -o "$OUT" --max-time 7200 --speed-time 30 --speed-limit 10000 "$URL"
  sleep 10
done
echo "[atlas][FAIL]"
