#!/bin/bash
# Download CELLxGENE-hosted GBM scRNA h5ad (Couturier 2020 SL040 + SL057) via S3/CloudFront.
set -u
cd "$(dirname "$0")/../data" || exit 1
mkdir -p CELLxGENE

download () {
  local url="$1" out="$2" want="$3" tag="$4"
  for i in $(seq 1 300); do
    local sz=0
    [ -f "$out" ] && sz=$(stat -c %s "$out")
    if [ "$sz" -ge "$want" ]; then
      echo "[$tag][ok] $out ($sz bytes) after $i round(s)"
      return 0
    fi
    echo "[$tag] round $i have $sz/$want"
    curl -sL -C - -o "$out" --max-time 3600 --speed-time 30 --speed-limit 10000 "$url"
    sleep 10
  done
  echo "[$tag][FAIL]"
  return 1
}

download "https://datasets.cellxgene.cziscience.com/d45b4ce6-9725-4d79-b97a-70a44158bdbf.h5ad" \
         "CELLxGENE/SL040_gbm.h5ad" 1512506446 SL040

download "https://datasets.cellxgene.cziscience.com/66e7fd4d-d2db-464d-9fad-6eda97aff2e1.h5ad" \
         "CELLxGENE/SL057_gbm.h5ad" 1197832973 SL057

echo "ALL DONE"
