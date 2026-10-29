#!/bin/bash
# Robust retry downloader for flaky connection to NCBI FTP.
# Loops curl with resume (-C -); aborts a round when throughput <1KB/s for 30s.
set -u
cd "$(dirname "$0")/../data" || exit 1
mkdir -p GSE84465 GSE131928

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
    curl -sL -A "Mozilla/5.0" -C - -o "$out" \
         --max-time 1800 --speed-time 30 --speed-limit 1000 --retry 0 "$url"
    sleep 15
  done
  echo "[$tag][FAIL]"
  return 1
}

download "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84465/suppl/GSE84465_GBM_All_data.csv.gz" \
         "GSE84465/GSE84465_GBM_All_data.csv.gz" 20971520 S84465

download "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131928/suppl/GSE131928_RAW.tar" \
         "GSE131928/GSE131928_RAW.tar" 669112320 S131928

echo "ALL DONE"
