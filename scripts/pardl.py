# -*- coding: utf-8 -*-
"""Parallel range downloader for GSE182109_RAW.tar"""
import os, sys, threading, urllib.request, ssl, time
URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE182nnn/GSE182109/suppl/GSE182109_RAW.tar"
OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'GSE182109', 'GSE182109_RAW.tar')
TOTAL = 2460129280
N = 16
CHUNK = TOTAL // N + 1
ctx = ssl.create_default_context()

def fetch(i, start, end, max_retry=60):
    part = OUT + f'.part{i}'
    want = end - start + 1
    for attempt in range(max_retry):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= want:
            print(f'[p{i}] done', flush=True); return
        try:
            req = urllib.request.Request(URL, headers={'Range': f'bytes={start+have}-{end}', 'User-Agent': 'python'})
            with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
                status = getattr(r, 'status', 200)
                if status == 200:  # server ignored Range -> rewrite whole part
                    mode, pos = 'wb', 0
                else:
                    mode, pos = 'ab', have
                got = 0
                with open(part, mode) as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b: break
                        f.write(b); got += len(b)
            if os.path.getsize(part) >= want:
                print(f'[p{i}] finished', flush=True); return
            print(f'[p{i}] short read ({os.path.getsize(part)}/{want}), retry', flush=True)
        except Exception as e:
            print(f'[p{i}] retry {attempt+1}: {e}', flush=True)
            time.sleep(60 * min(attempt + 1, 10))
    raise RuntimeError(f'part {i} failed after {max_retry} retries')

threads = []
t0 = time.time()
for i in range(N):
    s, e = i * CHUNK, min((i + 1) * CHUNK - 1, TOTAL - 1)
    t = threading.Thread(target=fetch, args=(i, s, e), daemon=True)
    t.start(); threads.append(t)
for t in threads: t.join()
# verify all parts complete before merge
for i in range(N):
    part = OUT + f'.part{i}'
    want = (min((i + 1) * CHUNK, TOTAL) - i * CHUNK)
    assert os.path.exists(part) and os.path.getsize(part) == want, f'part{i} incomplete'
# merge
with open(OUT, 'wb') as out:
    for i in range(N):
        with open(OUT + f'.part{i}', 'rb') as p:
            while True:
                b = p.read(1 << 22)
                if not b: break
                out.write(b)
print(f'[merged] {os.path.getsize(OUT)} bytes in {time.time()-t0:.0f}s', flush=True)
for i in range(N): os.remove(OUT + f'.part{i}')
print('[ok]', flush=True)
