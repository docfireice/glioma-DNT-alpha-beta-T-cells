# -*- coding: utf-8 -*-
"""Finish part11 with parallel sub-chunks, then merge all parts into the tar."""
import os, threading, urllib.request, ssl, time
URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE182nnn/GSE182109/suppl/GSE182109_RAW.tar"
D = os.path.join(os.path.dirname(__file__), '..', 'data', 'GSE182109')
OUT = os.path.join(D, 'GSE182109_RAW.tar')
TOTAL = 2460129280
N = 16
CHUNK = TOTAL // N + 1
I = 11
start, end = I * CHUNK, min((I + 1) * CHUNK - 1, TOTAL - 1)
want = end - start + 1
part = OUT + f'.part{I}'
have = os.path.getsize(part)
remaining = want - have
print(f'[p11] have {have}/{want}, remaining {remaining}', flush=True)
M = 8
sub = remaining // M + 1
ctx = ssl.create_default_context()

def fetch_sub(j, s, e):
    sp = part + f'.sub{j}'
    for attempt in range(200):
        h = os.path.getsize(sp) if os.path.exists(sp) else 0
        w = e - s + 1
        if h >= w:
            print(f'[s{j}] done', flush=True); return
        try:
            req = urllib.request.Request(URL, headers={'Range': f'bytes={start+have+s+h}-{e+have}', 'User-Agent': 'python'})
            with urllib.request.urlopen(req, context=ctx, timeout=60) as r, open(sp, 'ab') as f:
                while True:
                    b = r.read(1 << 19)
                    if not b: break
                    f.write(b)
        except Exception as ex:
            print(f'[s{j}] retry {attempt}: {ex}', flush=True)
            time.sleep(20)
    raise RuntimeError(f'sub {j} failed')

ts = []
for j in range(M):
    s, e = j * sub, min((j + 1) * sub - 1, remaining - 1)
    t = threading.Thread(target=fetch_sub, args=(j, s, e), daemon=True)
    t.start(); ts.append(t)
for t in ts: t.join()
with open(part, 'ab') as f:
    for j in range(M):
        sp = part + f'.sub{j}'
        with open(sp, 'rb') as p:
            while True:
                b = p.read(1 << 22)
                if not b: break
                f.write(b)
assert os.path.getsize(part) == want, f'p11 still incomplete {os.path.getsize(part)}'
for j in range(M): os.remove(part + f'.sub{j}')
print(f'[p11] complete: {os.path.getsize(part)}', flush=True)

# merge all
with open(OUT, 'wb') as out:
    for i in range(N):
        with open(OUT + f'.part{i}', 'rb') as p:
            while True:
                b = p.read(1 << 22)
                if not b: break
                out.write(b)
print(f'[merged] {os.path.getsize(OUT)}', flush=True)
for i in range(N): os.remove(OUT + f'.part{i}')
print('[ok]', flush=True)
