# -*- coding: utf-8 -*-
"""Parallel range downloader for GSE131928 (Neftel 2019) processed TPM matrices."""
import os, sys, threading, urllib.request, ssl, time

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131928/suppl/"
FILES = [
    ("GSE131928_RAW.tar", 669112320),
]
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'GSE131928')
os.makedirs(OUTDIR, exist_ok=True)
N = 4
ctx = ssl.create_default_context()


def fetch(url, out, total, idx, start, end, max_retry=80):
    part = out + f'.part{idx}'
    want = end - start + 1
    for attempt in range(max_retry):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= want:
            return
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={start+have}-{end}',
                                                      'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=180) as r:
                status = getattr(r, 'status', 200)
                if status == 200:
                    mode, pos = 'wb', 0
                else:
                    mode, pos = 'ab', have
                with open(part, mode) as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
            if os.path.getsize(part) >= want:
                print(f'  [p{idx}] finished', flush=True)
                return
            print(f'  [p{idx}] short {os.path.getsize(part)}/{want}, retry', flush=True)
        except Exception as e:
            print(f'  [p{idx}] retry {attempt+1}: {e}', flush=True)
            time.sleep(min(30 * (attempt + 1), 300))
    raise RuntimeError(f'part {idx} failed')


def download(name, total):
    url = BASE + name
    out = os.path.join(OUTDIR, name)
    if os.path.exists(out) and os.path.getsize(out) == total:
        print(f'[skip] {name} already complete', flush=True)
        return out
    print(f'[start] {name} ({total/1e6:.0f} MB)', flush=True)
    chunk = total // N + 1
    ths = []
    for i in range(N):
        s, e = i * chunk, min((i + 1) * chunk - 1, total - 1)
        t = threading.Thread(target=fetch, args=(url, out, total, i, s, e), daemon=True)
        t.start()
        ths.append(t)
    for t in ths:
        t.join()
    for i in range(N):
        part = out + f'.part{i}'
        want = min((i + 1) * chunk, total) - i * chunk
        assert os.path.exists(part) and os.path.getsize(part) == want, f'{name} part{i} incomplete'
    with open(out, 'wb') as o:
        for i in range(N):
            with open(out + f'.part{i}', 'rb') as p:
                while True:
                    b = p.read(1 << 22)
                    if not b:
                        break
                    o.write(b)
    for i in range(N):
        os.remove(out + f'.part{i}')
    print(f'[done] {name} {os.path.getsize(out)} bytes', flush=True)
    return out


if __name__ == '__main__':
    t0 = time.time()
    for name, total in FILES:
        download(name, total)
    print(f'[ALL DONE] {time.time()-t0:.0f}s', flush=True)
