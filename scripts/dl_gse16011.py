# -*- coding: utf-8 -*-
"""Parallel download of all GSE16011 CEL.gz files."""
import os, re, sys, time, urllib.request, ssl, threading

BASE = 'https://ftp.ncbi.nlm.nih.gov/geo/samples/'
OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'GSE16011', 'cel')
os.makedirs(OUT, exist_ok=True)
ctx = ssl.create_default_context()
lock = threading.Lock()
done = [0]

def get_gsm_list():
    url = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE16011&targ=gsm&form=text&view=brief'
    with urllib.request.urlopen(url, context=ctx, timeout=300) as r:
        txt = r.read().decode()
    return sorted(set(re.findall(r'!Sample_geo_accession = (GSM\d+)', txt)))

def fetch(gsm):
    n = gsm[3:6]  # GSM405 -> 405
    url = f'{BASE}{gsm[:3]}{n}nnn/{gsm}/suppl/{gsm}.CEL.gz'
    dest = os.path.join(OUT, gsm + '.CEL.gz')
    want = None
    for attempt in range(40):
        try:
            if want is None:
                req = urllib.request.Request(url, method='HEAD')
                with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
                    want = int(r.headers['Content-Length'])
            have = os.path.getsize(dest) if os.path.exists(dest) else 0
            if have >= want:
                with lock:
                    done[0] += 1
                    print(f'[{done[0]}/284] {gsm} ok', flush=True)
                return
            req = urllib.request.Request(url, headers={'Range': f'bytes={have}-', 'User-Agent': 'python'})
            with urllib.request.urlopen(req, context=ctx, timeout=180) as r, open(dest, 'ab') as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            if os.path.getsize(dest) >= want:
                with lock:
                    done[0] += 1
                    print(f'[{done[0]}/284] {gsm} ok', flush=True)
                return
        except Exception as e:
            print(f'[{gsm}] retry {attempt+1}: {e}', flush=True)
            time.sleep(min(10 * (attempt + 1), 60))
    print(f'[{gsm}] FAILED', flush=True)

gsm_list = get_gsm_list()
print(f'{len(gsm_list)} GSMs', flush=True)
TH = 8
sema = threading.Semaphore(TH)

def worker(g):
    with sema:
        fetch(g)

t0 = time.time()
threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gsm_list]
[t.start() for t in threads]
[t.join() for t in threads]
n_files = len([f for f in os.listdir(OUT) if f.endswith('.CEL.gz')])
print(f'[done] {n_files} files in {time.time()-t0:.0f}s', flush=True)
