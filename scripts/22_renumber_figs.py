# -*- coding: utf-8 -*-
"""Renumber figures: 10 main figures -> 7 (JITC Tables/Figures <= 7).

Mapping (old -> new):
  Fig1 a-d           -> Fig1 a-d
  Fig2 a,b,c         -> Fig1 e,f,g
  Fig3 a-e           -> Fig2 a-e
  Fig4 a-g           -> Fig3 a-g
  Fig5 a-c           -> Fig4 a-c
  Fig6 a-c           -> Fig5 a-c
  Fig8 a,b,c         -> Fig5 d,e,f
  Fig7 a-d           -> Fig6 a-d
  Fig9 a-e           -> Fig7 a-e
  Fig10 (3 panels)   -> Fig7 f,g,h
"""
import re, sys

SRC = 'submission/01_manuscript/_working.md'
DST = 'submission/01_manuscript/_working.md'

s = open(SRC, encoding='utf-8').read()
cut = s.index('# Figure Legends')
body, tail = s[:cut], s[cut:]

MAP = {}
for L in 'abc':          MAP[(2, L)] = (1, {'a': 'e', 'b': 'f', 'c': 'g'}[L])
for L in 'abcde':        MAP[(3, L)] = (2, L)
for L in 'abcdefg':      MAP[(4, L)] = (3, L)
for L in 'abc':          MAP[(5, L)] = (4, L)
for L in 'abc':          MAP[(6, L)] = (5, L)
for L in 'abc':          MAP[(8, L)] = (5, {'a': 'd', 'b': 'e', 'c': 'f'}[L])
for L in 'abcd':         MAP[(7, L)] = (6, L)
for L in 'abcde':        MAP[(9, L)] = (7, L)
for L in 'abcd':         MAP[(1, L)] = (1, L)

pat = re.compile(r'Fig\. (\d+)([a-z](?:\s*[,–-]\s*[a-z])*)')
report = []

def repl(m):
    fig = int(m.group(1))
    letters = re.findall(r'[a-z]', m.group(2))
    if fig == 10:
        return 'Fig. 7f–h'
    out = []
    for L in letters:
        if (fig, L) not in MAP:
            report.append(f'UNMAPPED old Fig.{fig}{L}')
            return m.group(0)
        out.append(MAP[(fig, L)])
    # group by new figure number, preserving order
    figs = []
    for nf, nl in out:
        if not figs or figs[-1][0] != nf:
            figs.append([nf, [nl]])
        else:
            figs[-1][1].append(nl)
    return '; '.join(f'Fig. {nf}{", ".join(ls)}' for nf, ls in figs)

newbody = pat.sub(repl, body)
open(DST, 'w', encoding='utf-8').write(newbody + tail)

changes = len(pat.findall(body))
print(f'Fig. references rewritten: {changes} matches')
if report:
    print('\n'.join(sorted(set(report))))
# audit
print('\n--- after ---')
for m in sorted(set(re.findall(r'Fig\. \d+[a-z]?(?:[,–][a-z])?', newbody))):
    print(' ', m, newbody.count(m))
