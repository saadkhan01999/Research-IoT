"""Independent check: re-read source zip and merged zip, compare row counts per class.

Usage: python -I verify.py <source.zip> <merged.zip>
"""
import sys, zipfile, collections, time

SRC, OUT = sys.argv[1], sys.argv[2]
CH = 32 * 1024 * 1024
t0 = time.time()

# 1) source: data rows per folder (newline count minus header, + 1 if no final newline)
src_rows = collections.Counter()
zs = zipfile.ZipFile(SRC)
for i in zs.infolist():
    if not i.filename.lower().endswith('.csv'):
        continue
    folder = i.filename.split('/')[1]
    n, last = 0, b'\n'
    with zs.open(i) as f:
        while True:
            c = f.read(CH)
            if not c:
                break
            n += c.count(b'\n')
            last = c[-1:]
    if last != b'\n':
        n += 1
    src_rows['BenignTraffic' if folder == 'Benign_Final' else folder] += n - 1
print('source scanned in %.0fs, rows=%d' % (time.time() - t0, sum(src_rows.values())), flush=True)

# 2) merged: header, field count of every line, label counts
zo = zipfile.ZipFile(OUT)
print('zip test:', zo.testzip() or 'OK (all CRCs valid)', flush=True)
out_rows = collections.Counter()
bad_fields = 0
header = None
carry = b''
with zo.open('CICIoT2023_merged.csv') as f:
    header = f.readline().rstrip(b'\n')
    ncomma = header.count(b',')
    while True:
        c = f.read(CH)
        if not c:
            break
        data = carry + c
        cut = data.rfind(b'\n')
        block, carry = data[:cut + 1], data[cut + 1:]
        for ln in block.split(b'\n')[:-1]:
            if ln.count(b',') != ncomma:
                bad_fields += 1
            out_rows[ln[ln.rfind(b',', 0, ln.rfind(b',')) + 1:ln.rfind(b',')].decode()] += 1
assert not carry, 'merged file does not end with newline'
print('merged scanned in %.0fs' % (time.time() - t0), flush=True)
print('header columns:', ncomma + 1, '| last two:', header.split(b',')[-2:])
print('lines with wrong field count:', bad_fields)
print('%-26s %12s %12s %s' % ('Label', 'source', 'merged', 'match'))
ok = True
for k in sorted(set(src_rows) | set(out_rows)):
    m = src_rows[k] == out_rows[k]
    ok &= m
    print('%-26s %12d %12d %s' % (k, src_rows[k], out_rows[k], 'OK' if m else 'DIFF'))
print('TOTAL source=%d merged=%d -> %s' % (sum(src_rows.values()), sum(out_rows.values()),
                                          'ALL MATCH' if ok and bad_fields == 0 else 'MISMATCH'))
