"""Stream-merge every CSV in the CICIoT2023 CSV.zip into one labelled CSV inside a new zip.

Usage: python -I merge.py <input.zip> <output.zip> <work_dir>

Nothing is extracted to disk: each member is decompressed in chunks, a Label and
Category column (taken from the folder name) is appended to every row, and the
result is compressed straight into the output zip. Along the way it validates
rows, counts classes, takes a 1% sample for feature statistics and writes row
hashes to bucket files for exact-duplicate analysis.
"""
import sys, os, re, zipfile, time, math, collections
from array import array

ZIN, ZOUT, WORK = sys.argv[1], sys.argv[2], sys.argv[3]
OUT_DIR = os.path.dirname(os.path.abspath(ZOUT))
MERGED_NAME = 'CICIoT2023_merged.csv'
CHUNK = 16 * 1024 * 1024
NB = 256                      # hash buckets for duplicate analysis
M64 = (1 << 64) - 1
SAMPLE_EVERY = 100            # 1% systematic sample for feature stats
ALLOWED = b'0123456789.,-+eE\n'

WEB = {'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking'}


def label_for(folder):
    if folder == 'Benign_Final':
        return 'BenignTraffic', 'Benign'
    if folder.startswith('DDoS-'):
        return folder, 'DDoS'
    if folder.startswith('DoS-'):
        return folder, 'DoS'
    if folder.startswith('Mirai-'):
        return folder, 'Mirai'
    if folder.startswith('Recon-') or folder == 'VulnerabilityScan':
        return folder, 'Recon'
    if folder in WEB:
        return folder, 'Web'
    if folder == 'DictionaryBruteForce':
        return folder, 'BruteForce'
    if folder in ('MITM-ArpSpoofing', 'DNS_Spoofing'):
        return folder, 'Spoofing'
    raise ValueError('Unknown class folder: ' + folder)


def natural_key(info):
    folder, fname = info.filename.split('/')[1:3]
    m = re.match(r'^(.*?)(\d*)\.pcap\.csv$', fname)
    stem, num = (m.group(1), int(m.group(2) or 0)) if m else (fname, 0)
    return (folder.lower(), stem.lower(), num)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


if os.path.exists(ZOUT):
    sys.exit('Output already exists, refusing to overwrite: ' + ZOUT)
os.makedirs(WORK, exist_ok=True)
for b in range(NB):
    p = os.path.join(WORK, 'b%03d.bin' % b)
    if os.path.exists(p):
        os.remove(p)

zin = zipfile.ZipFile(ZIN)
members = sorted([i for i in zin.infolist() if i.filename.lower().endswith('.csv')], key=natural_key)
readme = [i for i in zin.infolist() if i.filename.lower().endswith('.pdf')]

with zin.open(members[0]) as f:
    HEADER = f.readline().rstrip(b'\r\n')
COLS = HEADER.decode('ascii').split(',')
NCOL = len(COLS)
NCOMMA = NCOL - 1

labels = sorted({label_for(i.filename.split('/')[1])[0] for i in members})
CID = {l: n for n, l in enumerate(labels)}
assert len(labels) < 256

# ---- statistics containers -------------------------------------------------
rows_by_label = collections.Counter()
files_by_label = collections.Counter()
cat_of = {}
file_rows = []
q = collections.Counter()                     # data-quality counters
malformed_examples = []
empty_by_col = collections.Counter()
nonnum_by_col = collections.defaultdict(collections.Counter)
s_n = [0] * NCOL
s_sum = [0.0] * NCOL
s_min = [math.inf] * NCOL
s_max = [-math.inf] * NCOL
s_nonfinite = [0] * NCOL
sample_rows = 0

buf = [array('Q') for _ in range(NB)]
bucket_fh = [open(os.path.join(WORK, 'b%03d.bin' % b), 'ab') for b in range(NB)]
buffered = 0


def flush_buckets():
    global buffered
    for b in range(NB):
        if buf[b]:
            buf[b].tofile(bucket_fh[b])
            buf[b] = array('Q')
    buffered = 0


def add_hashes(lines, ci):
    global buffered
    for ln in lines:
        h = hash(ln) & M64
        buf[h & 0xFF].append((h & 0xFFFFFFFFFFFFFF00) | ci)
    buffered += len(lines)
    if buffered > 3_000_000:
        flush_buckets()


def add_sample(lines):
    global sample_rows
    for ln in lines[::SAMPLE_EVERY]:
        vals = ln.decode('ascii', 'replace').split(',')
        sample_rows += 1
        for j, v in enumerate(vals):
            try:
                x = float(v)
            except ValueError:
                continue
            if not math.isfinite(x):
                s_nonfinite[j] += 1
                continue
            s_n[j] += 1
            s_sum[j] += x
            if x < s_min[j]:
                s_min[j] = x
            if x > s_max[j]:
                s_max[j] = x


def slow_filter(lines, fname):
    """Per-line validation for blocks that failed the fast checks."""
    keep = []
    for ln in lines:
        if (ln.count(b',') == NCOMMA and not ln.translate(None, ALLOWED) and b',,' not in ln
                and not ln.startswith(b',') and not ln.endswith(b',')):
            keep.append(ln)            # clean line inside a block that had a problem elsewhere
            continue
        if not ln.strip():
            q['empty_lines_dropped'] += 1
            continue
        if ln == HEADER:
            q['repeated_header_rows_dropped'] += 1
            continue
        if ln.count(b',') != NCOMMA:
            q['malformed_rows_dropped'] += 1
            if len(malformed_examples) < 10:
                malformed_examples.append((fname, ln[:160]))
            continue
        fields = ln.split(b',')
        flagged = False
        for j, v in enumerate(fields):
            if v == b'':
                empty_by_col[COLS[j]] += 1
                flagged = True
            elif v.translate(None, ALLOWED):
                nonnum_by_col[COLS[j]][v[:24].decode('ascii', 'replace')] += 1
                flagged = True
        if flagged:
            q['rows_with_empty_or_nonnumeric_values_kept'] += 1
        keep.append(ln)
    return keep


t0 = time.time()
total_rows = 0
bytes_in = 0
bytes_out = 0
zout = zipfile.ZipFile(ZOUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True)
zi = zipfile.ZipInfo(MERGED_NAME, date_time=time.localtime()[:6])
zi.compress_type = zipfile.ZIP_DEFLATED
zi._compresslevel = 6
zi.external_attr = 0o644 << 16
w = zout.open(zi, 'w', force_zip64=True)
out_header = HEADER + b',Label,Category\n'
w.write(out_header)
bytes_out += len(out_header)

for idx, info in enumerate(members, 1):
    folder = info.filename.split('/')[1]
    label, cat = label_for(folder)
    cat_of[label] = cat
    ci = CID[label]
    suffix = (',%s,%s\n' % (label, cat)).encode('ascii')
    frows = 0
    with zin.open(info) as f:
        hdr = f.readline().rstrip(b'\r\n')
        if hdr != HEADER:
            sys.exit('Header mismatch in %s' % info.filename)
        carry = b''
        eof = False
        while not eof:
            chunk = f.read(CHUNK)
            if chunk:
                data = carry + chunk
                cut = data.rfind(b'\n')
                if cut < 0:
                    carry = data
                    continue
                block, carry = data[:cut + 1], data[cut + 1:]
            else:
                eof = True
                if not carry:
                    break
                block, carry = carry + b'\n', b''
                q['files_missing_final_newline'] += 1
            bytes_in += len(block)
            if b'\r' in block:
                block = block.replace(b'\r\n', b'\n')
            lines = block.split(b'\n')
            lines.pop()                      # trailing empty piece after last \n
            fast = (not block.translate(None, ALLOWED)
                    and b',,' not in block and b'\n,' not in block and b',\n' not in block
                    and not block.startswith(b',') and b'\n\n' not in block
                    and not block.startswith(b'\n')
                    and all(ln.count(b',') == NCOMMA for ln in lines))
            if fast:
                out = block.replace(b'\n', suffix)
            else:
                lines = slow_filter(lines, info.filename)
                out = suffix.join(lines) + suffix if lines else b''
            if out:
                w.write(out)
                bytes_out += len(out)
            add_hashes(lines, ci)
            add_sample(lines)
            frows += len(lines)
    total_rows += frows
    rows_by_label[label] += frows
    files_by_label[label] += 1
    file_rows.append((info.filename, label, frows))
    el = time.time() - t0
    log('[%3d/%d] %-60s %9d rows | total %10d | %6.0fs | %.0f MB/s'
        % (idx, len(members), info.filename, frows, total_rows, el, bytes_in / 1e6 / max(el, 1e-9)))

w.close()
flush_buckets()
for fh in bucket_fh:
    fh.close()
log('Merged CSV written: %d rows, %.2f GB uncompressed' % (total_rows, bytes_out / 1e9))

# ---- exact duplicate analysis ---------------------------------------------
log('Running duplicate analysis ...')
unique_feature_vectors = 0
unique_feature_label_pairs = 0
conflicting_vectors = 0
dup_within = collections.Counter()
conflict_rows_by_label = collections.Counter()
for b in range(NB):
    p = os.path.join(WORK, 'b%03d.bin' % b)
    if not os.path.exists(p):
        continue
    a = array('Q')
    with open(p, 'rb') as fh:
        a.frombytes(fh.read())
    seen = set()
    first_cls = {}
    conflict = set()
    for r in a:
        if r in seen:
            dup_within[r & 0xFF] += 1
            continue
        seen.add(r)
        k = r >> 8
        c = r & 0xFF
        fc = first_cls.get(k)
        if fc is None:
            first_cls[k] = c
        elif fc != c:
            conflict.add(k)
    if conflict:
        for r in a:
            if (r >> 8) in conflict:
                conflict_rows_by_label[r & 0xFF] += 1
    unique_feature_vectors += len(first_cls)
    unique_feature_label_pairs += len(seen)
    conflicting_vectors += len(conflict)
    del a, seen, first_cls, conflict
    os.remove(p)
log('Duplicate analysis done')

# ---- reports ---------------------------------------------------------------
inv = {v: k for k, v in CID.items()}
dup_within_by_label = {inv[c]: n for c, n in dup_within.items()}
conflict_by_label = {inv[c]: n for c, n in conflict_rows_by_label.items()}

cd_lines = ['Label,Category,Files,Rows,Percent,WithinClassDuplicateRows,RowsSharedWithOtherClasses']
for label, n in rows_by_label.most_common():
    cd_lines.append('%s,%s,%d,%d,%.4f,%d,%d' % (label, cat_of[label], files_by_label[label], n,
                                                100.0 * n / total_rows, dup_within_by_label.get(label, 0),
                                                conflict_by_label.get(label, 0)))
class_csv = '\n'.join(cd_lines) + '\n'

cat_rows = collections.Counter()
for label, n in rows_by_label.items():
    cat_rows[cat_of[label]] += n

R = []
R.append('CICIoT2023 - merged CSV report')
R.append('=' * 72)
R.append('Generated : %s' % time.strftime('%Y-%m-%d %H:%M'))
R.append('Source    : %s (%d CSV files in %d class folders)' % (os.path.basename(ZIN), len(members), len(labels)))
R.append('Output    : %s -> %s' % (os.path.basename(ZOUT), MERGED_NAME))
R.append('Rows      : %d data rows (+1 header row)' % total_rows)
R.append('Columns   : %d = %d original features + Label + Category' % (NCOL + 2, NCOL))
R.append('Size      : %.2f GB uncompressed' % (bytes_out / 1e9))
R.append('Time      : %.1f min' % ((time.time() - t0) / 60))
R.append('')
R.append('How the merge was done')
R.append('-' * 72)
R.append('* The original per-pcap CSVs have NO label column; the class is only the folder name.')
R.append('  Each row therefore gets:')
R.append('    Label    = folder name (34 classes; folder "Benign_Final" -> "BenignTraffic")')
R.append('    Category = attack family used in the CICIoT2023 paper (8 groups):')
R.append('               Benign, DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce')
R.append('  Binary task: Category == "Benign" vs. everything else.')
R.append('* All %d files had the identical %d-column header; it is written once at the top.' % (len(members), NCOL))
R.append('* Rows keep their original values and order, grouped by class (folders A-Z, files')
R.append('  in numeric order). The file is NOT shuffled - shuffle before train/test splitting.')
R.append('* Nothing was removed except the data-quality drops listed below (if any).')
R.append('')
R.append('Class distribution (34 classes)')
R.append('-' * 72)
R.append('%-26s %-10s %5s %12s %8s' % ('Label', 'Category', 'Files', 'Rows', '%'))
for label, n in rows_by_label.most_common():
    R.append('%-26s %-10s %5d %12d %7.3f%%' % (label, cat_of[label], files_by_label[label], n, 100.0 * n / total_rows))
R.append('%-26s %-10s %5d %12d %7.3f%%' % ('TOTAL', '', len(members), total_rows, 100.0))
big, small = rows_by_label.most_common()[0], rows_by_label.most_common()[-1]
R.append('Imbalance: largest class %s (%d) is %.0fx the smallest %s (%d).'
         % (big[0], big[1], big[1] / max(small[1], 1), small[0], small[1]))
R.append('')
R.append('Category distribution (8 groups)')
R.append('-' * 72)
for c, n in cat_rows.most_common():
    R.append('%-12s %12d %7.3f%%' % (c, n, 100.0 * n / total_rows))
R.append('')
R.append('Data quality')
R.append('-' * 72)
for k in ('empty_lines_dropped', 'repeated_header_rows_dropped', 'malformed_rows_dropped',
          'rows_with_empty_or_nonnumeric_values_kept', 'files_missing_final_newline'):
    R.append('%-44s %d' % (k.replace('_', ' '), q.get(k, 0)))
for fn, ln in malformed_examples:
    R.append('  malformed example (%s): %r' % (fn, ln))
if empty_by_col:
    R.append('Empty values by column: ' + ', '.join('%s=%d' % kv for kv in empty_by_col.most_common()))
for col, cnt in nonnum_by_col.items():
    R.append('Non-numeric values in %s: %s' % (col, ', '.join('%r x%d' % kv for kv in cnt.most_common(5))))
R.append('')
R.append('Exact duplicate rows (all 39 feature values identical)')
R.append('-' * 72)
R.append('Unique feature vectors                        %d' % unique_feature_vectors)
R.append('Duplicate rows, ignoring label                %d (%.2f%%)'
         % (total_rows - unique_feature_vectors, 100.0 * (total_rows - unique_feature_vectors) / total_rows))
R.append('Duplicate rows within the same class          %d (%.2f%%)'
         % (total_rows - unique_feature_label_pairs, 100.0 * (total_rows - unique_feature_label_pairs) / total_rows))
R.append('Feature vectors that appear in >1 class       %d' % conflicting_vectors)
R.append('Rows carrying such a cross-class vector       %d' % sum(conflict_rows_by_label.values()))
R.append('(Duplicates were kept. Per-class counts are in class_distribution.csv.)')
R.append('')
R.append('Feature summary (1%% systematic sample = %d rows)' % sample_rows)
R.append('-' * 72)
R.append('%-18s %16s %16s %16s %8s' % ('Feature', 'min', 'max', 'mean', 'NaN/inf'))
for j, c in enumerate(COLS):
    if s_n[j]:
        R.append('%-18s %16.6g %16.6g %16.6g %8d%s' % (c, s_min[j], s_max[j], s_sum[j] / s_n[j], s_nonfinite[j],
                                                    '  <- constant in sample' if s_min[j] == s_max[j] else ''))
    else:
        R.append('%-18s %16s %16s %16s %8d' % (c, '-', '-', '-', s_nonfinite[j]))
R.append('')
R.append('Rows per source file')
R.append('-' * 72)
for fn, label, n in file_rows:
    R.append('%10d  %s' % (n, fn))
report = '\n'.join(R) + '\n'

zout.writestr('merge_report.txt', report)
zout.writestr('class_distribution.csv', class_csv)
for i in readme:
    zout.writestr(os.path.basename(i.filename), zin.read(i))
zout.close()

with open(os.path.join(OUT_DIR, 'merge_report.txt'), 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(report)
with open(os.path.join(OUT_DIR, 'class_distribution.csv'), 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(class_csv)
log('DONE. Output zip %.2f GB' % (os.path.getsize(ZOUT) / 1e9))
