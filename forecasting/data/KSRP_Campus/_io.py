"""KSRP Campus - folder/Excel I/O helpers + per-yearly-file matcher.

ASCII-only (no em-dashes) to remain safe under workspace mount limits.
"""

import glob
import os
import re

import numpy as np
import pandas as pd


FOLDER_ALIASES = {
    'power':         ['Power', 'power'],
    'gas_usage':     ['Gas usage', 'gas_usage', 'Gas Usage'],
    'heat_recovery': ['Heat recovery', 'heat_recovery', 'Heat Recovery'],
    'loads':         ['Electricity load & Heating load & Cooling load & Hot water load',
                      'Electricity load Heating load Cooling load Hot water load',
                      'loads', 'Loads'],
    'weather':       ['Weather data', 'weather', 'Weather'],
    'buildings':     ['Cooling and heating load data of individual buildings',
                      'Individual buildings', 'Buildings'],
}


def _norm(s):
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def match_columns(columns, rules):
    """{canonical -> [matched raw columns]}. '!' prefix = negative token."""
    out = {k: [] for k in rules}
    taken = set()
    for canon, rule_lists in rules.items():
        for raw in columns:
            if raw in taken:
                continue
            n = _norm(raw)
            for tokens in rule_lists:
                pos = [t for t in tokens if not t.startswith('!')]
                neg = [t[1:] for t in tokens if t.startswith('!')]
                if all(t in n for t in pos) and not any(t in n for t in neg):
                    out[canon].append(raw)
                    taken.add(raw)
                    break
    return out


def resolve_folder(raw_dir, key):
    for alias in FOLDER_ALIASES[key]:
        cand = os.path.join(raw_dir, alias)
        if os.path.isdir(cand):
            return cand
    if not os.path.isdir(raw_dir):
        return None
    needles = [_norm(a) for a in FOLDER_ALIASES[key]]
    for entry in os.listdir(raw_dir):
        full = os.path.join(raw_dir, entry)
        if not os.path.isdir(full):
            continue
        n = _norm(entry)
        for needle in needles:
            if all(tok in n for tok in needle.split()):
                return full
    return None


def list_yearly_files(folder):
    paths = (glob.glob(os.path.join(folder, '*.xlsx'))
             + glob.glob(os.path.join(folder, '*.xls')))
    paths = [p for p in paths if not os.path.basename(p).startswith('~$')]
    def _year(p):
        m = re.search(r'(20\d{2})', os.path.basename(p))
        return int(m.group(1)) if m else 9999
    return sorted(paths, key=_year)


def read_yearly(path):
    df = pd.read_excel(path, engine='openpyxl')
    ts_col = None
    for c in df.columns:
        n = _norm(c)
        if 'time' in n or 'date' in n or 'datetime' in n:
            ts_col = c
            break
    if ts_col is None:
        ts_col = df.columns[0]
    df['__ts__'] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.dropna(subset=['__ts__']).set_index('__ts__').sort_index()
    if ts_col in df.columns:
        df = df.drop(columns=[ts_col])
    df.index.name = 'date'
    return df


def sum_matched(df, matches, agg='sum'):
    out = pd.DataFrame(index=df.index)
    for canon, raws in matches.items():
        if not raws:
            out[canon] = np.nan
            continue
        sub = df[raws].apply(pd.to_numeric, errors='coerce')
        if agg == 'mean':
            out[canon] = sub.mean(axis=1)
        else:
            out[canon] = sub.sum(axis=1, min_count=1)
    return out


def to_hourly(df, idx):
    if df.empty:
        return pd.DataFrame(index=idx)
    hourly = df.resample('h').mean()
    return hourly.reindex(idx).ffill(limit=3).bfill(limit=3)


def load_folder(raw_dir, key):
    folder = resolve_folder(raw_dir, key)
    if folder is None:
        print(f'  [warn] folder "{key}" not found in {raw_dir}')
        return pd.DataFrame()
    paths = list_yearly_files(folder)
    if not paths:
        print(f'  [warn] no .xlsx in {folder}')
        return pd.DataFrame()
    print(f'  [{key}] {len(paths)} yearly files from {os.path.basename(folder)}')
    frames = []
    for p in paths:
        try:
            frames.append(read_yearly(p))
        except Exception as e:
            print(f'    [warn] {os.path.basename(p)}: {e}')
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep='first')]


def materialise(df, rules, label, agg='sum'):
    if df.empty:
        return pd.DataFrame()
    matches = match_columns(df.columns, rules)
    for canon, raws in matches.items():
        tag = ' + '.join(raws) if raws else '(none)'
        print(f'    [{label}] {canon:32s} <- {tag}')
    return sum_matched(df, matches, agg=agg)


def load_folder_canonical(raw_dir, key, rules, label, agg='sum'):
    """Per-yearly-file canonical materialisation.

    Matches each yearly Excel in isolation against ``rules``, then
    concatenates the canonical frames. Handles unit-label drift across
    years (e.g. KSRP 2008 has "(GJ)"-labelled but kW-valued load columns
    while every other year uses "(kW)") without double-counting.
    """
    folder = resolve_folder(raw_dir, key)
    if folder is None:
        print(f'  [warn] folder "{key}" not found')
        return pd.DataFrame()
    paths = list_yearly_files(folder)
    if not paths:
        return pd.DataFrame()
    print(f'  [{key}] {len(paths)} yearly files from {os.path.basename(folder)}')
    seen = set()
    frames = []
    for p in paths:
        try:
            df = read_yearly(p)
        except Exception as e:
            print(f'    [warn] {os.path.basename(p)}: {e}')
            continue
        matches = match_columns(df.columns, rules)
        sig = tuple(sorted((c, tuple(rs)) for c, rs in matches.items()))
        if sig not in seen:
            seen.add(sig)
            yr = ''.join(c for c in os.path.basename(p) if c.isdigit())[:4]
            print(f'    [{label}] {yr}: column signature')
            for c, rs in matches.items():
                tag = ' + '.join(rs) if rs else '(none)'
                print(f'        {c:32s} <- {tag}')
        frames.append(sum_matched(df, matches, agg=agg))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep='first')]


# ---------------------------------------------------------------------------
# Trailing padding to keep this file at or above the cowork-mount historical
# high-water mark. The cowork filesystem fills any tail-truncation gap with
# null bytes, which Python rejects as 'source code string cannot contain null
# bytes'. By padding the file with a long inert string literal, the mount has
# nothing to null-fill and the module loads cleanly. Do not delete this block.
# ---------------------------------------------------------------------------

_COWORK_MOUNT_PADDING = (
    'X' * 400
)

