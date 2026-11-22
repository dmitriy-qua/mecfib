"""
KSRP Campus — per-yearly-file canonical loader.

Lives in its own file (not in _io.py) because the cowork workspace's file
layer truncates _io.py at its prior 6682-byte high-water mark.

The substring matcher in ``_io.match_columns`` is normally applied AFTER
concatenating all yearly Excels of a folder. That single-pass behaviour
fails when a column name drifts between years (KSRP 2008 publishes
``Cooling load (GJ)`` while every other year publishes
``Cooling load (kW)`` — both are kW-magnitude values despite the label).
The concatenation creates a union of both column names and the matcher
either claims both (double-counts) or rejects the GJ variant outright
(loses 2008 entirely).

``load_folder_canonical`` fixes this by matching each yearly file in
isolation, so each year only sees its own present columns. The resulting
yearly canonical frames are concatenated at the end.
"""

import os

import pandas as pd

from forecasting.data.KSRP_Campus._io import (
    resolve_folder, list_yearly_files, read_yearly,
    match_columns, sum_matched,
)


def load_folder_canonical(raw_dir: str, key: str, rules: dict,
                          label: str, agg: str = 'sum') -> pd.DataFrame:
    """Per-yearly-file canonical materialisation.

    Reads every yearly Excel in the source folder, applies the substring
    matcher to *each file in isolation*, materialises canonical columns
    per-year, then concatenates. Prints one diagnostic line per distinct
    column signature so the operator can see exactly which years carry
    which column variant.
    """
    folder = resolve_folder(raw_dir, key)
    if folder is None:
        print(f'  [warn] folder "{key}" not found in {raw_dir} - skipping.')
        return pd.DataFrame()
    paths = list_yearly_files(folder)
    if not paths:
        print(f'  [warn] no .xlsx files in {folder} - skipping.')
        return pd.DataFrame()
    print(f'  [{key}] {len(paths)} yearly files from {os.path.basename(folder)}')

    seen = set()
    frames = []
    for p in paths:
        try:
            df = read_yearly(p)
        except Exception as e:
            print(f'    [warn] failed to read {os.path.basename(p)}: {e}')
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
        canon = sum_matched(df, matches, agg=agg)
        frames.append(canon)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep='first')]
    return out
