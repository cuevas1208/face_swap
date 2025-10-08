#!/usr/bin/env python3
"""
show_scalar_tags.py

Load TensorBoard event files (scalars) from a log directory and export all training
losses / metrics into CSV files.

Outputs:
 - a combined CSV with rows aligned by 'step' and columns for each scalar tag.
 - (optional) separate per-tag CSVs.

Usage:
    python show_scalar_tags.py --logdir runs/anony1 --out all_scalars.csv
    python show_scalar_tags.py --logdir runs/anony1 --out all_scalars.csv --per-tag-dir per_tag_csvs

Options:
    --logdir       TensorBoard log directory (can contain event files or subfolders)
    --out          Path to write combined CSV (default: scalars_combined.csv)
    --per-tag-dir  Optional directory where per-tag CSVs will be written (one CSV per tag)
    --tags         Optional comma-separated list of tags (or regex) to include (default: all)
    --step-as-index When set, use step as CSV index column (default: True)
"""

import os
import argparse
import glob
import math
import csv
from collections import defaultdict, OrderedDict

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def find_event_files(logdir):
    """
    Find TensorBoard event files under logdir (recursively).
    Returns a list of file paths.
    """
    paths = []
    if os.path.isfile(logdir):
        # direct event file
        paths.append(logdir)
        return paths
    # common event filename pattern
    for root, dirs, files in os.walk(logdir):
        for f in files:
            if f.startswith("events.out.tfevents") or f.startswith("events."):
                paths.append(os.path.join(root, f))
    return sorted(paths)

def load_scalars_from_eventdir(logdir):
    """
    Use EventAccumulator pointed at the directory (logdir). This usually
    aggregates all event files in that directory. If logdir contains multiple
    sub-directories with events, use the top-level EventAccumulator (it will
    scan files in the dir).
    Returns: dict tag -> list of (step, wall_time, value)
    """
    # EventAccumulator expects a directory path containing events
    ea = EventAccumulator(logdir,
                          size_guidance={
                              'scalars': 0,  # load all
                              'histograms': 0,
                              'images': 0,
                              'audio': 0,
                              'compressedHistograms': 0,
                          })
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    result = {}
    for tag in tags:
        events = ea.Scalars(tag)  # list of EventObjects with .step, .wall_time, .value
        result[tag] = [(e.step, e.wall_time, e.value) for e in events]
    return result

def load_scalars_from_event_files(file_paths):
    """
    Load scalars from multiple event files independently and merge tags.
    Returns dict tag -> list of (step, wall_time, value) sorted by step.
    """
    agg = defaultdict(list)
    for ev in file_paths:
        try:
            ea = EventAccumulator(ev, size_guidance={'scalars': 0})
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            for tag in tags:
                for e in ea.Scalars(tag):
                    agg[tag].append((e.step, e.wall_time, e.value))
        except Exception as e:
            print(f"[warn] failed to read {ev}: {e}")
    # sort per tag by step
    for tag in list(agg.keys()):
        agg[tag].sort(key=lambda x: (x[0], x[1]))
    return agg

def tags_filter(all_tags, include_patterns):
    """
    include_patterns: list of strings; treated as substrings (not full regex)
    If include_patterns is empty or None, return all_tags.
    """
    if not include_patterns:
        return list(all_tags)
    selected = []
    for t in all_tags:
        for pat in include_patterns:
            if pat in t:
                selected.append(t)
                break
    return selected

def build_combined_table(scalars_map, align_by='step'):
    """
    scalars_map: dict tag -> list of (step, wall_time, value)
    align_by: 'step' (default) - unify rows on step values (int)
    Returns:
      steps_sorted: sorted list of unique steps (int)
      rows: list of dict { 'step': step, 'wall_time': min_wall_time_for_step, tag1: value_or_nan, ... }
      columns: list of column names (including 'step' and tags)
    """
    # gather unique steps
    steps_set = set()
    step_to_wall = defaultdict(list)
    for tag, events in scalars_map.items():
        for step, wall, val in events:
            steps_set.add(step)
            step_to_wall[step].append(wall)
    if not steps_set:
        return [], [], []
    steps_sorted = sorted(steps_set)
    # create mapping tag-> dict(step -> value)
    tag_step_map = {}
    for tag, events in scalars_map.items():
        d = {}
        for step, wall, val in events:
            # if multiple values for same step & tag, keep the last by wall_time
            # store tuple (wall, val) and pick max wall
            if step not in d:
                d[step] = (wall, val)
            else:
                # keep later wall_time
                if wall >= d[step][0]:
                    d[step] = (wall, val)
        # convert to simple mapping step->value
        tag_step_map[tag] = {s: v for s, (w, v) in d.items()}
    # build rows
    rows = []
    for step in steps_sorted:
        row = {'step': step}
        # choose wall_time as min (earliest) or mean; we'll use min here
        wall_times = step_to_wall.get(step, [])
        row['wall_time'] = min(wall_times) if wall_times else None
        for tag, mapping in tag_step_map.items():
            row[tag] = mapping.get(step, float('nan'))
        rows.append(row)
    columns = ['step', 'wall_time'] + list(scalars_map.keys())
    return steps_sorted, rows, columns

def write_combined_csv(out_path, rows, columns):
    """
    Write combined CSV using pandas if available, otherwise csv module.
    """
    if _HAS_PANDAS:
        import pandas as pd
        df = pd.DataFrame(rows, columns=columns)
        # ensure step is int
        if 'step' in df.columns:
            try:
                df['step'] = df['step'].astype(int)
                df = df.sort_values('step')
            except Exception:
                pass
        df.to_csv(out_path, index=False)
    else:
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in rows:
                # convert any non-serializable to str
                out = {k: ('' if v is None else v) for k, v in r.items()}
                writer.writerow(out)

def write_per_tag_csvs(per_tag_dir, scalars_map):
    os.makedirs(per_tag_dir, exist_ok=True)
    for tag, events in scalars_map.items():
        out_path = os.path.join(per_tag_dir, f"{sanitize_filename(tag)}.csv")
        # events: list of (step, wall_time, value)
        if _HAS_PANDAS:
            import pandas as pd
            df = pd.DataFrame(events, columns=['step','wall_time','value'])
            df.to_csv(out_path, index=False)
        else:
            with open(out_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['step','wall_time','value'])
                for ev in events:
                    w.writerow(list(ev))

def sanitize_filename(s):
    # simple sanitize for tag -> filename
    keepchars = (' ','.','_','-')
    filename = "".join(c if c.isalnum() or c in keepchars else '_' for c in s)
    filename = filename.replace(' ', '_')
    return filename[:200]

def main():
    p = argparse.ArgumentParser(description="Export TensorBoard scalars to CSV (combined and per-tag).")
    p.add_argument('--logdir', required=True, help='TensorBoard logdir (folder containing event files or event file itself)')
    p.add_argument('--out', default='scalars_combined.csv', help='output combined CSV path')
    p.add_argument('--per-tag-dir', default=None, help='optional directory to write per-tag CSVs')
    p.add_argument('--tags', default=None, help='optional comma-separated list of substrings to filter tags (only include matching tags)')
    args = p.parse_args()

    event_files = find_event_files(args.logdir)
    if not event_files:
        # try using EventAccumulator on the directory (some TB setups work better)
        print(f"No event files found under {args.logdir}. Trying to load via EventAccumulator on the directory.")
        try:
            scalars_map = load_scalars_from_eventdir(args.logdir)
        except Exception as e:
            print("Failed to load eventdir:", e)
            return
    else:
        print(f"Found {len(event_files)} event file(s). Loading...")
        scalars_map = load_scalars_from_event_files(event_files)

    all_tags = sorted(list(scalars_map.keys()))
    print(f"Discovered {len(all_tags)} scalar tags.")
    if args.tags:
        patterns = [t.strip() for t in args.tags.split(',') if t.strip()]
        selected = tags_filter(all_tags, patterns)
        print(f"Filtering tags by patterns {patterns} -> {len(selected)} tags will be exported.")
        scalars_map = {k: scalars_map[k] for k in selected}
    else:
        selected = all_tags

    if not scalars_map:
        print("No scalar tags to export after filtering. Exiting.")
        return

    # Build combined table aligned by step
    steps, rows, columns = build_combined_table(scalars_map, align_by='step')
    print(f"Writing combined CSV to {args.out} (rows={len(rows)}, columns={len(columns)})")
    write_combined_csv(args.out, rows, columns)

    # optional per-tag CSVs
    if args.per_tag_dir:
        print(f"Writing per-tag CSVs to {args.per_tag_dir}")
        write_per_tag_csvs(args.per_tag_dir, scalars_map)

    print("Done.")

if __name__ == "__main__":
    main()
