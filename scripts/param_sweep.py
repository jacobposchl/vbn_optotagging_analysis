"""
Parameter sweep for optotagging thresholds.

Loads one representative session once, builds the opto PSTH array once,
then iterates over all (increase_in_FR, min_evoked_rate) combinations.
For each combination it generates the same diagnostics as parameter_tuning.ipynb
(scatter, heatmap, reliability, latency, shuffled FP control) and saves them
to an individual folder so you can compare configs side-by-side.

A summary CSV and sweep grid image are written at the end.

Usage:
    python scripts/param_sweep.py              # full sweep
    python scripts/param_sweep.py --plot-only  # regenerate sweep_grid.png from existing summary.csv

Edit the SWEEP PARAMETERS section below to change what gets tested.
"""

import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='hdmf')

import csv
import itertools
from pathlib import Path

_parser = argparse.ArgumentParser()
_parser.add_argument('--plot-only', action='store_true',
                     help='Skip the sweep; regenerate sweep_grid.png from existing summary.csv')
_args = _parser.parse_args()

import matplotlib
matplotlib.use('Agg')  # non-interactive — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import yaml

from visb_analysis import SessionHandler, UnitCollection
from visb_analysis.psth import make_population_psth
from visb_analysis.optotagging import (
    compute_trial_reliability,
    compute_spike_latency,
    compute_shuffled_fp_rate,
)
from visb_analysis.plots import (
    plot_opto_scatter,
    plot_opto_heatmaps,
    plot_reliability_histogram,
    plot_latency_histogram,
    plot_fp_summary,
)


# ---------------------------------------------------------------------------
# SWEEP PARAMETERS — edit this section
# ---------------------------------------------------------------------------

GENOTYPE      = 'Sst'   # 'Sst' or 'Vip'
SESSION_INDEX = 0        # which session to use (0 = first available)
REGIONS       = ['VIS']  # brain regions to include

# Grid values to sweep
FOLD_VALUES     = [2, 3, 5]   # increase_in_FR candidates
MIN_RATE_VALUES = [10, 30,50]  # min_evoked_rate candidates (Hz)

# Evoked window (kept fixed — only fold and min_rate are swept)
EVOKED_START_MS = 1
EVOKED_END_MS   = 9

# Shuffled FP control settings
N_SHUFFLES = 20
SHIFT_S    = 5.0

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

ROOT      = Path(__file__).resolve().parents[1]
OUT_ROOT  = ROOT / 'results' / 'param_sweep' / GENOTYPE
CONFIG    = ROOT / 'configs' / 'base.yaml'
summary_path = OUT_ROOT / 'summary.csv'

if _args.plot_only:
    # ---------------------------------------------------------------------------
    # PLOT-ONLY: read existing summary CSV, skip all data loading
    # ---------------------------------------------------------------------------
    if not summary_path.exists():
        raise FileNotFoundError(
            f'No summary CSV found at {summary_path}\n'
            'Run the full sweep first (without --plot-only).'
        )
    with summary_path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        summary_rows = [
            {
                'fold':             int(r['fold']),
                'min_rate_hz':      int(r['min_rate_hz']),
                'n_cre_pos':        int(r['n_cre_pos']),
                'pct_cre_pos':      float(r['pct_cre_pos']),
                'med_reliability':  float(r['med_reliability']),
                'frac_rel_gt_0.5':  float(r['frac_rel_gt_0.5']),
                'med_latency_ms':   float(r['med_latency_ms']),
                'mean_shuffled_fp': float(r['mean_shuffled_fp']),
                'std_shuffled_fp':  float(r['std_shuffled_fp']),
            }
            for r in reader
        ]
    print(f'Loaded {len(summary_rows)} rows from {summary_path}')

else:
    # ---------------------------------------------------------------------------
    # FULL SWEEP
    # ---------------------------------------------------------------------------

    with CONFIG.open('r') as f:
        config = yaml.safe_load(f)

    # --- Load session (once) ---
    print(f'Loading {GENOTYPE} session {SESSION_INDEX}...')
    session_handler = SessionHandler(Path(config['cache_dir']))
    session_table   = session_handler.session_table

    filtered   = session_table[
        session_table['genotype'].str.contains(GENOTYPE) &
        (session_table['experience_level'] == 'Novel')
    ]
    session_id = filtered.index[SESSION_INDEX]
    session    = session_handler.cache.get_ecephys_session(session_id)

    row = filtered.loc[session_id]
    print(f'  Session ID : {session_id}')
    print(f'  Genotype   : {row["genotype"]}')
    print(f'  Experience : {row["experience_level"]}')

    # --- Filter units (once) ---
    units = (
        UnitCollection(session=session)
        .filter_quality(config['unit_filters'])
        .filter_region(REGIONS)
    )
    print(f'  Units after filters: {len(units)}')

    # --- Build opto array (once) ---
    opto_table = session.optotagging_table
    sel_pulses = opto_table[
        (opto_table['duration'] <= config['optotagging']['max_pulse_duration']) &
        (opto_table['level'] == opto_table['level'].max())
    ]
    pulse_times = sel_pulses['start_time'].values
    print(f'  Selected pulses: {len(pulse_times)}')

    TIME_BEFORE = config['psth']['time_before']
    DURATION    = config['psth']['duration']
    BIN_SIZE    = config['psth']['bin_size']

    print('Building opto PSTH array (this takes a moment)...')
    opto_array, time_bins, unit_ids = make_population_psth(
        units, pulse_times, TIME_BEFORE, DURATION, BIN_SIZE
    )
    print(f'  opto_array shape: {opto_array.shape}  (neurons x time_bins x trials)')

    mean_opto    = np.nanmean(opto_array, axis=2)
    baseline_idx = (time_bins >= -0.010) & (time_bins < -0.002)
    evoked_idx   = (
        (time_bins >= EVOKED_START_MS / 1000) &
        (time_bins <  EVOKED_END_MS   / 1000)
    )
    baseline_rate = np.mean(mean_opto[:, baseline_idx], axis=1)
    evoked_rate   = np.mean(mean_opto[:, evoked_idx],   axis=1)

    evoked_window  = [EVOKED_START_MS / 1000, EVOKED_END_MS / 1000]
    reliability, _ = compute_trial_reliability(units, pulse_times, evoked_window)
    latency_ms, _  = compute_spike_latency(units, pulse_times, evoked_window)

    spike_times_dict   = units.get_spike_times()
    unit_id_list       = list(units.units.index)
    opto_config_tuning = {**config['optotagging'], 'evoked_window': evoked_window}
    n_total            = len(units)

    # --- Sweep ---
    summary_rows = []
    combos = list(itertools.product(FOLD_VALUES, MIN_RATE_VALUES))
    print(f'\nRunning sweep: {len(combos)} configs...\n')

    for increase_in_FR, min_evoked_rate in combos:
        label   = f'fold{increase_in_FR}_rate{min_evoked_rate}'
        out_dir = OUT_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)

        cre_pos_idx = (
            (evoked_rate > min_evoked_rate) &
            ((evoked_rate / (baseline_rate + 1)) > increase_in_FR)
        )
        n_pos = int(cre_pos_idx.sum())
        print(f'  {label}  ->  Cre+={n_pos}/{n_total}  ({100*n_pos/n_total:.1f}%)')

        fig = plot_opto_scatter(
            baseline_rate, evoked_rate, cre_pos_idx,
            increase_in_FR, min_evoked_rate,
            session_id=session_id, genotype=GENOTYPE,
        )
        fig.savefig(out_dir / 'scatter.png', dpi=120)
        plt.close(fig)

        fig = plot_opto_heatmaps(
            mean_opto, cre_pos_idx, time_bins,
            EVOKED_START_MS, EVOKED_END_MS,
            increase_in_FR, min_evoked_rate, genotype=GENOTYPE,
        )
        fig.savefig(out_dir / 'heatmap.png', dpi=120)
        plt.close(fig)

        fig = plot_reliability_histogram(reliability, cre_pos_idx, genotype=GENOTYPE)
        fig.savefig(out_dir / 'reliability.png', dpi=120)
        plt.close(fig)

        fig = plot_latency_histogram(latency_ms, cre_pos_idx, genotype=GENOTYPE)
        fig.savefig(out_dir / 'latency.png', dpi=120)
        plt.close(fig)

        fp_counts, _ = compute_shuffled_fp_rate(
            spike_times_dict, unit_id_list,
            pulse_times, opto_config_tuning,
            increase_in_FR, min_evoked_rate,
            n_shuffles=N_SHUFFLES, shift_s=SHIFT_S,
        )
        mean_fp = float(np.mean(fp_counts))
        std_fp  = float(np.std(fp_counts))

        fig = plot_fp_summary(
            [{'session_id': session_id, 'cre_pos_mask': cre_pos_idx}],
            [{'session_id': session_id, 'mean_fp': mean_fp, 'std_fp': std_fp}],
            genotype=GENOTYPE,
        )
        fig.savefig(out_dir / 'fp_control.png', dpi=120)
        plt.close(fig)

        if n_pos > 0:
            med_reliability = float(np.nanmedian(reliability[cre_pos_idx]))
            frac_above_half = float((reliability[cre_pos_idx] > 0.5).mean())
            valid_lat       = latency_ms[cre_pos_idx]
            valid_lat       = valid_lat[~np.isnan(valid_lat)]
            med_latency_ms  = float(np.median(valid_lat)) if len(valid_lat) > 0 else float('nan')
        else:
            med_reliability = frac_above_half = med_latency_ms = float('nan')

        summary_rows.append({
            'fold':              increase_in_FR,
            'min_rate_hz':       min_evoked_rate,
            'n_cre_pos':         n_pos,
            'pct_cre_pos':       round(100 * n_pos / n_total, 2),
            'med_reliability':   round(med_reliability, 3),
            'frac_rel_gt_0.5':   round(frac_above_half, 3),
            'med_latency_ms':    round(med_latency_ms, 2),
            'mean_shuffled_fp':  round(mean_fp, 2),
            'std_shuffled_fp':   round(std_fp, 2),
        })

    # --- Write CSV ---
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys())
    with summary_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f'\nDone. Results in: {OUT_ROOT}')
    print(f'Summary CSV   : {summary_path}')

print(f'\nDone. Results in: {OUT_ROOT}')
print(f'Summary CSV   : {summary_path}')

# ---------------------------------------------------------------------------
# SWEEP GRID VISUALIZATION
# ---------------------------------------------------------------------------

folds     = sorted(set(r['fold']        for r in summary_rows))
min_rates = sorted(set(r['min_rate_hz'] for r in summary_rows))

# Build lookup: (fold, rate) -> row
lookup = {(r['fold'], r['min_rate_hz']): r for r in summary_rows}

metrics = [
    # (column key,       panel title,                        colormap,     fmt,        filename)
    ('n_cre_pos',        'Cre+ count',                       'YlOrRd',    '{:.0f}',   'sweep_cre_count.png'),
    ('med_reliability',  'Median reliability',               'YlGn',      '{:.2f}',   'sweep_reliability.png'),
    ('frac_rel_gt_0.5',  'Fraction reliability > 0.5',       'YlGn',      '{:.2f}',   'sweep_frac_reliability.png'),
    ('med_latency_ms',   'Median first-spike latency (ms)\n(target: 2–4 ms)', 'Blues', '{:.1f}', 'sweep_latency.png'),
    ('mean_shuffled_fp', 'Mean shuffled FP',                 'YlOrRd_r',  '{:.1f}',   'sweep_shuffled_fp.png'),
]

suptitle = (
    f'{GENOTYPE}  |  Parameter sweep  |  window={EVOKED_START_MS}-{EVOKED_END_MS} ms\n'
    f'y = fold threshold,  x = min evoked rate (Hz)'
)

for col, title, cmap, fmt, filename in metrics:
    grid = np.full((len(folds), len(min_rates)), np.nan)
    for i, fold in enumerate(folds):
        for j, rate in enumerate(min_rates):
            row = lookup.get((fold, rate))
            if row is not None:
                v = row[col]
                grid[i, j] = v if v == v else np.nan  # NaN check

    vmin = np.nanmin(grid)
    vmax = np.nanmax(grid)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(suptitle, fontsize=9)
    im = ax.imshow(grid, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(len(folds)):
        for j in range(len(min_rates)):
            val = grid[i, j]
            if not np.isnan(val):
                normed = (val - vmin) / (vmax - vmin + 1e-9)
                text_color = 'black' if normed < 0.6 else 'white'
                ax.text(j, i, fmt.format(val),
                        ha='center', va='center', fontsize=11,
                        color=text_color, fontweight='bold')

    ax.set_xticks(range(len(min_rates)))
    ax.set_xticklabels([str(r) for r in min_rates])
    ax.set_yticks(range(len(folds)))
    ax.set_yticklabels([f'{f}x' for f in folds])
    ax.set_xlabel('Min evoked rate (Hz)')
    ax.set_ylabel('Fold threshold')
    ax.set_title(title, fontsize=10)

    fig.tight_layout()
    out_path = OUT_ROOT / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  {filename}')
