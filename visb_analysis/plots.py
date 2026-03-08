"""
Plotting functions shared across analysis notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt

# Optotagging scatter

def plot_opto_scatter(baseline_rate, evoked_rate, cre_pos_idx,
                      increase_in_FR, min_evoked_rate,
                      session_id=None, genotype=None, figsize=(7, 7)):
    """
    Scatter plot of baseline vs evoked firing rate with threshold lines.

    Returns the matplotlib Figure.
    """
    n_pos = int(cre_pos_idx.sum())
    n_neg = int((~cre_pos_idx).sum())

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(baseline_rate[~cre_pos_idx], evoked_rate[~cre_pos_idx],
               s=5, alpha=0.5, label=f'Cre- (n={n_neg})')
    ax.scatter(baseline_rate[cre_pos_idx], evoked_rate[cre_pos_idx],
               s=10, color='red', label=f'Cre+ (n={n_pos})')

    lim = max(evoked_rate.max(), min_evoked_rate * 1.5) * 1.1
    ax.plot([0, lim], [0, lim * increase_in_FR], ':r', linewidth=1,
            label=f'{increase_in_FR}x fold')
    ax.axhline(min_evoked_rate, color='blue', linestyle=':', linewidth=1,
               label=f'{min_evoked_rate} Hz min')

    ax.set_xlim([0, lim])
    ax.set_ylim([0, lim])
    ax.set_xlabel('Baseline Rate (Hz)')
    ax.set_ylabel('Evoked Rate (Hz)')

    title = 'Baseline vs Evoked Rate'
    if session_id is not None and genotype is not None:
        title = f'Session {session_id} — {genotype}'
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# Optotagging heatmaps

def plot_opto_heatmaps(mean_opto, cre_pos_idx, time_bins,
                       evoked_start_ms, evoked_end_ms,
                       increase_in_FR, min_evoked_rate, genotype=None):
    """
    Side-by-side heatmaps of Cre+ and Cre- mean opto responses.

    mean_opto: (neurons, time_bins) array of trial-averaged firing rates.
    Returns the matplotlib Figure.
    """
    view_mask = (time_bins >= -0.010) & (time_bins < 0.025)
    t_ms = time_bins[view_mask] * 1000

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, mask, label in zip(axes,
                                [cre_pos_idx, ~cre_pos_idx],
                                ['Cre+', 'Cre-']):
        data = mean_opto[mask, :][:, view_mask]
        im = ax.imshow(
            data,
            extent=[t_ms[0], t_ms[-1], 0, data.shape[0]],
            origin='lower', aspect='auto', vmin=0, vmax=300
        )
        ax.axvline(0, color='white', linestyle=':', linewidth=1)
        ax.axvline(evoked_start_ms, color='cyan', linestyle=':', linewidth=1)
        ax.axvline(evoked_end_ms,   color='cyan', linestyle=':', linewidth=1)
        ax.set_xlabel('Time from laser onset (ms)')
        ax.set_ylabel('Unit #')
        ax.set_title(f'{label} (n={int(mask.sum())})')
        plt.colorbar(im, ax=ax, label='Mean FR (Hz)')

    suptitle = (
        f'{genotype}  |  ' if genotype else ''
    ) + f'fold={increase_in_FR}x  |  min={min_evoked_rate} Hz  |  window={evoked_start_ms}-{evoked_end_ms} ms'
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig


# Units per session bar chart

def plot_units_per_session(all_units_meta, title='Units per session'):
    """
    Stacked bar chart showing optotagged vs non-optotagged units per session.

    all_units_meta: list of per-session unit metadata DataFrames.
    Returns the matplotlib Figure.
    """
    opto_counts     = [meta['optotagged'].fillna(False).sum() for meta in all_units_meta]
    non_opto_counts = [(~meta['optotagged'].fillna(False)).sum() for meta in all_units_meta]
    x = np.arange(len(opto_counts))

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.bar(x, non_opto_counts, label='Non-optotagged', color='steelblue')
    ax.bar(x, opto_counts, bottom=non_opto_counts, label='Optotagged', color='coral')
    ax.set_xlabel('Session index')
    ax.set_ylabel('Units')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# Population PSTH

def plot_population_psth(psth_opto, psth_non_opto, time_bins, title=None):
    """
    Population PSTH with SEM shading for optotagged vs non-optotagged units.

    Returns the matplotlib Figure.
    """
    groups = {
        'Optotagged':     (psth_opto,     'coral'),
        'Non-optotagged': (psth_non_opto, 'steelblue'),
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, (psth, color) in groups.items():
        if len(psth) == 0:
            continue
        mean = psth.mean(axis=0)
        sem  = psth.std(axis=0) / np.sqrt(len(psth))
        ax.plot(time_bins, mean, color=color, lw=1.5,
                label=f'{label} (n={len(psth)})')
        ax.fill_between(time_bins, mean - sem, mean + sem, alpha=0.3, color=color)

    ax.axvline(0, color='k', lw=1, ls='--')
    ax.set_xlabel('Time from change (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(title or 'Population PSTH — optotagged vs non-optotagged')
    ax.legend()
    fig.tight_layout()
    return fig


# PSTH heatmaps sorted by response magnitude

def plot_psth_heatmaps(psth_opto, psth_non_opto, time_bins, suptitle=None,
                       time_range=None, shared_scale=False):
    """
    Side-by-side heatmaps of optotagged and non-optotagged units,
    sorted by post-stimulus response magnitude.

    time_range : (t_start, t_end) in seconds to crop the x-axis view.
                 e.g. (-0.25, 0.75) to zoom around the change. Default: full range.
    shared_scale : if True, both panels share the same vmax (95th pct of all units).
                   if False (default), each panel uses its own 95th pct.

    Returns the matplotlib Figure.
    """
    if time_range is not None:
        view_mask = (time_bins >= time_range[0]) & (time_bins <= time_range[1])
        t_view = time_bins[view_mask]
    else:
        view_mask = np.ones(len(time_bins), dtype=bool)
        t_view = time_bins

    post_mask = (time_bins >= 0) & (time_bins < 0.5)

    if shared_scale:
        psth_all = np.concatenate([psth_opto, psth_non_opto], axis=0) if len(psth_non_opto) else psth_opto
        vmax_opto = vmax_non_opto = np.percentile(psth_all, 95)
    else:
        vmax_opto     = np.percentile(psth_opto, 95) if len(psth_opto) else 1
        vmax_non_opto = np.percentile(psth_non_opto, 95) if len(psth_non_opto) else 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, psth, title, vmax in [
        (axes[0], psth_opto,     'Optotagged',     vmax_opto),
        (axes[1], psth_non_opto, 'Non-optotagged', vmax_non_opto),
    ]:
        if len(psth) == 0:
            ax.set_title(f'{title} (n=0)')
            continue
        sort_idx = np.argsort(psth[:, post_mask].mean(axis=1))[::-1]
        im = ax.imshow(
            psth[sort_idx][:, view_mask],
            aspect='auto',
            extent=[t_view[0], t_view[-1], len(psth), 0],
            vmin=0, vmax=vmax,
            cmap='viridis'
        )
        ax.axvline(0, color='w', lw=1, ls='--')
        ax.set_xlabel('Time from change (s)')
        ax.set_ylabel('Unit (sorted by response)')
        ax.set_title(f'{title} (n={len(psth)})')
        plt.colorbar(im, ax=ax, label='Firing rate (Hz)')

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


# Multi-session combined scatter

def plot_multi_session_scatter(session_results, increase_in_FR, min_evoked_rate,
                               genotype=None, figsize=(7, 7)):
    """
    Combined scatter of baseline vs evoked rate across all sessions.

    session_results: list of dicts with keys:
        'session_id', 'baseline_rates', 'evoked_rates', 'cre_pos_mask'

    Cre- units are plotted in gray; Cre+ units are colored per session.
    Returns the matplotlib Figure.
    """
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=figsize)

    # Plot all Cre- first so they sit behind
    for res in session_results:
        neg = ~res['cre_pos_mask']
        ax.scatter(res['baseline_rates'][neg], res['evoked_rates'][neg],
                   s=5, alpha=0.6, color='gray', linewidths=0)

    # Plot Cre+ per session with distinct colors
    for i, res in enumerate(session_results):
        pos = res['cre_pos_mask']
        if pos.sum() == 0:
            continue
        color = colors[i % len(colors)]
        ax.scatter(res['baseline_rates'][pos], res['evoked_rates'][pos],
                   s=18, color=color, zorder=3,
                   label=f"Session {i} (n={pos.sum()})")

    # Threshold lines — square axes scaled to evoked range
    all_evok = np.concatenate([r['evoked_rates'] for r in session_results])
    lim = max(all_evok.max(), min_evoked_rate * 1.5) * 1.1
    ax.plot([0, lim], [0, lim * increase_in_FR], ':r', linewidth=1,
            label=f'{increase_in_FR}x fold')
    ax.axhline(min_evoked_rate, color='blue', linestyle=':', linewidth=1,
               label=f'{min_evoked_rate} Hz min')

    ax.set_xlim([0, lim])
    ax.set_ylim([0, lim])
    ax.set_xlabel('Baseline Rate (Hz)')
    ax.set_ylabel('Evoked Rate (Hz)')
    title = f'{genotype} — all sessions' if genotype else 'All sessions'
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


# Multi-session Cre+ count bar chart

def plot_multi_session_counts(session_results, genotype=None):
    """
    Bar chart of Cre+ and Cre- unit counts per session.

    session_results: list of dicts with keys:
        'session_id', 'cre_pos_mask'

    Returns the matplotlib Figure.
    """
    n_sessions   = len(session_results)
    pos_counts   = [int(r['cre_pos_mask'].sum()) for r in session_results]
    neg_counts   = [int((~r['cre_pos_mask']).sum()) for r in session_results]
    x = np.arange(n_sessions)

    fig, ax = plt.subplots(figsize=(max(8, n_sessions * 0.5), 3))
    ax.bar(x, neg_counts, label='Cre-', color='steelblue')
    ax.bar(x, pos_counts, bottom=neg_counts, label='Cre+', color='coral')

    # Annotate each bar with the Cre+ count
    for i, (pos, neg) in enumerate(zip(pos_counts, neg_counts)):
        ax.text(i, neg + pos + 0.5, str(pos), ha='center', va='bottom',
                fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=8)
    ax.set_xlabel('Session index')
    ax.set_ylabel('Units')
    title = f'{genotype} — Cre+ counts per session' if genotype else 'Cre+ counts per session'
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# Region distribution bar chart

def plot_region_distribution(meta_all, genotype=None):
    """
    Bar chart of Cre+ and Cre- unit counts per brain region,
    pooled across all sessions, sorted by Cre+ count descending.

    meta_all: combined unit metadata DataFrame with 'structure_acronym'
              and 'optotagged' columns.
    Returns the matplotlib Figure.
    """
    import pandas as pd
    opto_col = meta_all['optotagged'].fillna(False).astype(bool)
    counts = (
        meta_all.groupby('structure_acronym')['optotagged']
        .agg(
            cre_pos=lambda x: x.fillna(False).astype(bool).sum(),
            cre_neg=lambda x: (~x.fillna(False).astype(bool)).sum(),
        )
        .sort_values('cre_pos', ascending=False)
    )

    x = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(max(10, len(counts) * 0.6), 4))
    ax.bar(x, counts['cre_neg'], label='Cre-', color='steelblue')
    ax.bar(x, counts['cre_pos'], bottom=counts['cre_neg'], label='Cre+', color='coral')

    for i, (pos, neg) in enumerate(zip(counts['cre_pos'], counts['cre_neg'])):
        if pos > 0:
            ax.text(i, neg + pos + 0.3, str(pos), ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Brain region')
    ax.set_ylabel('Units')
    title = f'{genotype} — Cre+ distribution by region' if genotype else 'Cre+ distribution by region'
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# Cre+ fraction per region

def plot_cre_fraction_by_region(meta_all, genotype=None, min_units=10):
    """
    Bar chart of Cre+ fraction (%) per brain region, sorted descending.
    Regions with fewer than min_units total units are excluded.

    meta_all   : combined unit metadata DataFrame with 'structure_acronym'
                 and 'optotagged' columns.
    min_units  : minimum total units a region must have to be shown (default 10).
    Returns the matplotlib Figure.
    """
    opto = meta_all['optotagged'].fillna(False).astype(bool)
    counts = (
        meta_all.groupby('structure_acronym')['optotagged']
        .agg(
            cre_pos=lambda x: x.fillna(False).astype(bool).sum(),
            total=lambda x: len(x),
        )
    )
    counts = counts[counts['total'] >= min_units].copy()
    counts['fraction'] = counts['cre_pos'] / counts['total'] * 100
    counts = counts.sort_values('fraction', ascending=False)

    x = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(max(10, len(counts) * 0.6), 4))
    bars = ax.bar(x, counts['fraction'], color='coral')

    for i, (frac, pos, tot) in enumerate(
        zip(counts['fraction'], counts['cre_pos'], counts['total'])
    ):
        ax.text(i, frac + 0.3, f'{frac:.1f}%\n({pos}/{tot})',
                ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Brain region')
    ax.set_ylabel('Cre+ fraction (%)')
    title = f'{genotype} — Cre+ fraction by region (min {min_units} units)' if genotype \
        else f'Cre+ fraction by region (min {min_units} units)'
    ax.set_title(title)
    fig.tight_layout()
    return fig


# Trial reliability histogram

def plot_reliability_histogram(reliability, cre_pos_idx, genotype=None):
    """
    Overlaid histogram of trial reliability for all units (gray) and Cre+ (red).

    reliability  : np.ndarray, shape (n_units,) — fraction of trials with >=1 spike
    cre_pos_idx  : np.ndarray of bool, shape (n_units,)
    Returns the matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)

    ax.hist(reliability, bins=bins, color='gray', alpha=0.6,
            label=f'All units (n={len(reliability)})')
    ax.hist(reliability[cre_pos_idx], bins=bins, color='red', alpha=0.7,
            label=f'Cre+ (n={int(cre_pos_idx.sum())})')

    ax.set_xlabel('Trial reliability (fraction of trials with ≥1 spike in evoked window)')
    ax.set_ylabel('Unit count')
    title = 'Trial Reliability Distribution'
    if genotype:
        title = f'{genotype} — {title}'
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# Spike latency histogram

def plot_latency_histogram(latency_ms, cre_pos_idx, genotype=None):
    """
    Histogram of mean first-spike latency (ms) for Cre+ units.
    NaN values (units with no spikes in any trial) are excluded.

    latency_ms   : np.ndarray, shape (n_units,) — NaN where no spikes
    cre_pos_idx  : np.ndarray of bool, shape (n_units,)
    Returns the matplotlib Figure.
    """
    cre_pos_latency = latency_ms[cre_pos_idx]
    valid = cre_pos_latency[~np.isnan(cre_pos_latency)]

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(valid) > 0:
        ax.hist(valid, bins=20, color='red', alpha=0.8,
                label=f'Cre+ with spikes (n={len(valid)})')
        med = np.median(valid)
        ax.axvline(med, color='darkred', linestyle='--', linewidth=1.5,
                   label=f'Median = {med:.2f} ms')
    else:
        ax.text(0.5, 0.5, 'No Cre+ units with spikes in evoked window',
                transform=ax.transAxes, ha='center', va='center')

    ax.set_xlabel('Mean first-spike latency (ms)')
    ax.set_ylabel('Unit count')
    title = 'Cre+ First-Spike Latency'
    if genotype:
        title = f'{genotype} — {title}'
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# Shuffled FP summary

def plot_fp_summary(session_results, fp_rates, genotype=None):
    """
    Grouped bar chart: true Cre+ count vs mean shuffled FP count per session.

    session_results : list of dicts with keys 'session_id', 'cre_pos_mask'
    fp_rates        : list of dicts with keys 'session_id', 'mean_fp', 'std_fp'
    Returns the matplotlib Figure.
    """
    n = len(session_results)
    true_counts = [int(r['cre_pos_mask'].sum()) for r in session_results]
    mean_fps    = [d['mean_fp'] for d in fp_rates]
    std_fps     = [d['std_fp']  for d in fp_rates]

    if n == 1:
        # Single-session: two side-by-side bars labelled directly
        fig, ax = plt.subplots(figsize=(5, 4))
        width = 0.4
        ax.bar(0, true_counts[0], width, label='True Cre+', color='coral')
        ax.bar(1, mean_fps[0], width, yerr=std_fps[0],
               label='Shuffled FP (mean ± std)', color='steelblue',
               capsize=4, error_kw={'linewidth': 1})
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['True Cre+', 'Shuffled FP'], fontsize=10)
        ax.set_ylabel('Unit count')
    else:
        x = np.arange(n)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.7), 4))
        width = 0.35
        ax.bar(x - width / 2, true_counts, width, label='True Cre+', color='coral')
        ax.bar(x + width / 2, mean_fps, width, yerr=std_fps,
               label='Shuffled FP (mean ± std)', color='steelblue',
               capsize=3, error_kw={'linewidth': 1})
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x], fontsize=8)
        ax.set_xlabel('Session index')
        ax.set_ylabel('Unit count')
    title = 'True Cre+ vs Shuffled False Positive Estimate'
    if genotype:
        title = f'{genotype} — {title}'
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# Region comparison PSTH

def plot_region_comparison(psth_opto, psth_non_opto,
                           meta_opto, meta_non_opto, time_bins,
                           top_n=10):
    """
    Side-by-side PSTH plots comparing the top_n regions (by total unit count)
    within optotagged and non-optotagged populations.

    Returns the matplotlib Figure.
    """
    import pandas as pd
    all_meta = pd.concat([meta_opto, meta_non_opto])
    top_regions = (
        all_meta['structure_acronym'].value_counts()
        .head(top_n)
        .index.tolist()
    )
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, (psth, meta, group_label) in zip(axes, [
        (psth_opto,     meta_opto,     'Optotagged'),
        (psth_non_opto, meta_non_opto, 'Non-optotagged'),
    ]):
        for region, color in zip(top_regions, colors):
            mask = (meta['structure_acronym'] == region).values
            if mask.sum() == 0:
                continue
            mean = psth[mask].mean(axis=0)
            sem  = psth[mask].std(axis=0) / np.sqrt(mask.sum())
            ax.plot(time_bins, mean, color=color, lw=1.5,
                    label=f'{region} (n={mask.sum()})')
            ax.fill_between(time_bins, mean - sem, mean + sem,
                            alpha=0.3, color=color)
        ax.axvline(0, color='k', lw=1, ls='--')
        ax.set_xlabel('Time from change (s)')
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_title(f'{group_label} — top {top_n} regions')
        ax.legend(fontsize=7)

    fig.tight_layout()
    return fig
