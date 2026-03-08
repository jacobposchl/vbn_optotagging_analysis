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
    Two-panel chart per session: total units (top) and Cre+ count (bottom).
    Splitting axes prevents tiny Cre+ counts from being invisible on a shared scale.

    all_units_meta: list of per-session unit metadata DataFrames.
    Returns the matplotlib Figure.
    """
    opto_counts = [int(meta['optotagged'].fillna(False).sum()) for meta in all_units_meta]
    total_counts = [len(meta) for meta in all_units_meta]
    x = np.arange(len(opto_counts))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(8, len(x) * 0.7), 5),
                                          sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Top: total units per session
    ax_top.bar(x, total_counts, color='steelblue', alpha=0.85)
    ax_top.set_ylabel('Total units')
    ax_top.set_title(title)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    # Bottom: Cre+ count per session, with count labels
    ax_bot.bar(x, opto_counts, color='coral', alpha=0.85)
    for xi, val in zip(x, opto_counts):
        ax_bot.text(xi, val + 0.1, str(val), ha='center', va='bottom', fontsize=8)
    ax_bot.set_ylabel('Cre+ units')
    ax_bot.set_xlabel('Session index')
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([str(i) for i in x], fontsize=8)
    ax_bot.spines['top'].set_visible(False)
    ax_bot.spines['right'].set_visible(False)

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


# Cross-genotype region distribution (Slide 1 — absolute counts)

def plot_labeled_cells_by_region(genotype_meta, min_labeled=3):
    """
    Side-by-side subplots of absolute labeled (Cre+) neuron counts per brain region,
    one panel per genotype, each with its own y-scale. Regions are sorted by total
    labeled cells (summed across genotypes) and only shown if they have >= min_labeled
    total across all genotypes.

    genotype_meta : dict mapping genotype name -> combined unit metadata DataFrame
                    (must have 'structure_acronym' and 'optotagged' columns)
    min_labeled   : minimum total labeled cells (across all genotypes) for a region
                    to be shown. Default 3.
    Returns the matplotlib Figure.
    """
    import pandas as pd

    genotypes = list(genotype_meta.keys())
    colors = ['#E07B54', '#5B8DB8', '#6DBF7E', '#B85B8D'][:len(genotypes)]

    region_data = {}
    for gt, meta in genotype_meta.items():
        opto = meta['optotagged'].fillna(False).astype(bool)
        counts = opto.groupby(meta['structure_acronym']).sum().rename(gt)
        region_data[gt] = counts

    df = pd.DataFrame(region_data).fillna(0).astype(int)
    df['total_labeled'] = df.sum(axis=1)
    df = df[df['total_labeled'] >= min_labeled].sort_values('total_labeled', ascending=False)
    df = df.drop(columns='total_labeled')

    n_regions = len(df)
    x = np.arange(n_regions)
    n_gt = len(genotypes)

    fig, axes = plt.subplots(n_gt, 1, figsize=(max(12, n_regions * 0.6), 3.5 * n_gt),
                             sharex=True)
    if n_gt == 1:
        axes = [axes]

    for ax, gt, color in zip(axes, genotypes, colors):
        vals = df[gt].values
        bars = ax.bar(x, vals, color=color, alpha=0.85, label=f'{gt}+')
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.2, str(int(val)),
                        ha='center', va='bottom', fontsize=8)
        ax.set_ylabel('Labeled neuron count')
        ax.set_title(f'{gt}+ neurons by brain region')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(df.index, rotation=45, ha='right', fontsize=9)
    axes[-1].set_xlabel('Brain region')
    fig.suptitle('Labeled neuron counts by brain region (Novel sessions)', fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


# Cross-genotype region density (Slide 2 — % of region Cre+)

def plot_cre_density_by_region(genotype_meta, min_units=50, min_labeled=3):
    """
    Side-by-side subplots of Cre+ density (% of region) per brain region, one panel
    per genotype with its own y-scale. Regions are shared across panels and sorted
    by the maximum density seen in any genotype.

    Only regions with >= min_units total units (in that genotype's sessions) AND
    >= min_labeled Cre+ cells in at least one genotype are shown.

    genotype_meta : dict mapping genotype name -> combined unit metadata DataFrame
    min_units     : minimum total units in a region per genotype to compute density
    min_labeled   : minimum Cre+ cells in at least one genotype to include the region
    Returns the matplotlib Figure.
    """
    import pandas as pd

    genotypes = list(genotype_meta.keys())
    colors = ['#E07B54', '#5B8DB8', '#6DBF7E', '#B85B8D'][:len(genotypes)]

    density_data = {}
    labeled_data = {}
    for gt, meta in genotype_meta.items():
        opto = meta['optotagged'].fillna(False).astype(bool)
        grp = pd.DataFrame({
            'cre_pos': opto.groupby(meta['structure_acronym']).sum(),
            'total':   opto.groupby(meta['structure_acronym']).count(),
        })
        grp['density'] = grp['cre_pos'] / grp['total'] * 100
        grp.loc[grp['total'] < min_units, 'density'] = np.nan
        density_data[gt] = grp['density']
        labeled_data[gt] = grp['cre_pos']

    density_df = pd.DataFrame(density_data)
    labeled_df = pd.DataFrame(labeled_data).fillna(0)

    has_enough_units = density_df.notna().any(axis=1)
    has_labeled      = (labeled_df >= min_labeled).any(axis=1)
    density_df = density_df[has_enough_units & has_labeled].copy()

    density_df['_max'] = density_df.max(axis=1)
    density_df = density_df.sort_values('_max', ascending=False).drop(columns='_max')

    n_regions = len(density_df)
    n_gt = len(genotypes)
    x = np.arange(n_regions)

    fig, axes = plt.subplots(n_gt, 1, figsize=(max(12, n_regions * 0.6), 3.5 * n_gt),
                             sharex=True)
    if n_gt == 1:
        axes = [axes]

    for ax, gt, color in zip(axes, genotypes, colors):
        raw_vals = density_df[gt].values
        plot_vals = np.where(np.isnan(raw_vals), 0, raw_vals)
        bars = ax.bar(x, plot_vals, color=color, alpha=0.85, label=f'{gt}+')
        for bar, val, raw in zip(bars, plot_vals, raw_vals):
            if not np.isnan(raw) and val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.05, f'{val:.1f}%',
                        ha='center', va='bottom', fontsize=8)
        ax.set_ylabel('% of region Cre+')
        ax.set_title(f'{gt}+ density within {gt}-line sessions')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(density_df.index, rotation=45, ha='right', fontsize=9)
    axes[-1].set_xlabel('Brain region')
    fig.suptitle(
        f'Labeled neuron density by region (Novel sessions, \u2265{min_units} units/region)',
        fontsize=12, y=1.01
    )
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
