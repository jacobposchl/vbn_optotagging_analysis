"""
Performs calculation of light-driven analysis from unit spike times & light times
"""

import numpy as np


# Count spikes relative to each pulse onset and convert to Hz
def _mean_rate_in_window(spikes, pulse_times, window):
    window_duration = window[1] - window[0]
    total_spikes = sum(
        ((spikes >= t + window[0]) & (spikes < t + window[1])).sum()
        for t in pulse_times
    )
    return (total_spikes / len(pulse_times)) / window_duration


def _get_pulse_times(session, opto_config):
    """
    Extract short, max-power laser pulse onset times from a session.

    Returns
    -------
    pulse_times : np.ndarray of float, shape (n_pulses,)
    """
    opto_table = session.optotagging_table
    pulses = opto_table[
        (opto_table["duration"] <= opto_config["max_pulse_duration"]) &
        (opto_table["level"] == opto_table["level"].max())
    ]
    return pulses["start_time"].values


# Return per-unit baseline and evoked rates without modifying the collection
def get_opto_rates(unit_collection, session, opto_config):
    """
    Compute per-unit baseline and evoked firing rates for the short opto pulses.

    Returns
    -------
    baseline_rates : np.ndarray, shape (n_units,)
    evoked_rates   : np.ndarray, shape (n_units,)
    unit_ids       : list
    """
    pulse_times = _get_pulse_times(session, opto_config)
    baseline = opto_config["baseline_window"]
    evoked   = opto_config["evoked_window"]

    unit_ids       = list(unit_collection.units.index)
    baseline_rates = np.zeros(len(unit_ids))
    evoked_rates   = np.zeros(len(unit_ids))

    for i, uid in enumerate(unit_ids):
        spikes = unit_collection.spike_times.get(uid)
        if spikes is None:
            continue
        baseline_rates[i] = _mean_rate_in_window(spikes, pulse_times, baseline)
        evoked_rates[i]   = _mean_rate_in_window(spikes, pulse_times, evoked)

    return baseline_rates, evoked_rates, unit_ids


# Add optotagging labels to unit collection
def add_optotagging_labels(unit_collection, session, opto_config):
    """
    Adds an 'optotagged' bool column to the unit_collection.units in-place.
    """
    pulse_times = _get_pulse_times(session, opto_config)
    baseline = opto_config["baseline_window"]
    evoked   = opto_config["evoked_window"]

    labels = {}
    for unit_id, spikes in unit_collection.get_spike_times().items():
        baseline_rate = _mean_rate_in_window(spikes, pulse_times, baseline)
        evoked_rate   = _mean_rate_in_window(spikes, pulse_times, evoked)
        fold = evoked_rate / (baseline_rate + 1)
        labels[unit_id] = (
            fold > opto_config["min_fold_increase"] and
            evoked_rate > opto_config["min_evoked_rate"]
        )

    unit_collection.units["optotagged"] = unit_collection.units.index.map(labels)
    return unit_collection


def compute_trial_reliability(unit_collection, pulse_times, evoked_window):
    """
    Fraction of opto trials on which each unit fires >= 1 spike in the
    evoked window.  Computed directly from raw spike times.

    Parameters
    ----------
    unit_collection : UnitCollection
    pulse_times     : np.ndarray, shape (n_pulses,) — laser onset times (s)
    evoked_window   : list/tuple [t_start, t_end] in seconds relative to pulse

    Returns
    -------
    reliability : np.ndarray, shape (n_units,) — values in [0, 1]
    unit_ids    : list
    """
    t0, t1 = evoked_window[0], evoked_window[1]
    n_pulses = len(pulse_times)
    unit_ids = list(unit_collection.units.index)
    reliability = np.zeros(len(unit_ids))

    for i, uid in enumerate(unit_ids):
        spikes = unit_collection.spike_times.get(uid)
        if spikes is None or len(spikes) == 0:
            continue
        trials_with_spike = sum(
            int(((spikes >= t + t0) & (spikes < t + t1)).any())
            for t in pulse_times
        )
        reliability[i] = trials_with_spike / n_pulses

    return reliability, unit_ids


def compute_spike_latency(unit_collection, pulse_times, evoked_window):
    """
    Mean first-spike latency (ms) per unit across opto trials.

    Only trials where at least one spike exists in the evoked window
    contribute to the mean.  Units with zero spikes across all trials get NaN.

    Parameters
    ----------
    unit_collection : UnitCollection
    pulse_times     : np.ndarray, shape (n_pulses,) — laser onset times (s)
    evoked_window   : list/tuple [t_start, t_end] in seconds relative to pulse

    Returns
    -------
    mean_latency_ms : np.ndarray, shape (n_units,) — NaN where no spikes
    unit_ids        : list
    """
    t0, t1 = evoked_window[0], evoked_window[1]
    unit_ids = list(unit_collection.units.index)
    mean_latency_ms = np.full(len(unit_ids), np.nan)

    for i, uid in enumerate(unit_ids):
        spikes = unit_collection.spike_times.get(uid)
        if spikes is None or len(spikes) == 0:
            continue
        first_latencies = []
        for t in pulse_times:
            in_window = spikes[(spikes >= t + t0) & (spikes < t + t1)]
            if len(in_window) > 0:
                first_latencies.append((in_window.min() - t) * 1000.0)  # ms
        if first_latencies:
            mean_latency_ms[i] = np.mean(first_latencies)

    return mean_latency_ms, unit_ids


def compute_shuffled_fp_rate(
    spike_times_dict,
    unit_ids,
    pulse_times,
    opto_config,
    increase_in_FR,
    min_evoked_rate,
    n_shuffles=20,
    shift_s=5.0,
    max_time=None,
    exclusion_s=0.1,
    min_pulses=10,
):
    """
    Estimate false-positive classification rate using time-shifted pulse trains.

    For each shuffle k (1..n_shuffles), shifts all pulse times by k * shift_s
    seconds and re-classifies units.  Units passing threshold on shifted pulses
    are counted as false positives.

    Shifted pulses that fall within exclusion_s of any real pulse are removed
    before classification to avoid contaminating the null distribution with
    genuine light-evoked responses.  Shuffles with fewer than min_pulses
    remaining are skipped (appends 0).

    Uses evoked / (baseline + 1) > increase_in_FR to match the notebook's
    Section 4 classification convention.

    Parameters
    ----------
    spike_times_dict : dict {unit_id -> np.ndarray} — raw spike times (s)
    unit_ids         : list — unit IDs matching spike_times_dict
    pulse_times      : np.ndarray — original laser onset times (s)
    opto_config      : dict with 'baseline_window' and 'evoked_window'
    increase_in_FR   : float — fold-change threshold
    min_evoked_rate  : float — minimum evoked Hz threshold
    n_shuffles       : int   — number of time shifts (default 20)
    shift_s          : float — shift step in seconds (default 5.0)
    max_time         : float or None — drop shifted pulses beyond this time
    exclusion_s      : float — shifted pulses within this many seconds of any
                               real pulse are dropped (default 0.1 s = 100 ms)
    min_pulses       : int   — minimum remaining shifted pulses required to
                               run classification; shuffles below this are
                               skipped (default 10)

    Returns
    -------
    fp_counts : list of int, length n_shuffles
    n_units   : int — total number of units
    """
    baseline = opto_config["baseline_window"]
    evoked   = opto_config["evoked_window"]
    n_units  = len(unit_ids)
    fp_counts = []

    for k in range(1, n_shuffles + 1):
        shifted = pulse_times + k * shift_s
        if max_time is not None:
            shifted = shifted[shifted < max_time]

        # Drop shifted pulses that overlap with real pulse times so genuine
        # light-evoked responses don't contaminate the null distribution.
        keep = np.ones(len(shifted), dtype=bool)
        for real_t in pulse_times:
            keep &= np.abs(shifted - real_t) > exclusion_s
        shifted = shifted[keep]

        if len(shifted) < min_pulses:
            fp_counts.append(0)
            continue

        n_pos = 0
        for uid in unit_ids:
            spikes = spike_times_dict.get(uid)
            if spikes is None:
                continue
            b_rate = _mean_rate_in_window(spikes, shifted, baseline)
            e_rate = _mean_rate_in_window(spikes, shifted, evoked)
            if e_rate > min_evoked_rate and (e_rate / (b_rate + 1)) > increase_in_FR:
                n_pos += 1
        fp_counts.append(n_pos)

    return fp_counts, n_units
