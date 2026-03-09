"""

"""

import numpy as np

# Single-unit psth
def make_psth(spikes , start_times,  time_before, duration, bin_size):
    """
    Returns (rates , time_bins) for a single unit
    """
    # Create time bins relative to stimulus change onset
    bins = np.arange(-time_before, duration - time_before + bin_size, bin_size)
    counts = np.zeros(len(bins) - 1)

    # For each event at time t, grab the spikes that fall within that window
    for t in start_times:
        window_spikes = spikes[
            (spikes >= t - time_before) &
            (spikes < t + duration - time_before)
        ]
        counts += np.histogram(window_spikes - t, bins = bins)[0]

    # Average spikes per bin -> divide by bin size -> Hz
    rates = (counts / len(start_times)) / bin_size
    return rates , bins[ : -1]

# Population level psth
def make_population_psth(unit_collection, start_times, time_before, duration, bin_size):
    """
    Population level of make_psth function
    
    """
    bins = np.arange(-time_before, duration - time_before + bin_size, bin_size)
    n_bins = len(bins) - 1
    n_trials = len(start_times)
    unit_ids = list(unit_collection.units.index)
    
    psth_array = np.zeros((len(unit_ids), n_bins, n_trials))

    for i, uid in enumerate(unit_ids):
        spikes = unit_collection.spike_times.get(uid)
        if spikes is None:
            continue
        for j, t in enumerate(start_times):
            window_spikes = spikes[(spikes >= t - time_before) & (spikes < t + duration - time_before)]
            psth_array[i, :, j] = np.histogram(window_spikes - t, bins=bins)[0] / bin_size

    return psth_array, bins[:-1], unit_ids
