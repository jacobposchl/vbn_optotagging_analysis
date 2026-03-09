from .sessions import SessionHandler
from .units import UnitCollection
from .optotagging import (
    add_optotagging_labels,
    get_opto_rates,
    compute_trial_reliability,
    compute_spike_latency,
    compute_shuffled_fp_rate,
)
from .psth import make_population_psth
from .plots import (
    plot_opto_scatter,
    plot_opto_heatmaps,
    plot_units_per_session,
    plot_population_psth,
    plot_psth_heatmaps,
    plot_region_distribution,
    plot_multi_session_scatter,
    plot_multi_session_counts,
    plot_reliability_histogram,
    plot_latency_histogram,
    plot_fp_summary,
    plot_cre_fraction_by_region,
    plot_labeled_cells_by_region,
    plot_cre_density_by_region,
)