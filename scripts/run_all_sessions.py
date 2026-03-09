'''
Loads configuration yaml file and batch processes all sessions.

Saves results in results/ folder
'''

from pathlib import Path
import yaml
from argparse import ArgumentParser
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")

from visb_analysis import SessionHandler, UnitCollection
from visb_analysis import add_optotagging_labels, make_population_psth

def main(config_path: Path):
    
    # Open config file
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    if config:
        print(f"Loaded config at: {config_path} safely")
    else:
        return ValueError("Unable to load config")

    cache_dir = Path(config["cache_dir"])

    # Filter Sessions by Config
    session_handler = SessionHandler(cache_dir)
    all_sessions_ids = session_handler.return_all_sessions_list()
    print(f"Sessions before filtering: {len(all_sessions_ids)}")

    session_filters = config.get("sessions")
    session_table = session_handler.session_table
    filtered_sessions = session_table[
        session_table["genotype"].str.contains(session_filters["genotype"])
    ]
    print(f"Filters: {session_filters}")
    filtered_session_ids = filtered_sessions.index.tolist()
    print(f"Sessions after filtering: {len(filtered_session_ids)}")

    # Process each session
    results_dir = Path(config["results_dir"]) / config_path.stem
    results_dir.mkdir(parents = True, exist_ok = True)
    
    print(f"Processing sessions...  ")
    
    unit_filters = config["unit_filters"]
    print(f"Unit Filter Settings: {unit_filters}")

    optotagging = config["optotagging"]
    print(f"Optotagging Settings: {optotagging}")

    psth_settings = config["psth"]
    print(f"psth Settings: {psth_settings}")

    psth_visual_settings = config["psth_visual"]
    print(f"psth_visual Settings: {psth_visual_settings}")

    for session_id in filtered_session_ids:
        session = session_handler.cache.get_ecephys_session(session_id)

        # Filter Units
        # Retain all regions for analysis
        units = UnitCollection(session = session).filter_quality(unit_filters)
        add_optotagging_labels(units, session, optotagging)

        print(f"    -> Session: {session_id} has {len(units)} optotagged units")

        # Opto PSTH — aligned to laser pulses (short, max-power)
        opto_table = session.optotagging_table
        pulse_times = opto_table[
            (opto_table["duration"] <= optotagging["max_pulse_duration"]) &
            (opto_table["level"] == opto_table["level"].max())
        ]["start_time"].values

        psth_opto, time_bins_opto, unit_ids = make_population_psth(
            units, pulse_times, mean_over_trials=True, **psth_settings
        )

        # Visual PSTH — aligned to image change onsets
        stim = session.stimulus_presentations
        change_times = stim[stim["is_change"] == True]["start_time"].values

        psth_visual, time_bins_visual, _ = make_population_psth(
            units, change_times, mean_over_trials=True, **psth_visual_settings
        )

        # Save Results
        np.savez(
            results_dir / f"{session_id}.npz",
            psth_opto        = psth_opto,
            time_bins_opto   = time_bins_opto,
            psth_visual      = psth_visual,
            time_bins_visual = time_bins_visual,
            unit_ids         = unit_ids,
        )

        # Unit metadata
        units.units.to_csv(results_dir / f"{session_id}_units.csv")

        # Explicitly release NWB session data before loading the next session
        del psth_opto, psth_visual, units, session
        gc.collect()

    print(f"Processed all filtered sessions")
    print(f"Results saved under: {results_dir}")

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("config_path")
    args = argparser.parse_args()
    config_path = args.config_path
    main(Path(config_path))