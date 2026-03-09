# Optotagging Analysis Repo for AllenSDK Visual Behavior Neuropixels Dataset

This is a python package for classifying neuronal subclasses from the Allen Institute's electrophysiological data. 

Note: Much of the code used within this repository is from the AllenSDK Package by the Allen Institute for Brain Science. This package includes various notebook tutorials for in-depth examples of how to classify neuronal subclasses (e.g., SST, VIP). This repo expands on these tutorials in a few areas, primarily for robust tagging.

## Modularization:
While focused on the Visual Behavior Neuropixels Dataset, this repo could be expanded and refined to allow for future datasets, or integration of the Visual Coding dataset.

This repo is split into four main sections:
1. `visb_analysis/`: The main python package containing helper Classes, functions, plots, etc.
2. `notebooks/`: A folder containing ipynb files for running the main analyses
3. `scripts/`: A folder containing scripts to run before notebook analyses
4. `configs/`: A folder containing the configuration settings for optotagging criteria parameters and essential filters. This is where your polished parameters will go.

## How to run:
1. You will need to download the data from the AllenSDK package. There are more than one ways to do this, in this repo, we will follow the local installation procedure. If you wish to download the full dataset, be sure to have more than 500GB worth of storage for this. You can download the dataset from the `scripts/download_all_sessions.py` script.
2. Once downloaded, the `notebooks/parameter_tuning.ipynb` notebook will help you refine your parameters until you find a promising set of configurations. 
3. You can run the `scripts/param_sweep.py` script to verify a parameter selection. Note that this is still for a single session, it is recommended to re-run the `notebooks/parameter_tuning.ipynb` until you verify your selection is robust. 
4. Next, replace the values in the `configs/` yaml files, as to your parameters. You can now run `scripts/run_all_sessions.py` to process all of the sessions within the Visual Behavior Neuropixels dataset, on your chosen parameters. Note that this is for the `experience_level = 'Novel'` sessions. Future updates will allow for running on all sessions.
5. Finally, the `notebooks/all_session_analysis.ipynb` will output relevant results of the tagged neuron subclasses given your chosen parameters. 