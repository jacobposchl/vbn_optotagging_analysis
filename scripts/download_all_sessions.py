from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache

# May need to be changed
output_dir = "c:/Users/Jacob Poschl/le-jepa-snn/snn-jepa/visual_behavior_neuropixels_data"

output_dir = Path(output_dir)
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=output_dir)
cache.load_latest_manifest()

# Get all session IDs
session_table = cache.get_ecephys_session_table()
session_ids = session_table.index.tolist()

# Download all sessions (make sure you have around 500 GB)
for i, session_id in enumerate(session_ids):
    print(f"Downloading session {i+1}/{len(session_ids)}: {session_id}")
    try:
        session = cache.get_ecephys_session(ecephys_session_id=session_id)
    except Exception as e:
        print(f"  Failed: {e}")
        continue