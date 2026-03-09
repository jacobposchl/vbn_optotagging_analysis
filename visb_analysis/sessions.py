"""
Handles session cache and ids from allensdk
"""

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
from pathlib import Path


class SessionHandler():
    def __init__(self, cache_path : Path):
        
        self.cache_path = cache_path
        
        self.cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_path)
        self.cache.load_latest_manifest()

        self.session_table = self.cache.get_ecephys_session_table()

    def return_all_sessions_list(self):
        self.session_table = self.cache.get_ecephys_session_table()
        session_ids = self.session_table.index.tolist()
        return session_ids

    