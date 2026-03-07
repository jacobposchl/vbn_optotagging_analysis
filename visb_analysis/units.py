"""
Handles filtering for units from sessions
"""

class UnitCollection:
    def __init__(self, session):
        units = session.get_units()
        channels = session.get_channels()
        self.units = units.merge(
            channels[["structure_acronym"]],
            left_on="peak_channel_id",
            right_index=True
        )
        self.spike_times = session.spike_times

    def filter_quality(self, unit_filters : dict):
        mask = (
            (self.units["snr"] >= unit_filters["snr_min"]) &
            (self.units["isi_violations"] <= unit_filters["isi_violations_max"]) &
            (self.units["firing_rate"] >= unit_filters["firing_rate_min"]) &
            (self.units["quality"] == unit_filters["quality"])
        )

        self.units = self.units[mask]

        return self
    
    def filter_region(self, regions : list[str]):
        pattern = '|'.join(regions)
        self.units = self.units[
            self.units["structure_acronym"].str.contains(pattern, regex=True)
        ]

        return self
    
    def get_spike_times(self, unit_id = None):
        if unit_id is not None:
            return self.spike_times[unit_id]
        return {uid: self.spike_times[uid] for uid in self.units.index if uid in self.spike_times}
    
    def __len__(self):
        return len(self.units)