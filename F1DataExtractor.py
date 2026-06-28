import pandas as pd 
import fastf1  
import numpy as np
import os

class F1DataExtractor:
    def __init__(self, cache_dir: str="f1_cache"):
        self.cache_dir = cache_dir 
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        fastf1.Cache.set_cache_dir(self.cache_dir)
        self.session = None
    def load_session(self, year: int, weekend: str,session: str) -> bool:
        self.session = fastf1.get_session(year, weekend, session)
        self.session.load(weather=True, telemetry=False, laps=True, messages=False)
        return True
    def get_driver_session(self, driver: str) -> pd.DataFrame:
        if not self.session:
            raise ValueError("No session available")
        laps_df = self.session.laps.pick_driver(driver).copy()
        laps_df = laps_df.pick_quicklaps()
        if laps_df.empty:
            print(f"No lap data found for driver: {driver}")
            return pd.DataFrame()
        laps_df['LapTime_Seconds'] = laps_df['LapTime'].dt.total_seconds()
    

        target_columns = ["LapNumber", "Stint", "Compound", "TyreLife", "LapTime_Seconds"]
        
        result_df = laps_df[target_columns].astype({
            'LapNumber': 'int',
            'Stint': 'int',
            'Compound': 'str',
            'TyreLife': 'int',
            'LapTime_Seconds': 'float'
        })
        return result_df 
    


