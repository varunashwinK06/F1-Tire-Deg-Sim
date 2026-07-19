import pandas as pd
import numpy as np

class RacePaceFormulator:
    def __init__(self, result_df:pd.DataFrame):
        self.result_df=result_df.groupby('Stint').agg(
            compound=('Compound', 'first'),
            first_lap_time=('LapTime_Seconds', 'first'),
            last_lap_time=('LapTime_Seconds', 'last'),
            total_laps=('LapTime_Seconds', 'count')
        )
    def calculate_deg_rate(self) -> pd.DataFrame:
        real_stints=self.result_df[self.result_df['total_laps']>1].copy()
        if real_stints.empty:
            print("No stints with more than one lap found.")
            return pd.DataFrame()
        fuel_loss=0.035
        material_deg_multiplier = {'SOFT': 1.06, 'MEDIUM': 1.04, 'HARD': 1.02}
        
        raw_net_delta=real_stints['last_lap_time']-real_stints['first_lap_time']

        fuel_burn_factor= (real_stints['total_laps']-1)*fuel_loss
        fuel_adjusted_delta=raw_net_delta+fuel_burn_factor

        real_stints['multipliers'] = real_stints['compound'].str.upper().map(material_deg_multiplier).fillna(1.02)
        n=real_stints['total_laps']
        k=real_stints['multipliers']

        exponential_wear= k**n - k

        thermal_deg_delta= fuel_adjusted_delta-exponential_wear
        real_stints['thermal_deg_rate'] = thermal_deg_delta / (real_stints['total_laps'] - 1)
        real_stints['fuel_loss'] = 0.035

        summary_columns = ['compound', 'multipliers', 'thermal_deg_rate', 'fuel_loss']
        return real_stints[summary_columns].reset_index()









        
        
        
        
    
    


