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
        self.raw_data=result_df.sort_values(['Stint', 'LapNumber']).copy()
    def calculate_naive_deg_rate(self) -> pd.DataFrame:
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
        real_stints['base_time'] = real_stints['first_lap_time']

        summary_columns = ['compound', 'multipliers', 'thermal_deg_rate', 'fuel_loss', 'base_time']
        return real_stints[summary_columns].reset_index()

    def calculate_fitted_deg_rate(self, fuel_loss: float = 0.035) -> pd.DataFrame:
        real_stints=self.result_df[self.result_df['total_laps']>2].copy()
        if real_stints.empty:
            print("No stints with more than one lap found.")
            return pd.DataFrame()
        base_times, deg_rates, deg_curvatures = [], [], []

        for stint_id in real_stints.index:
            stint_laps = self.raw_data[self.raw_data['Stint'] == stint_id]
            laps = stint_laps['LapNumber'].to_numpy(dtype=float)
            laptimes = stint_laps['LapTime_Seconds'].to_numpy(dtype=float)
            fuel_corrected = laptimes + fuel_loss * laps

           
            X = np.column_stack([np.ones_like(laps), laps, laps ** 2])
            coeffs, *_ = np.linalg.lstsq(X, fuel_corrected, rcond=None)

            base_times.append(coeffs[0])
            deg_rates.append(coeffs[1])
            deg_curvatures.append(coeffs[2])

        real_stints['base_time'] = base_times
        real_stints['deg_rate'] = deg_rates
        real_stints['deg_curvature'] = deg_curvatures
        real_stints['fuel_loss'] = fuel_loss

        summary_columns = ['compound', 'base_time', 'deg_rate', 'deg_curvature', 'fuel_loss']
        return real_stints[summary_columns].reset_index()
        
        









        
        
        
        
    
    


