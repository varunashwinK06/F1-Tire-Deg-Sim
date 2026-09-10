import numpy as np
import pandas as pd
from F1DataExtractor import F1DataExtractor
from RacePaceFormulator import RacePaceFormulator
from MonteCarlo import MonteCarloSimulator

Year = 2023
Driver= "PIA"
Weekend = "Dutch Grand Prix"
Session= "R"

extractor= F1DataExtractor()
extractor.load_session(Year, Weekend, Session)
driver_df=extractor.get_driver_session(Driver)

if driver_df.empty:
    raise ValueError(f"No lap data found for driver {Driver} in session {Session} at {Weekend} {Year}")

formulator = RacePaceFormulator(driver_df[0:15])
fitted_df=formulator.calculate_fitted_deg_rate()

if fitted_df.empty:
    raise ValueError(f"No valid stints found for driver {Driver} in session {Session} at {Weekend} {Year}")

stint_row = fitted_df.loc[fitted_df['Stint'].map(lambda s: formulator.result_df.loc[s, 'total_laps']).idxmax()]
stint_id = stint_row['Stint']
n_laps  = 20
params = {
    'base_time': stint_row['base_time'],
    'deg_rate': stint_row['deg_rate'],
    'deg_curvature': stint_row['deg_curvature'],
    'fuel_loss': stint_row['fuel_loss'],
    'pcov': stint_row['pcov'],
    'sigma': stint_row['sigma']
}

mc=MonteCarloSimulator(seed=42)
sim_size = 1000
simulated_laptimes = mc.simulate_stint_from_covariance(params, n_laps, sim_size)

lower_bound, upper_bound = np.percentile(simulated_laptimes, [2.5, 97.5], axis=0)
confidence_range=upper_bound - lower_bound
print(*confidence_range, sep=", ")
