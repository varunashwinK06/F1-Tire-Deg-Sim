import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

formulator = RacePaceFormulator(driver_df)
fitted_df=formulator.calculate_fitted_deg_rate()

if fitted_df.empty:
    raise ValueError(f"No valid stints found for driver {Driver} in session {Session} at {Weekend} {Year}")

stint_row = fitted_df.loc[fitted_df['Stint'].map(lambda s: formulator.result_df.loc[s, 'total_laps']).idxmax()]
stint_id = stint_row['Stint']
n_laps  = int(formulator.result_df.loc[stint_id, 'total_laps'])
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
laps_axis = np.arange(1, n_laps + 1)

real_laps = formulator.raw_data[formulator.raw_data['Stint'] == stint_id]
real_x = real_laps['LapNumber'].to_numpy() - real_laps['LapNumber'].min() + 1
real_y = real_laps['LapTime_Seconds'].to_numpy()
 
fig, ax = plt.subplots(figsize=(9, 5.5))
 

ax.plot(laps_axis, simulated_laptimes, color='steelblue', alpha=0.05, linewidth=1)
ax.plot([], [], color='steelblue', alpha=0.6, linewidth=1, label=f'Simulated trials (n={sim_size})')
 
ax.scatter(real_x, real_y, color='black', marker='x', s=40, zorder=5, label='Real laps')
 
ax.set_xlabel('Lap number')
ax.set_ylabel('Laptime (s)')
ax.set_title(f'{Driver} — {Year} {Weekend} {Session}, Stint {stint_id} ({stint_row["compound"]})\nsimulated spread vs. real laps')
ax.legend()
plt.tight_layout()
plt.savefig('sim_vs_real.png', dpi=150)
plt.show()




