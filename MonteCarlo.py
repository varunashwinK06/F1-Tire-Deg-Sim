import pandas as pd
import numpy as np


class MonteCarloSimulator:
    def __init__(self, noise_range: float = 0.5, seed: int = 42):
        self.noise_range=noise_range
        self.rng=np.random.default_rng(seed)
    def generate_gaussian_noise(self, n_laps: int, sim_size:int, scale: float) -> np.ndarray:
        noise_scale = scale if scale is not None else self.noise_range
        return self.rng.normal(loc=0, scale=noise_scale, size=(sim_size, n_laps))
    def simulate_stint_from_covariance(self, params: dict, n_laps: int, sim_size: int) -> np.ndarray:
        pcov = params['pcov']  
        mean_coeffs = np.array([params['base_time'], params['deg_rate'], params['deg_curvature']])
        fuel_loss = params['fuel_loss']
        sigma=params['sigma']

        try:
            L = np.linalg.cholesky(pcov)
        except np.linalg.LinAlgError:
            L = np.zeros_like(pcov)

        z = self.rng.standard_normal(size=(3, sim_size))          # (3, sim_size)
        sampled_coeffs = mean_coeffs[:, None] + L @ z              # (3, sim_size)

        laps = np.arange(1, n_laps + 1, dtype=float) - 1           # laps-1, consistent with build_base_model
        base_t = sampled_coeffs[0, :]
        deg_r = sampled_coeffs[1, :]
        deg_c = sampled_coeffs[2, :]

        simulated = (
            base_t[None, :]
            + deg_r[None, :] * laps[:, None]
            + deg_c[None, :] * (laps[:, None] ** 2)
            - fuel_loss * laps[:, None]
        )

        gaussian_noise = self.generate_gaussian_noise(sim_size, n_laps, scale=sigma)
        return simulated+gaussian_noise
    
    
        
        
        
    
