import F1DataExtractor 
import RacePaceFormulator 
import pandas as pd
import numpy as np
import matplotlib as plt

class MonteCarloSimulator:
    def __init__(self, noise_range: float = 0.5, seed: int = 42):
        self.noise_range=noise_range
        self.rng=np.random.default_rng(seed)
    def generate_gaussian_noise(self, n_laps: int, sim_size:int) -> np.ndarray:
        return self.rng.normal(loc=0, scale=self.noise_range, size=(sim_size, n_laps))
    def build_base_model(self, model_type: str, params: dict, n_laps: int) -> np.ndarray:
        laps = np.arange(1, n_laps + 1, dtype=float)
        if model_type == 'naive':
            base_time = params['base_time']
            deg_rate = params['thermal_deg_rate']
            fuel_loss = params['fuel_loss']
            k = params['multipliers']
            exponential_wear = k ** laps - k
            return base_time + deg_rate * (laps - 1) - fuel_loss * (laps - 1) + exponential_wear
        if model_type == 'fitted':
            base_time = params['base_time']
            deg_rate = params['deg_rate']
            deg_curvature = params['deg_curvature']
            fuel_loss = params['fuel_loss']
            return base_time + deg_rate * (laps - 1) + deg_curvature * (laps - 1) ** 2 - fuel_loss * (laps - 1)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'naive' or 'fitted'.")
    def simulate_stint(self, model_type: str, params: dict, n_laps: int, sim_size: int) -> np.ndarray:
        base_signal = self._build_base_model(model_type, params, n_laps)
        noise_matrix = self.generate_gaussian_noise(n_laps, sim_size)
        return base_signal[:, None] + noise_matrix
    
        
        
        
    
