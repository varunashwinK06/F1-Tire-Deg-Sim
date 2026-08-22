import unittest
import numpy as np
import pandas as pd

from RacePaceFormulator import RacePaceFormulator
from MonteCarlo import MonteCarloSimulator

def make_synthetic_race_df(n_stints: int = 3, laps_per_stint: int = 15, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    rows = []
    for stint_id in range(1, n_stints + 1):
        compound = compounds[(stint_id - 1) % len(compounds)]
        base_time = 90.0
        deg_rate = 0.05
        deg_curvature = 0.002
        for lap_num in range(1, laps_per_stint + 1):
            lap_in_stint = lap_num - 1
            true_time = base_time + deg_rate * lap_in_stint + deg_curvature * lap_in_stint ** 2
            noisy_time = true_time + rng.normal(0, 0.1)
            rows.append({
                'LapNumber': lap_num,
                'Compound': compound,
                'LapTime_Seconds': noisy_time,
                'Stint': stint_id,
            })
    return pd.DataFrame(rows)

class TestRacePaceFormulatorFitted(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_race_df()
        self.formulator = RacePaceFormulator(self.df)
        self.result = self.formulator.calculate_fitted_deg_rate()

    def test_returns_nonempty_dataframe(self):
        self.assertFalse(self.result.empty)

    def test_expected_columns_present(self):
        expected_cols = {'Stint', 'compound', 'base_time', 'deg_rate', 'deg_curvature', 'fuel_loss', 'pcov', 'sigma'}
        self.assertTrue(expected_cols.issubset(set(self.result.columns)))

    def test_fitted_params_close_to_known_generating_values(self):
        row = self.result.iloc[0]
        self.assertAlmostEqual(row['base_time'], 90.0, delta=0.2)
        self.assertAlmostEqual(row['deg_rate'], 0.05, delta=0.05)
        self.assertAlmostEqual(row['deg_curvature'], 0.002, delta=0.01)

    def test_pcov_shape_is_3x3(self):
        row = self.result.iloc[0]
        self.assertEqual(row['pcov'].shape, (3, 3))

    def test_pcov_is_symmetric(self):
        row = self.result.iloc[0]
        np.testing.assert_allclose(row['pcov'], row['pcov'].T, atol=1e-8)

    def test_sigma_is_nonnegative(self):
        for sigma in self.result['sigma']:
            self.assertGreaterEqual(sigma, 0.0)

    def test_short_stint_excluded_below_min_laps(self):
        df = self.df.copy()
        df = df[~((df['Stint'] == 1) & (df['LapNumber'] > 2))]  # stint 1 now has 2 laps
        result = RacePaceFormulator(df).calculate_fitted_deg_rate()
        self.assertNotIn(1, result['Stint'].values)

    def test_exactly_determined_stint_has_zero_pcov(self):
        # exactly 3 laps -> 3 obs, 3 params -> dof=0 -> pcov should be all zeros
        df = self.df.copy()
        df = df[~((df['Stint'] == 1) & (df['LapNumber'] > 3))]
        result = RacePaceFormulator(df).calculate_fitted_deg_rate()
        row = result[result['Stint'] == 1].iloc[0]
        np.testing.assert_allclose(row['pcov'], np.zeros((3, 3)))
        self.assertEqual(row['sigma'], 0.0)


class TestMonteCarloSimulator(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_race_df()
        self.formulator = RacePaceFormulator(self.df)
        self.fitted_df = self.formulator.calculate_fitted_deg_rate()
        self.naive_df = self.formulator.calculate_naive_deg_rate()
        self.n_laps = 15
        self.sim_size = 500

    def _fitted_params(self, stint_id=1):
        row = self.fitted_df[self.fitted_df['Stint'] == stint_id].iloc[0]
        return {
            'base_time': row['base_time'],
            'deg_rate': row['deg_rate'],
            'deg_curvature': row['deg_curvature'],
            'fuel_loss': row['fuel_loss'],
            'pcov': row['pcov'],
            'sigma': row['sigma'],
        }

    def test_generate_gaussian_noise_shape(self):
        mc = MonteCarloSimulator(seed=1)
        noise = mc.generate_gaussian_noise(self.sim_size, self.n_laps, scale=0.05)
        self.assertEqual(noise.shape, (self.n_laps, self.sim_size))

    def test_generate_gaussian_noise_explicit_scale_overrides_noise_range(self):
        mc = MonteCarloSimulator(noise_range=0.2, seed=1)
        noise = mc.generate_gaussian_noise(self.n_laps, 20000, scale=0.05)
        self.assertAlmostEqual(noise.std(), 0.05, delta=0.005)


    def test_simulate_stint_from_covariance_shape(self):
        mc = MonteCarloSimulator(seed=1)
        sims = mc.simulate_stint_from_covariance(self._fitted_params(), self.n_laps, self.sim_size)
        self.assertEqual(sims.shape, (self.n_laps, self.sim_size))

    def test_seeded_reproducibility(self):
        mc1 = MonteCarloSimulator(seed=99)
        mc2 = MonteCarloSimulator(seed=99)
        sims1 = mc1.simulate_stint_from_covariance(self._fitted_params(), self.n_laps, self.sim_size)
        sims2 = mc2.simulate_stint_from_covariance(self._fitted_params(), self.n_laps, self.sim_size)
        np.testing.assert_array_equal(sims1, sims2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
