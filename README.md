A Python based simulator that uses real F1 lap time and stint data to extract degradation rate of tyre compounds with Pandas. 
Uses a polynomial model with Gaussian noise to account for thermal degradation and material degradation as well as random events.
Monte Carlo methods are used to then simulate multiple runnings of the same race to see the variance of possibilities.
