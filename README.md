

A python program that uses FastF1 API to extract real race data, fitting stints to a statistical tire degradation model, and then simulates uncertainty of the model using Monte Carlo methods to estimate plausible race pace range.
## Class Structures


F1DataExtractor: Uses fastF1 to pull real data and caches data to reduce subsequent load times. Cleans data to remove VSC/SC/In/Out laps, and fixes typing for columns such as la- Rtime.

RacePaceFormulator: Fits race data to either an OLS fitted model, or a naive model which uses net delta over a - stint.

MonteCarloSimulator: Generates a Gaussian noise matrix and simulates a set number of stints using inputs from RacePaceFormul- ator. 

UnitTests: Unit tests that test basic functionality and edge cases such as minimum stints that can be passed into the model (n_lap- s>=3).

model_testing: A script to run the model for a specific race stint, and plot it with Matplotlib to run quick visual sanity checks to ensure baseline correct model behavior
## Methodology

__Naive Model (Deprecated)__
- Each compound is arbitrarily assigned a constant multiplier value for material deg.  
- Fuel loss is added to the net delta (last lap-first lap) in a stint. The multiplier taken to the number of laps is subtracted as well, yielding a final delta which is divided by n_laps to find the thermal deg rate/
-Model was deprecated since the constants are assigned arbitrarily on no logical basis. Using the delta can also produce odd results, as early laps in stints are generally more likely to have slow outlier laps due to driver conflict or cold tires. In general, this model was intended as a scaffolding upon which more complex ideas could be structured.- _ OLS 

__Fitted Model__

 
-  In this model the fuel loss is again initially added to each lap to focus on tire degradation.- Laptimes are fit the the following equation: y=an + bn^2 + c. Where y is laptime, and there are linear and quadratic terms representing thermal and material degradation. Using NumPy lstsq, the laptimes are fit to this equation using a standard OLS fit and constants a, b, and c are fit to minimize deviation from the model. - If the dof>0 whihch is true when n_laps>=3, the residual sum of squares is divided by the degrees of freedom to sigma_sq and subsequently, sigma, which is the standard error.
- The OLS fit allows for two layers of uncertainty. The parameter covariance matrix that is returned estimates the level of confidence in the parameter relationship itself, and the  residual standard error estimates the presence of uncertain events. This means the range of simulation results is directly related to the confidence of the model, which is a useful calibration when working with limited information. 





__Gaussian Noise Generation and Monte Carlo Simulation__
 - Both the naive model and fitted model rely on a Gaussian noise matrix to simulate random race events, although they are scaled differentl
 - In the OLS fitted model, the noise matrix is scaled by the standard error, whereas in the naive model it is scaled arbitrarily. The OLS method is preferred as it estimates the realistic amount of unexplained laptime variance.



