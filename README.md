# PaleoBeasts
A bestiary of minimal paleoclimate models and the scars that taphonomy carves on their hides.

## Rationale
Much of the progress in [strong inference](https://www.science.org/doi/10.1126/science.146.3642.347) in modern biology derives from the study of model organisms: non-human species extensively studied to understand fundamental biological phenomena. Then findings often translate to other organisms, including humans. These model organisms (model models?) are still lacking the climate world of climate modeling, despite [influential calls to explore this framework](https://doi.org/10.1175/BAMS-86-11-1609). The purpose of `PaleoBeasts` is to gather a collection of existing, simple models illuminating key aspects of climate dynamics (chaos, multiple equilibria, intermittency, etc) using a unified interface that makes it easy to experiment with those models, including:
- exploring model sensitivity via parameter sweeps or forcing scenarios
- exploring taphonomic effects like observational noise, bioturbation or age errors
- comparing the appropriateness of various timeseries analysis methods (e.g. causal analysis, tipping point detection) on well-understood models exhibiting nonlinear behavior.

## Climate Models
1. **Lorenz63**: Ed Lorenz's seminal paper, soberly titled [_Deterministic Nonperiodic Flow_](https://doi.org/10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2) singlehandedly birthed chaos theory into existence. Though first intended to model Rayleigh-Bénard convection, this 3 equation, 3 variable model has become the paragon of a chaotic system. Now solvable nearly instantly using standard [ODE](https://en.wikipedia.org/wiki/Ordinary_differential_equation) solvers, the Lorenz63 model is an incredibly versatile and acccessible tool to understand not only [weather prediction](https://doi.org/10.1175/1520-0477(1993)074<0049:ERAPAT>2.0.CO;2), but also the period-doubling route to chaos, or even [climate change](https://doi.org/10.1002/j.1477-8696.1993.tb05802.x).
2. **Ganopolski2024** [Ganopolski (2024)](https://doi.org/10.5194/cp-20-151-2024) proposed a minimal model to simulate glacial cycles ... (to be completed by Jordan)
3. **DO25**  [Melcher et al. (2025)](https://doi.org/10.5194/cp-21-115-2025)  proposed a conceptual model for Dansgaard–Oeschger event dynamics ..(to be completed by Maryam) 



## General Structure:

- **Signal**: this class will generate a pure (noise-free) series based on a given model. For instance model='Ganopolski2024.Model3' or  'Ganopolski2024.MiM'. Need to think about how to pass inititial conditions, boundary conditions, and model parameters. Giant dictionary exported to a yml file for traceability?  Signal should be a Pyleoclim series (trivial to export to csv, if one needs to). 
- **Noise**: this class will add various noise colors to the signals. Should also use pyleo.utils.tsmodel.random_time_index() to mimic the time sampling process, and create irregularly sampled series with reasonable characteristics.

