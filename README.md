# PaleoBeasts
A bestiary of minimal paleoclimate models and the scars that taphonomy carves on their hides.

## Rationale
Much of the progress in [strong inference]() in modern biology derives from the study of model organisms: non-human species extensively studied to understand fundamental biological phenomena. Then findings often translate to other organisms, including humans. These model organisms (model models?) are still lacking the climate world of climate modeling, despite [influential calls to explore this framework](https://doi.org/10.1175/BAMS-86-11-1609). The purpose of `PaleoBeasts` is to gather a collection of existing, simple models illuminating key aspects of climate dynamics (chaos, multiple equilibria, energy balance, etc) using a unified interface that makes it easy to experiment with those models, including:
- exploring sensitivity via parameter sweeps
- exploring taphonomic effects like observational noise, bioturbation or age errors
- 


General Structure:

- **Signal**: this class will generate a pure (noise-free) series based on a given model. For instance model='Ganopolski2024.Model3' or  'Ganopolski2024.MiM'. Need to think about how to pass inititial conditions, boundary conditions, and model parameters. Giant dictionary exported to a yml file for traceability?  Signal should be a Pyleoclim series (trivial to export to csv, if one needs to). 
- **Noise**: this class will add various noise colors to the signals. Should also use pyleo.utils.tsmodel.random_time_index() to mimic the time sampling process, and create irregularly sampled series with reasonable characteristics.

 Jordan is on the signal, Alex is on the noise.  
