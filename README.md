# PaleoBeasts
A bestiary of minimal paleoclimate models and the scars that taphonomy carves on their hides

General Structure:

- **Signal**: this class will generate a pure (noise-free) series based on a given model. For instance model='Ganopolski2024.Model3' or  'Ganopolski2024.MiM'. Need to think about how to pass inititial conditions, boundary conditions, and model parameters. Giant dictionary exported to a yml file for traceability?  Signal should be a Pyleoclim series (trivial to export to csv, if one needs to). 
- **Noise**: this class will add various noise colors to the signals. Should also use pyleo.utils.tsmodel.random_time_index() to mimic the time sampling process, and create irregularly sampled series with reasonable characteristics.

 Jordan is on the signal, Alex is on the noise.  

## Time-varying parameters
Model parameters can be constants, callables, or `pb.core.Forcing` objects. This enables time-varying
or state-dependent parameters with a consistent API across models.

Example:
```python
lorenz = pb.signal_models.Lorenz63(
    forcing=pb.core.Forcing(lambda t: 0.0),
    sigma=lambda t, x, m: 10 + 2*np.sin(t/5),
    rho=pb.core.Forcing(lambda t: 28 + 5*np.sin(t/20)),
    beta=8/3,
)
```
