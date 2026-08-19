# ClimateCritters
A menagerie of minimal paleoclimate models and the scars that taphonomy carves on their hides.

## Rationale
Much of the [strong inference](https://www.science.org/doi/10.1126/science.146.3642.347) in modern biology derives from the study of model organisms: non-human species extensively studied to understand fundamental biological phenomena. These findings often translate to other organisms, including humans. Climate science is bereft of organisms but rich in models. However, the emphasis is often on building the latest, greatest, most comprehensive model out there, which makes it challenging to experiment with and understand behavior. Thus, despite [influential calls to explore this framework](https://doi.org/10.1175/BAMS-86-11-1609) (see also [Polvani et al, (2017)](https://eos.org/opinions/when-less-is-more-opening-the-door-to-simpler-climate-models)), model organisms (model models?) are still lacking in climate science. Another issue facing the study of past climates is that, prior to the instrumental era (CE 1850 or so), the records we have of them are often blurred, noisy, sparse and fragmentary. 

Because climate is capable of abrupt jumps, climate science pioneer Wally Broecker nicknamed it "The Angry Beast", and argued that our use of fossil fuels was akin to poking at this beast with sticks.  The purpose of `ClimateCritters` is to gather a collection of model "organisms" illuminating key aspects of climate dynamics (chaos, multiple equilibria, intermittency, tipping points), and how this behavior gets recorded in paleoclimate archives like ice or sediment cores. The core design principle is to code existing, simple models within a unified, object-oriented Python interface that makes it easy to experiment with those models, including:
- exploring model sensitivity via parameter sweeps or forcing scenarios
- exploring taphonomic effects like observational noise, bioturbation or age errors
- comparing the appropriateness of various timeseries analysis methods (e.g. causal analysis, tipping point detection) on well-understood models exhibiting nonlinear behavior.

## Climate Models
`ClimateCritters` gathers models spanning energy balance models, box models, low-dimensional chaotic systems, pendulums/oscillators, and individual models drawn from the paleoclimate literature. The full, up-to-date catalogue — with descriptions and API links for each model — lives in the [Model Catalogue](http://linked.earth/ClimateCritters/latest/get-started/models.html).

## General Structure
Every model in `ClimateCritters` shares a common `Model` / `Forcing` / `Output` interface. See [Core Concepts](http://linked.earth/ClimateCritters/latest/get-started/concepts.html) in the docs for the full explanation.

## Time-varying parameters
Model parameters can be constants, callables, or `cc.Forcing` objects. This enables time-varying
or state-dependent parameters with a consistent API across models.

Example:
```python
lorenz = cc.Lorenz63(
    sigma=lambda t, x, m: 10 + 2*np.sin(t/5),
    rho=lambda t: 28 + 5*np.sin(t/20),
    beta=8/3,
)
```
## Citation
If you use `ClimateCritters` in your work, please cite it. Citation metadata
lives in [`CITATION.cff`](CITATION.cff) — GitHub's "Cite this repository"
button in the sidebar reads it automatically, and it will also be used to
mint a Zenodo DOI on release.

<!-- Once the repo is connected to Zenodo and a first release is cut, replace
this line with the DOI badge, e.g.:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->
