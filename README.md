# Intertwin-hython


## Description

A python package aims to exploit state-of-the-art hydrological timeseries prediction and forcasting.

The package should supports the **surrogate training**, **parameter learning** and **inference application** components of the drought forecasting InterTwin's use case.

<p align="center">
 <a href="https://github.com/interTwin-eu/hython/"><img src="https://github.com/interTwin-eu/hython/blob/dev/data/static/overview.png" alt="layout"></a>
</p>


## Installation
This package is currently under development.

```
git clone git@gitlab.inf.unibz.it:REMSEN/intertwin-hython.git

```

## Usage
Please review the workflow notebook for a demonstration of the expected inputs, outputs, and how to use the package.


## Support
Please open an issue if you have a bug, feature request or have an idea to improve the package.


## Roadmap


* Domain sampling

Training the model on large domains is time and energy consuming. 
This functionality samples the full domain producing a smaller subsample, with different degree of representativeness based on the sampling strategy, enabling decisions about the trade-off between model performance and computation time. It is likely that good enough performance can be achieved with representative sampling scheme.

Planned strategies: 
    - no sampling (implemented)
    - regular grid sampling (implemented)
    - stratified sampling (coming soon)
    - spatial correlation sampling (coming soon)


* Spatio-temporal validation consisting in (at least) three options: space, time and spacetime. 

This feature generates training and validation sets that allow testing how well the model is performing in extrapolating in different dimensions.


* Simulation of river discharge

Surrogate's simulation of river discharge in addition to soil moisture and evapotranspiration 


* Parameter learning
Calibratig the surrogate

* Add metrics with hydrological meaning 


* Parallel and Distributed ML tasks.


* Uncertainty & Explainable AI


* Model evaluation

Assessing different model architectures and structures




## Contact
For further information please contact:

mohammadhussein.alasawedah@eurac.edu

iacopofederico.ferrario@eurac.edu
