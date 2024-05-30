# Intertwin-hython


## Description

A python package aims to exploit state-of-the-art hydrological timeseries prediction and forcasting.

The package should supports the **surrogate training**, **parameter learning** and **prediction** application components of the interTwin's drought forecasting use case.

<p align="center">
 <a href="https://github.com/interTwin-eu/hython/"><img src="https://github.com/interTwin-eu/hython/blob/dev/data/static/overview.png" alt="layout"></a>
</p>


## Installation

This package is currently under development.


```bash
git clone https://github.com/interTwin-eu/hython.git

cd ./hython

pip install .

```

## Usage

### Demo Jupyter Notebooks 

Please review the workflow notebook for a demonstration of the expected inputs, outputs, and how to use the package.

### Command Line Interface

```bash

python preprocess.py --config preprocessing.yaml

python train.py --config training.yaml

python evaluate.py --config evaluating.yaml

```

## Support
Please open an issue if you have a bug, feature request or have an idea to improve the package.

## Design

Class diagram

## Roadmap

- [x] Predict vertical fluxes and storages 
    - [x] surface soil moisture 
    - [x] evapotranspiration 

- [] Predict streamflow 

- [] Parameter learning

- [] Seasonal forecast

- [] Evaluation 
    - [] Hydrological metrics

- [] Distributed training

- [] Uncertainty & Explainable AI


## Contact

For further information please contact:

iacopofederico.ferrario@eurac.edu
