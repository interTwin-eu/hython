# Intertwin-hython


## Description

The package enables the development of deep learning based surrogates for semi-distributed and distributed hydrological models.


## High level class diagram

<p align="center">
 <a href="https://github.com/interTwin-eu/hython"><img src="https://github.com/interTwin-eu/hython/blob/dev/assets/img/hython.png" alt="layout" width="700" height="500"></a>
</p>

TODOs:
- consider associating downsampler to sampler class. Remove association with dataset class.
- move train function into trainer class
- consider tighter integration betweeen hython and itwinain trainers

## Installation


```bash
git clone https://github.com/interTwin-eu/hython.git

cd ./hython

pip install .

```

## Usage

### Demo

The goal of the demo is to show how to train and evaluate the models offered in hython.

Every package needed to run the notebooks can be installed by running:

```bash

cd ./hython

pip install .[complete]

```

There are two notebooks showcasing the training of a simple LSTM model and a Convolutional LSTM.

```bash
./demo/train_lstm.ipynb
./demo/train_convlstm.ipynb
```

Todo: evaluation notebooks


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


## Contact

For further information please contact:

iacopofederico.ferrario@eurac.edu
