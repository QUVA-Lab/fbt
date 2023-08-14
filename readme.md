# Research virus spread and contact tracing

This repo contains accompanying code for the Master Thesis "Feature Based Testing" by Sander Broos.

## Typical usage

Starting point for experiments will be the following command:

```
python3 fbt/experiments/compare_stats.py \
    --inference_method "dummy" \
    --experiment_setup "prequential" \
    --config_data intermediate_graph_abm_02 \
    --config_model model_IG02
```

Experiments take two configs: one for the model and one for the simulator (data).
Whenever 'abm' is in the data config, the ABM simulator will be used.

Experimental setup could be 'single', where inference will be performed on a single, static graph, or 'prequential',
where an experiment with conditional testing and quarantining will be performed (similar to related research like CRISP and SIB).

## Code convention

Code convention: We care deeply about good code and scientific reproducibility. As of July 2023, the code contains
42 unittests, spanning more than one thousands line of code (`make test` or `nose2 -v`).

The code includes abundant type hints (`make hint` or `pytype fbt`).

Code is styled with included '.pylintrc' and pycodestyle (`make lint` or `pylint fbt`)

## Installation

For GSL, follow [these instructions](https://coral.ise.lehigh.edu/jild13/2016/07/11/hello/)

```
# get the installation file
wget ftp://ftp.gnu.org/gnu/gsl/gsl-latest.tar.gz

# Unpack archive
tar -zxvf gsl-latest.tar.gz

# make a directory for the gsl installation
mkdir /var/scratch/${USER}/projects/gsl

# installation
./configure --prefix=/var/scratch/${USER}/projects/gsl
make
make check
make install
```

[SWIG](https://www.swig.org/) install
```
sudo apt-get update
sudo apt-get -y install swig
```

ABM install
```
# Get the ABM code
cd ../

mkdir abm
cd abm

git clone https://github.com/aleingrosso/OpenABM-Covid19.git .

cd src
make all

make swig-all
```

## Run a sweep with WandB
To run a sweep with WandB, run the following command

`$ wandb sweep sweep/default.yaml`

Copy the sweepid. Then on the cluster, or another computer, start up an agent with

```
$ export SWEEP=sweepid
$ wandb agent "$USERNAME/fbt-fbt_experiments/$SWEEP"
```

## Analysis of results
Some analysis plots of example results can be found in the file testing_policy_analysis.ipynb.

## Attribution

This repo continued from a fork of upstream repo [NTTW](https://github.com/QUVA-Lab/nttw)