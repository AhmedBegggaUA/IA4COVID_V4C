# IA4COVID: Predicting the COVID19 pandemic
curve with vaccination

## Introduction
In this paper, we present a deep learning-based approach to
predict the number of daily COVID-19 cases in over 170 countries in the
world, taking into account the non-pharmaceutical interventions (NPIs)
applied in those countries, including vaccination.
We also describe a novel prescriptor of NPIs to minimize the expected
number of COVID-19 cases while minimizing the social and economic
cost of applying such interventions. We empirically validate the proposed
approach for six months between January and June 2021, once vaccination
is available and applied in most countries. Our model outperforms state-of-
the-art methods and opens the door to accurate, scalable, and data-driven
approaches to modeling the pandemic curve.
Within this repository you will find:
* Sample predictors and prescriptors 
* Sample implementations of the "predict" API and the "prescribe" API, which you will be required to implement 
as part of your submission
* Sample IP (Intervention Plan) data to test your submission

## Pre-requisites
To run the examples, you will need:
* A computer or cloud image running a recent version of OS X or Ubuntu (Using Microsoft Windows™.) 
* Your machine must have sufficient resources in terms of memory, CPU, and disk space to train machine learning models 
and run Python programs.
* An installed version of Python, version ≥ 3.6. To avoid dependency issues, we strongly recommend using a standard Python [virtual environment](https://docs.python.org/3/tutorial/venv.html) with `pip` for package management. The examples in this repo assume you are using such an environment.

Having registered for the contest, you should also have:
* A copy of the Competition Guidelines
* Access to the Support Slack channel
* A pre-initialized sandbox within the XPRIZE system

## Examples
Under the `covid_xprize/examples` directory you will find some examples of predictors and prescriptors that you can 
inspect to learn more about what you need to do:
* `predictors/linear` contains a simple linear model, using the 
[Lasso algorithm](https://en.wikipedia.org/wiki/Lasso_(statistics)).
* `predictors/lstm` contains a more sophisticated [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) 
model for making predictions.
* `prescriptors/zero` contains a trivial prescriptor that always prescribes no interventions; 
`prescriptors/random` contains one that prescribes random interventions.
* `prescriptors/neat` contains code for training prescriptors with [NEAT](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)
 
The instructions below assume that you are using a standard Python virtual environment, and `pip` for package 
management. Installations using other environments (such as `conda`) are outside the scope of these steps.

In order to run the examples locally:
1. Ensure your current working directory is the root folder of this repository (the same directory as this README 
resides in). The examples assume your working directory is set to the project root and all paths are relative to 
it.
1. Ensure your `PYTHONPATH` includes your current directory:
    ```shell script
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    ```
1. Create a Python [virtual environment](https://docs.python.org/3/tutorial/venv.html)
1. Activate the virtual environment
1. Install the necessary requirements:
    ```shell script
    pip install -r requirements.txt --upgrade
    ```    
1. Start Jupyter services:
    ```shell script
    jupyter notebook
    ```
    This causes a browser window to launch
1.  Browse to and launch one of the examples (such as `linear`) and run through the steps in the associated 
notebook -- in the case of `v4c`, `predictor_train_no_region.ipynb`.
1. The result should be a trained predictor, and some predictions generated by running the predictor on test data. 
Details are in the notebooks.