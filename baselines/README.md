# Baselines

## Requirements
The requirements provided are the ones used in a conda environment so you should also use a conda environment before trying to install those dependencies.

To create an environment with the correct dependencies:

``
conda create --name destination --file requirements.txt
``

The run information about each destination prediction model can be found in its own folder.



## Evaluation
To evaluate the models, please go to `destination_predictors/evaluation` and run `evaluation.py`.
This will output the results in the paper.
If you want to evaluate a newly trained baseline, please run it's test script which will produce a `metrics.json`.
Copy this file to `evaluation/jsons` and add this file to the model dict in `evaluation.py`
