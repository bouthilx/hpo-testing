import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

import numpy as np
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO


def objective(config):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    svc_c = config["svc_c"]
    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


cs = ConfigurationSpace()
C = UniformFloatHyperparameter("svc_c", 0.001, 1.0, log=True)
cs.add_hyperparameters([C])

scenario = Scenario(
    {"run_obj": "quality",  # we optimize quality (alternatively runtime)
     "runcount-limit": 50,  # max. number of function evaluations;
    })

# Traceback (most recent call last):
#   File "main.py", line 38, in <module>
#     tae_runner=objective)
#   File "/home/bouthilx/.virtualenvs/smac/lib/python3.6/site-packages/smac/facade/smac_hpo_facade.py", line 41, in __init__
#     if len(scenario.cs.get_hyperparameters()) <= 40:
# AttributeError: 'NoneType' object has no attribute 'get_hyperparameters'
# How should be use SMAC4HPO? Outdated doc? T-T

smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                tae_runner=objective)

incumbent = smac.optimize()

# We can also validate our results (though this makes a lot more sense with instances)
smac.validate(
    config_mode='inc',  # We can choose which configurations to evaluate
    repetitions=100,  # Ignored, unless you set "deterministic" to "false" in line 95
    n_jobs=1)  # How many cores to use in parallel for optimization
