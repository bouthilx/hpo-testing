import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

import hyperopt as hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray import tune


def objective(config):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    svc_c = config["svc_c"]
    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    # tune.report(error_rate=1 - accuracy)
    tune.track.log(error_rate=1 - accuracy)
    return 1 - accuracy

# Create a HyperOpt search space
search_space = {
    "svc_c": hp.hp.loguniform('svc_c', 0.001, 0.1)}

# Specify the search space and maximize score
hyperopt = HyperOptSearch(search_space, metric="error_rate", mode="min")

# Execute 20 trials using HyperOpt and stop after 20 iterations
analysis = tune.run(
    objective,
    name='test-hyperopt',
    search_alg=hyperopt,
    num_samples=20,
    stop={"training_iteration": 20}
)

print("Best config: ", analysis.get_best_config(metric="error_rate"))
