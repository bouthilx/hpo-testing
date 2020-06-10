import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

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


search_space = {
    "svc_c": tune.loguniform(0.001, 0.1)}

analysis = tune.run(
    objective, config=search_space)

print("Best config: ", analysis.get_best_config(metric="error_rate"))
