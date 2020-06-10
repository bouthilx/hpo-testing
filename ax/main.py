import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

from ax.service.managed_loop import optimize


def objective(config):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    svc_c = config["svc_c"]
    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return {'error_rate': (1 - accuracy, 0)}


best_parameters, values, experiment, model = optimize(
    parameters=[
        {
            "name": "svc_c",
            "type": "range",
            "bounds": [0.001, 0.1],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": True,  # Optional, defaults to False.
        }
    ],
    experiment_name="test",
    objective_name="error_rate",
    evaluation_function=objective,
    minimize=True,  # Optional, defaults to False.
    total_trials=30, # Optional.
)

print(best_parameters)
