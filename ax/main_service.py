import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

from ax.service.managed_loop import optimize
from ax.service.ax_client import AxClient


def objective(config):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    svc_c = config["svc_c"]
    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return {'error_rate': (1 - accuracy, 0)}


ax_client = AxClient()

ax_client.create_experiment(
    name="test",
    parameters=[
        {
            "name": "svc_c",
            "type": "range",
            "bounds": [0.001, 0.1],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": True,  # Optional, defaults to False.
        }
    ],
    objective_name="error_rate",
    minimize=True,  # Optional, defaults to False.
    # total_trials=30, # Optional.
)

for i in range(25):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=objective(parameters))

best_parameters, (means, covariances) = ax_client.get_best_parameters()
print(best_parameters)
print(means)
print(covariances)
