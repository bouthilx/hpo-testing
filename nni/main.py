import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

import nni


def objective(config):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    svc_c = config["svc_c"]
    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    nni.report_final_result(1 - accuracy)
    return 1 - accuracy


# search_space = {
#     "svc_c": 'loguniform(0.001, 0.1)'}


if __name__ == '__main__':
    objective(nni.get_next_parameter())
