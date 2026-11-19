"""k-NN, SVM and MLP classifiers on landmark feature vectors (Mustafa).

Every classifier is a sklearn Pipeline(StandardScaler -> model), so the scaler is fit on the
training data only. Hyperparameters are tuned (optional) with GroupKFold on the TRAIN split,
grouped by (letter, block), never on the test set.
"""
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSIFIERS = ["knn", "svm", "mlp"]

DEFAULTS = {
    "knn": dict(n_neighbors=5, weights="distance"),
    "svm": dict(kernel="rbf", C=10.0, gamma="scale"),
    "mlp": dict(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=300, random_state=0),
}

GRIDS = {
    "knn": {"clf__n_neighbors": [3, 5, 9, 15]},
    "svm": {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01]},
    "mlp": {"clf__hidden_layer_sizes": [(64,), (128, 64)], "clf__alpha": [1e-4, 1e-2]},
}


def make_classifier(name: str, **params) -> Pipeline:
    if name not in CLASSIFIERS:
        raise ValueError(f"unknown classifier {name!r}; choose from {CLASSIFIERS}")
    cfg = {**DEFAULTS[name], **params}
    model = {"knn": KNeighborsClassifier, "svm": SVC, "mlp": MLPClassifier}[name](**cfg)
    return Pipeline([("scale", StandardScaler()), ("clf", model)])


def fit_classifier(name: str, X, y, groups=None, tune: bool = False, n_splits: int = 4):
    """Fit on (X, y). With tune=True, grid-search on `groups`-aware CV folds inside the
    training data. Returns (fitted pipeline, best_params dict or {})."""
    pipe = make_classifier(name)
    if not tune:
        return pipe.fit(X, y), {}
    if groups is None:
        raise ValueError("tune=True needs groups (letter+block ids) to avoid leakage")
    search = GridSearchCV(pipe, GRIDS[name], cv=GroupKFold(n_splits=n_splits),
                          scoring="accuracy", n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_
