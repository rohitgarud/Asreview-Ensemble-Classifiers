from asreview.models.classifiers.base import BaseTrainClassifier

from asreviewcontrib.models.ensemble_classifier import EnsembleClassifier


class EnsembleNBLRClassifier(BaseTrainClassifier):
    """Ensemble of Naive Bayes and Logistic Regression classifier"""

    name = "ensemble_nb_lr"
    label = "Ensemble NB-LR"

    def __init__(
        self,
        alpha=3.822,
        C=1.0,
        class_weight=1.0,
        random_state=None,
        n_jobs=1,
        models=["nb", "logistic"],
        strategy="multiply",  # mean, max, multiply, random
    ):

        super(EnsembleNBLRClassifier, self).__init__()
        self.alpha = alpha
        self.C = C
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.models = models
        self.strategy = strategy
        self._model = EnsembleClassifier(
            alpha=alpha,
            C=C,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            models=models,
            strategy=strategy,
        )

    def fit(self, X, y):
        """Fit the model to the data."""
        self._model.fit(X, y)


class EnsembleNBRFClassifier(BaseTrainClassifier):
    """Ensemble of Naive Bayes and Random Forest classifier"""

    name = "ensemble_nb_rf"

    def __init__(
        self,
        alpha=3.822,
        class_weight=1.0,
        random_state=None,
        n_estimators=100,
        max_features=10,
        models=["nb", "rf"],
        strategy="multiply",  # mean, max, multiply, random
    ):

        super(EnsembleNBRFClassifier, self).__init__()
        self.alpha = alpha
        self.class_weight = class_weight
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.models = models
        self.strategy = strategy
        self._model = EnsembleClassifier(
            alpha=alpha,
            random_state=random_state,
            n_estimators=n_estimators,
            max_features=max_features,
            models=models,
            strategy=strategy,
        )

    def fit(self, X, y):
        """Fit the model to the data."""
        self._model.fit(X, y)


class EnsembleLRRFClassifier(BaseTrainClassifier):
    """Ensemble of Logistic Regression and Random Forest classifier"""

    name = "ensemble_lr_rf"

    def __init__(
        self,
        C=1.0,
        class_weight=1.0,
        random_state=None,
        n_jobs=1,
        n_estimators=100,
        max_features=10,
        models=["logistic", "rf"],
        strategy="multiply",  # mean, max, multiply, random
    ):

        super(EnsembleLRRFClassifier, self).__init__()
        self.C = C
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.models = models
        self.strategy = strategy
        self._model = EnsembleClassifier(
            C=C,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            max_features=max_features,
            models=models,
            strategy=strategy,
        )

    def fit(self, X, y):
        """Fit the model to the data."""
        self._model.fit(X, y)


class EnsembleNBLRRFClassifier(BaseTrainClassifier):
    """Ensemble of Naive Bayes, Logistic Regression and Random Forest classifier"""

    name = "ensemble_nb_lr_rf"

    def __init__(
        self,
        alpha=3.822,
        C=1.0,
        class_weight=1.0,
        random_state=None,
        n_jobs=1,
        n_estimators=100,
        max_features=10,
        models=["nb", "logistic", "rf"],
        strategy="multiply",  # mean, max, multiply, random
    ):

        super(EnsembleNBLRRFClassifier, self).__init__()
        self.alpha = alpha
        self.C = C
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.models = models
        self.strategy = strategy
        self._model = EnsembleClassifier(
            alpha=alpha,
            C=C,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            max_features=max_features,
            models=models,
            strategy=strategy,
        )

    def fit(self, X, y):
        """Fit the model to the data."""
        self._model.fit(X, y)
