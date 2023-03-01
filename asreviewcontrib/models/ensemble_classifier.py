import logging
import random

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier

from asreview.models.classifiers.utils import _set_class_weight


class EnsembleClassifier:
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
        strategy="max",  # mean, max
    ):
        self.alpha = alpha
        self.C = C
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.models = models
        self.strategy = strategy
        if "nb" in models:
            self._model_nb = MultinomialNB()
            logging.debug(self._model_nb)
            
        if "logistic" in models:
            self._model_lr = LogisticRegression(
                solver="liblinear",
                C=C,
                class_weight=_set_class_weight(class_weight),
                n_jobs=n_jobs,
                random_state=random_state,
            )
            logging.debug(self._model_lr)
            
        if "rf" in models:
            self._model_rf = SKRandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features,
                class_weight=_set_class_weight(class_weight),
                random_state=random_state,
            )
            logging.debug(self._model_rf)

    def fit(self, X, y):
        """Fit the model to the data."""
        if "nb" in self.models:
            self._model_nb.fit(X, y)
        if "logistic" in self.models:
            self._model_lr.fit(X, y)
        if "rf" in self.models:
            self._model_rf.fit(X, y)

    def predict_proba(self, X):
        """Get the inclusion probability for each sample."""
        predictions = np.zeros((X.shape[0], 2))
        if self.strategy == "mean":
            n_models = 0
            if "nb" in self.models:
                predictions_nb = self._model_nb.predict_proba(X)
                predictions += predictions_nb
                n_models += 1
            if "logistic" in self.models:
                predictions_lr = self._model_lr.predict_proba(X)
                predictions += predictions_lr
                n_models += 1
            if "rf" in self.models:
                predictions_rf = self._model_rf.predict_proba(X)
                predictions += predictions_rf
                n_models += 1

            predictions /= n_models

        if self.strategy == "max":
            if "nb" in self.models:
                predictions_nb = self._model_nb.predict_proba(X)
                predictions = np.maximum(predictions, predictions_nb)
            if "logistic" in self.models:
                predictions_lr = self._model_lr.predict_proba(X)
                predictions = np.maximum(predictions, predictions_lr)
            if "rf" in self.models:
                predictions_rf = self._model_rf.predict_proba(X)
                predictions = np.maximum(predictions, predictions_rf)

        if self.strategy == "multiply":
            predictions = np.ones((X.shape[0], 2))
            if "nb" in self.models:
                predictions_nb = self._model_nb.predict_proba(X)
                predictions = np.multiply(predictions, predictions_nb)
            if "logistic" in self.models:
                predictions_lr = self._model_lr.predict_proba(X)
                predictions = np.multiply(predictions, predictions_lr)
            if "rf" in self.models:
                predictions_rf = self._model_rf.predict_proba(X)
                predictions = np.multiply(predictions, predictions_rf)

        if self.strategy == "random":
            random_choice = random.choice(self.models)
            if random_choice == "nb":
                predictions = self._model_nb.predict_proba(X)
            if random_choice == "logistic":
                predictions = self._model_lr.predict_proba(X)
            if random_choice == "rf":
                predictions = self._model_rf.predict_proba(X)

        return predictions
