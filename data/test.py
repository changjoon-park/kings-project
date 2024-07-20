from dataclasses import dataclass, field
from sklearn.metrics import confusion_matrix
from typing import Any


@dataclass
class Model:
    name: str
    model: Any
    accuracy: float = field(init=False, default=None)
    precision: float = field(init=False, default=None)
    recall: float = field(init=False, default=None)
    f1: float = field(init=False, default=None)
    cm: np.ndarray = field(init=False, default=None)
    roc_auc: float = field(init=False, default=None)
    fpt: float = field(init=False, default=None)
    tpr: float = field(init=False, default=None)
    precision_curve: float = field(init=False, default=None)
    recall_curve: float = field(init=False, default=None)
    average_precision: float = field(init=False, default=None)

    def __str__(self):
        return self.name

    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        self.accuracy, self.precision, self.recall, self.f1 = calculate_metrics(
            y_test, predictions
        )
        self.cm = confusion_matrix(y_test, predictions)

        y_scores = self.get_prediction_scores(X_test)
        (
            self.fpr,
            self.tpr,
            self.roc_auc,
            self.precision_curve,
            self.recall_curve,
            self.average_precision,
        ) = calculate_roc_pr(y_test, y_scores)

    def get_prediction_scores(self, X_test):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)[:, 1]
        elif hasattr(self.model, "decision_function"):
            return self.model.decision_function(X_test)
        else:
            raise AttributeError(
                "Model has neither predict_proba nor decision_function method"
            )

    def print_result(self):
        print_data_info(X_train, X_test, y_train, y_test, self.name)
        print_metrics(
            self.name,
            self.accuracy,
            self.precision,
            self.recall,
            self.f1,
            self.cm,
            self.roc_auc,
            self.average_precision,
        )
        plot_results(
            self.name,
            self.fpr,
            self.tpr,
            self.roc_auc,
            self.precision_curve,
            self.recall_curve,
            self.average_precision,
            self.cm,
            coef=None,
            importances=None,
            feature_names=None,
        )
