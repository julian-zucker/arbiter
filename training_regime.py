import csv
import pickle
import tensorflow as tf
import numpy as np


class TrainingRegime:
    """A TrainingRegime is a full specification of a series of steps to take to train and validate
    a machine learning model."""

    # TODO model explanation, will require new dataset

    def __init__(self, directives):
        self.from_data = directives["from_data"]
        self.model_type = directives["train_a"]
        self.label_column_name = directives["predicting"]
        self.write_model_to = directives["write_model_to"]
        self.protected_class_column_name = directives["protected_classes"]
        self.required_fairness = directives["required_fairness"]
        self.explanation = directives["explanation"]

    def run(self):
        data, fieldnames = _read_data(self.from_data)
        tabular_data, labels = _tabular_data(data, fieldnames, self.label_column_name)
        protected_class_index = fieldnames.index(self.protected_class_column_name)
        model = _preprocess_data(
            data,
            self.protected_class_column_name,
            protected_class_index,
            self.label_column_name,
            self.required_fairness,
        )
        # model = _train_model(train_data, labels, self.model_type)
        # _check_fairness(tabular_data, model, protected_class_index, self.required_fairness)
        _write_model(model, self.write_model_to)


def _read_data(csv_filename):
    def try_float(v):
        try:
            return float(v)
        except:
            return v

    with open(csv_filename) as fd:
        reader = csv.DictReader(fd)
        return [{k: try_float(v) for k, v in row.items()} for row in reader], reader.fieldnames


def _train_model(features, labels, model_type):
    if model_type != "decision tree":
        raise ValueError("Can only train decision trees")

    from sklearn.model_selection import train_test_split

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels)

    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier().fit(features_train, labels_train)
    predictions = model.predict(features_test)
    print(f"Accuracy: {sum(predictions == labels_test) / float(len(features_test))}")
    return model


def _tabular_data(data, fieldnames, label_column_name):
    features = []
    labels = []

    for row in data:
        labels.append(row[label_column_name])

        features_in_order = []
        for fieldname in fieldnames:
            if fieldname != label_column_name:
                features_in_order.append(row[fieldname])
        features.append(features_in_order)
    return features, labels


def _preprocess_data(
    data, protected_attribute_name, protected_attribute_index, label_name, required_fairness
):
    from pandas import DataFrame
    from aif360.datasets import BinaryLabelDataset

    dataset = BinaryLabelDataset(
        df=DataFrame(data),
        protected_attribute_names={protected_attribute_name},
        label_names={label_name},
        favorable_label=2,
        unfavorable_label=1,
    )
    train, test = dataset.split([0.8])

    from aif360.algorithms.inprocessing import AdversarialDebiasing

    sess = tf.compat.v1.Session()
    debiaser = AdversarialDebiasing(
        unprivileged_groups=({protected_attribute_name: 0},),
        privileged_groups=({protected_attribute_name: 1},),
        scope_name="debiaser",
        debias=True,
        sess=sess,
    )
    debiaser.fit(train)

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(class_weight="balanced")

    X_tr = np.delete(train.features, protected_attribute_index, axis=1)
    y_tr = train.labels.ravel()
    model.fit(X_tr, y_tr)

    test_pred = test.copy(deepcopy=True)
    test_pred.scores = model.predict(np.delete(debiaser.predict(test).features, protected_attribute_index, axis=1))

    accuracy = np.sum(np.equal(test.scores, test_pred.scores))

    from aif360.metrics import ClassificationMetric
    disparate_impact = ClassificationMetric(
        test,
        test_pred,
        unprivileged_groups=({protected_attribute_name: 0},),
        privileged_groups=({protected_attribute_name: 1},),
    ).disparate_impact()

    print(f"Accuracy: {accuracy}")
    print(f"Disparate impact: {disparate_impact}")
    if disparate_impact > float(required_fairness):
        raise ValueError(
            f"Too unfair! Disparate impact was {disparate_impact} but must be less than {required_fairness}"
        )


def _write_model(model, output_model_filename):
    with open(output_model_filename, "wb") as output_file:
        pickle.dump(model, output_file)
