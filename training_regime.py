import csv
import pickle


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
        model = _train_model(tabular_data, labels, self.model_type)
        _check_fairness(tabular_data, model, fieldnames.index(self.protected_class_column_name), self.required_fairness)
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
    print(f"Accuracy: {sum(predictions == labels_test) / len(features_test)}")
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


def _check_fairness(data, model, protected_class_index, required_fairness):
    priv_data = []
    non_priv_data = []
    for row in data:
        if row[protected_class_index] == 1:
            priv_data.append(row)
        else:
            non_priv_data.append(row)

    print(sum(model.predict(priv_data) - 1))
    disparate_impact = float(sum(model.predict(priv_data) - 1) / sum(model.predict(non_priv_data) - 1))
    print(f"Disparate impact: {disparate_impact}")
    if disparate_impact > float(required_fairness):
        raise ValueError("Too unfair!")

def _write_model(model, output_model_filename):
    with open(output_model_filename, "wb") as output_file:
        pickle.dump(model, output_file)
