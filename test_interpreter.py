import csv
import os
import pickle

import interpreter


def test_interpreter():
    interpreter.run("test_fixtures/sample_program.untitledlanguage")

    with open("test_fixtures/credit_data.csv") as datafile:
        datafile.readline() # skip header
        reader = csv.reader(datafile)
        datapoint = next(reader)[:-1] # skip label

    with open("test_fixtures/credit_score.model", "rb") as modelfile:
        model = pickle.load(modelfile)
        print(model)
        assert(model.predict([datapoint]) in [1, 2])

    os.remove("test_fixtures/credit_score.model")
