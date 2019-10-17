from training_regime import TrainingRegime
from parser import parse_file


def run(filename):
    """Runs the interpreter on the data in the file with the provided filename."""
    directives = parse_file(filename)
    training_regime = TrainingRegime(directives)
    training_regime.run()

