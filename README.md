### untitledlanguage

### Goal
A declarative language for specifying machine learning models, in a way that helps the practice of ethical machine learning.

### Code sample
```
FROM DATA "credit_data.csv"
TRAIN A decision tree
PREDICTING "credit_decision"
WRITE MODEL TO "credit_score.model"
PROTECTED CLASSES "race", "gender", "age"
REQUIRED FAIRNESS (disparate impact < 1.1)
EXPLANATION "decision_reason"
```

### Known limitations
This implementation basically only supports the above code sample. Very little else is supported, for now.