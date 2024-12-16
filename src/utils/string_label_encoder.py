import numpy as np

class StringLabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.class_to_index = None
        self.index_to_class = None

    def fit(self, y):
        self.classes_ = np.unique(y)
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes_)}
        self.index_to_class = {i: cls for i, cls in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return np.array([self.class_to_index[cls] for cls in y])

    def inverse_transform(self, y):
        return np.array([self.index_to_class[i] for i in y])