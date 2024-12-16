import numpy as np

class SlitlessPreprocessor:
    def __init__(self, a=1, b=99):
        self.a, self.b = a, b
        self.x_min, self.x_max = None, None

    def fit(self, x):
        self.x_min, self.x_max = np.percentile(x, [self.a, self.b])
        return self

    def transform(self, x):
        x_clip = np.clip(x, self.x_min, self.x_max)
        x_minmax = (x_clip - self.x_min) / (self.x_max - self.x_min)
        return x_minmax