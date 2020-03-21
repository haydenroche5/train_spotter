import numpy as np


class Detector:
    def __init__(self, model, preprocessors=[]):
        self.model = model
        self.preprocessors = preprocessors

    def detect(self, imgs):
        for p in self.preprocessors:
            imgs = p.preprocess(imgs)

        return np.squeeze(self.model.predict_on_batch(np.stack(imgs)))
